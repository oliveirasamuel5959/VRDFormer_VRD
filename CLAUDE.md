# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research implementation of **VRDFormer** (CVPR 2022) — end-to-end video visual relation detection with
transformers, evaluated on ImageNet-VidVRD and VidOR. The codebase is forked from DETR /
Deformable-DETR / TrackFormer, so much of `models/` and `util/` is DETR/TrackFormer lineage adapted
from single-object queries to **subject-object pair** queries (every prediction head is duplicated
into `sub_*` / `obj_*`, plus a multi-label `verb_*` head).

There is no test suite, linter config, or CI. Verification means running training/eval on real data.

## Commands

Install (Python 3.7, PyTorch 1.10+cu111 — see `docs/INSTALL.md`):

```bash
pip install -r docs/requirements.txt
pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'
cd models/ops && sh make.sh        # MultiScaleDeformableAttention; only needed for --deformable
```

Data preparation (`docs/DATA.md`). **`data/prepare.py` resolves `metadata/` and `<dbname>/action.txt`
relative to CWD, so run it from inside `data/`** — otherwise it writes the pickles where the datasets
can't find them (they load `data/metadata/...` relative to repo root):

```bash
cd data
python prepare.py --func prep_vidor --root_dir <root>          # reorganize raw VidOR tree
python prepare.py --func get_anno --dbname vidvrd              # -> data/metadata/<db>_annotations.pkl
python prepare.py --func get_fid --dbname vidvrd --split train --stage 1 --timestep 1 --minmax_dur 24
python prepare.py --func get_fid --dbname vidvrd --split val   --timestep 1 --minmax_dur 24
```

Training — always through `torch.distributed.launch`, even on one GPU (see gotcha below). The
`scripts/` wrappers are the canonical invocations:

```bash
sh scripts/stage1/train.sh              # VidVRD, DETR backbone, 8 GPUs
sh scripts/stage2/train.sh              # VidVRD stage 2, 1 GPU
sh scripts/stage1/train_deform.sh       # deformable variants
sh scripts/stage1/train_vidor.sh        # VidOR
```

Each script is a thin wrapper around:

```bash
python -m torch.distributed.launch --master_port 47749 --nproc_per_node=8 main.py \
    --accumulate_steps 1 --lr_backbone 1e-5 --lr 5e-5 --num_queries 200 \
    --dataset_config configs/vidvrd_stage1.json
```

Evaluation only (stage 2 only):

```bash
python -m torch.distributed.launch --nproc_per_node=1 main.py --eval \
    --dataset_config configs/vidvrd_stage2.json --resume data/ckpts/.../checkpoint.pth
```

Quick smoke run: add `--debug` (forces `num_workers=0`, skips raw-annotation loading and the
zero-shot triplet diff), or use `configs/vidorpart_stage1.json` which caps the dataset at 100 videos.

## Configuration model

`main.py` parses argparse defaults, then **`--dataset_config <json>` is loaded and overwrites
`vars(args)` wholesale** (`main.py:143-148`). The JSON wins over anything passed on the command line,
so to change `batch_size`, `epochs`, `seq_len`, `pretrain`, `output_dir`, `deformable`, etc., edit the
config in `configs/`, not the CLI flags. Some keys used at runtime (`cautious`, `by_ratio`) exist
*only* in the config JSONs and have no argparse default — a config missing them crashes in
`make_video_transforms`.

Configs also hardcode absolute dataset paths (`/home/zhengsipeng/data/...`) and checkpoint paths
under `data/weights/` and `data/ckpts/`. Expect to rewrite these for any new machine.

`num_obj_classes` / `num_verb_classes` are **ignored** from args and re-derived from the dataset name
in `models/__init__.py:23-24` (vidor → 80/50, else 35/132).

## Two-stage architecture

The whole pipeline is selected by `stage` (1 or 2) in the config; it switches the dataset `__getitem__`,
the model class, the criterion, and the training loop simultaneously.

**Stage 1 — pair detection + tracking** (`models/vrdformer.py`, `models/vrdformer_track.py`,
`engine.py:train_stage1`)

- Dataset yields a frame pair: `prepare_data_stage1` samples `(frame_id, post_frame_id)` and packs the
  earlier frame into `target['prev_image'] / target['prev_target']`.
- `VRDFormerTracking` (mixin over `TrackingBase` + `VRDFormer`) runs the previous frame under
  `no_grad`, Hungarian-matches it (`models/matcher.py`), and `add_track_queries_to_targets` writes
  `track_query_hs_embeds`, `track_query_{sub,obj}_boxes`, `track_queries_mask`,
  `track_queries_fal_pos_mask`, `track_query_match_ids` back into the targets.
- The transformer prepends those embeddings to the learned static queries
  (`models/transformer.py:169-176`), so the decoder input is `[recurrent queries | static queries]`.
  Query-target matching in `SetCriterionTrack` is therefore *partly forced* (track queries must match
  their known target index) and only the rest goes through the matcher.
- Note `TrackingBase.forward` calls `super().forward(..., stage=2)` — that argument is vestigial and
  not the config `stage`.

**Stage 2 — relation classification over time** (`models/vrdformer_stage2.py`, `engine.py:train_stage2`)

- No tracking, no matcher, no box losses. Queries are **initialized from ground-truth boxes**:
  `Transformer.extract_roi_feat` does `roi_align` on the encoder input at `unscaled_{sub,obj}_boxes`,
  fuses s/o features through `so_linear`, and `prepare_tag_query` pads to `num_queries`.
- `batch_size` must be 1: the training loop iterates frames of a single clip
  (`samples.select_frame(fid)` for `fid in range(seq_len)`) and threads a `memory` dict through the
  frames.
- `memory` is keyed by `"<sub_tid>-<obj_tid>"` and accumulates per-frame `rel_embed`, `s_embed`,
  `o_embed`, labels, and (eval only) `frame_ids`. At `eos` the whole sequence is mean-pooled per pair
  and pushed through `relation_classifier` → one loss for the clip.
- Loss is computed once per clip on the accumulated memory, not per frame.

Shared plumbing: `models/__init__.py:build_model` is the single place that wires backbone +
(deformable or vanilla) transformer + model class + criterion + `weight_dict` per stage.
`--deformable` swaps `models/transformer.py` for `models/deformable_transformer.py` and requires
`num_feature_levels > 1` and the compiled `models/ops` extension; non-deformable asserts
`num_feature_levels == 1`.

## Data pipeline

`datasets/dataset.py:VRDBase` holds nearly all logic; `datasets/vidvrd.py` and `datasets/vidor.py`
are thin subclasses that only set `num_verb_classes` (132 / 50), validate the annotation `version`
field, and provide `build_dataset`. `datasets/__init__.py:build_dataset` dispatches on
`args.dataset == "vidvrd"` else VidOR — so `vidorpart` routes to the VidOR class.

Three input artifacts per dataset:

- `<data_dir>/videos/<video_id>.mp4` — frames decoded on the fly with **decord**.
- `<data_dir>/annotations/{train,val}/*.json` — raw VidVRD/VidOR annotations; also the source of
  `self.video_ids` and of `get_relation_insts` used for evaluation ground truth.
- `data/metadata/<db>_annotations.pkl` and `data/metadata/<db>_{split}_frames[_stage<N>].json` —
  produced by `data/prepare.py`. The frames JSON holds `train_begin_fids` + `durations` for train, or
  a per-video positive-frame bitmap for val.

Boxes are normalized to `[0,1]` when raw annotations are loaded, converted to cxcywh by the
transforms, and stage 2 additionally materializes `unscaled_{sub,obj}_boxes` in xyxy pixel space for
`roi_align`. `datasets/video_transforms.py` operates on lists of frames with per-frame target lists.

Evaluation (`util/evaluate.py`) reports relation *detection* (mean AP, recall@50/100) and *tagging*
(precision@1/5/10), each under overall / zero-shot / generalized-zero-shot settings.
`dataset_val.zeroshot_triplets` is computed in `dataloader_initializer` as val-triplets minus
train-triplets — which means **building the val loader loads and scans the full train annotation set**
(slow), unless `--debug`.

Dead code inherited from TrackFormer: `datasets/tracking/`, `datasets/coco.py`,
`datasets/coco_eval.py`, `datasets/crowdhuman.py` are unreachable from `build_dataset` and reference
args that no longer exist (`args.crowdhuman_path`, `args.coco_and_crowdhuman_prev_frame_rnd_augs`).

## Known rough edges

Research code, mid-refactor. These bite immediately, so check before assuming a bug is yours:

- **Stage 1 has no eval path.** `main.py` only imports `eval_stage2` when `stage == 2`, but calls
  `eval_one_epoch` unconditionally after each epoch (`main.py:224`) → `NameError` at the end of
  stage-1 epoch 0. Same for `--eval --stage 1`.
- **DDP launcher is mandatory.** `engine.py:229` calls `model.module.relation_classifier(...)`, which
  only exists when the model is DDP-wrapped. Running `python main.py` directly fails in eval.
- `util/checkpoints.py:resume_value_deformable` contains live `import pdb;pdb.set_trace()` calls on
  several shape-mismatch branches; loading a deformable pretrain can drop into the debugger.
  `models/vrdformer.py:159` has one too.
- `engine.py:18` calls `datetime.now()` while only the `datetime` *module* is imported — the NaN-loss
  path itself raises.
- `configs/vidor_stage1_deform.json` sets `"dataset": "vidvrd"` and `vidvrd_path` pointing at the
  vidor directory; the other VidOR configs mix `vidor_path` and `vidvrd_path` inconsistently while
  `datasets/vidvrd.py` reads `args.vidvrd_path` and `datasets/vidor.py` reads `args.vidor_path`.
- Argparse mixes separators: `--output-dir`, `--start-epoch`, `--world-size` use hyphens; everything
  else uses underscores.
- The working tree currently shows every tracked file as modified — that is a CRLF line-ending
  conversion, not real content change. Use `git diff --stat` / `git diff -w` before concluding
  anything about local edits.
