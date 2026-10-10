# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research implementation of **VRDFormer** (CVPR 2022) — end-to-end video visual relation detection with
transformers, evaluated on ImageNet-VidVRD and VidOR. The codebase is forked from DETR /
Deformable-DETR / TrackFormer, so much of `models/` and `util/` is DETR/TrackFormer lineage adapted
from single-object queries to **subject-object pair** queries (every prediction head is duplicated
into `sub_*` / `obj_*`, plus a multi-label `verb_*` head).

The repo targets Python 3.10 (`.python-version`). `pyproject.toml` declares PyTorch 2.5.1+ and Ruff;
the legacy install instructions in `docs/INSTALL.md` and `docs/requirements-legacy.txt` describe the
original, older Python 3.7 / PyTorch 1.10 setup and should be treated as historical where they conflict. There is no general unit-test suite or
CI; the two `models/ops/test*.py` scripts exercise only the optional deformable-attention extension.

## Commands

Install (the current `pyproject.toml` / `uv.lock` are Python 3.10+; the pinned legacy requirements
and install instructions are not aligned with those files):

```bash
uv sync                             # sync the declared project dependencies
ruff check .                        # lint (Ruff config is in pyproject.toml)
ruff format --check .               # check formatting; apply with `ruff format .`
```

There is no project-wide pytest suite or configured CI. The standalone deformable-attention checks
require the CUDA extension and a CUDA-capable environment; run them only after building the extension:

```bash
cd models/ops && sh make.sh && cd ../..
python models/ops/test.py
python models/ops/test_double_precision.py
```

Training and evaluation wrappers are the canonical launch commands; run from the repository root.
They use `torch.distributed.launch` even for one GPU. Choose a config matching the dataset, stage,
and machine; configs include machine-specific dataset/checkpoint paths:

```bash
sh scripts/stage1/train_vidorsmall.sh   # small VidOR subset, one GPU by default
sh scripts/stage1/train.sh              # VidVRD, DETR backbone, 8 GPUs
sh scripts/stage2/train.sh              # VidVRD stage 2
sh scripts/stage2/eval_vidorsmall.sh    # stage-2 evaluation with configured checkpoint
```

Set `NPROC_PER_NODE=<gpu_count>` to override the small-subset wrappers' one-GPU default. The main
training scripts under `scripts/stage1/` and `scripts/stage2/` cover other datasets and deformable
variants. Deformable variants require a compatible CUDA build of `models/ops`.

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

Quick smoke run: use `configs/vidorpart_stage1.json` (100 videos), or a small-subset wrapper. `--debug` forces `num_workers=0` and skips the zero-shot triplet scan.

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
- `batch_size` > 1 is supported (config sets 4): the collate packs clips as `(b,t,c,h,w)`
  NestedTensors (`util/misc.py` video branch of `from_tensor_list`), the training loop feeds one
  frame per clip per step (`samples.select_frame(fid)` for `fid in range(seq_len)`), and threads a
  **list of B per-clip `memory` dicts** through the frames. The val loader is forced to batch 1
  (val clips have variable frame counts; eval stays per-video).
- Each `memory` dict is keyed by `"<sub_tid>-<obj_tid>"` and accumulates per-frame `rel_embed`,
  `s_embed`, `o_embed`, labels, and (eval only) `frame_ids`. At `eos` each clip's sequence is
  mean-pooled per pair and pushed through `relation_classifier` → per-clip losses are summed over
  the batch (preserves batch-1 gradient scale; no LR change needed).
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

- Stage 1 can train but has no evaluation implementation. `--eval` is stage-2-only and asserts otherwise; stage-1 training does not run validation.
- Deformable transformer configs require `num_feature_levels > 1` and a successfully built `models/ops` CUDA extension; the vanilla transformer asserts `num_feature_levels == 1`.
- `datasets/__init__.py` routes only the exact dataset name `vidvrd` to VidVRD; names such as `vidorpart` use VidOR.
