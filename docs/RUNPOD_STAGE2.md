# RunPod Stage-2 Setup

This guide continues where [`RUNPOD_STAGE1.md`](RUNPOD_STAGE1.md) leaves off. It trains,
evaluates, and runs inference for the **standard VidOR-small Stage-2 config**, using the Stage-1
checkpoints already saved at `data/ckpts/vidorsmall_stage1`. It does **not** re-install the Python
environment and does **not** require `models/ops` (this config is non-deformable).

## Prerequisite: the Stage-1 environment

Stage 2 uses the exact same runtime as Stage 1 — follow the install section of
[`RUNPOD_STAGE1.md`](RUNPOD_STAGE1.md) (`scripts/runpod/install_stage1_deps.sh`) and activate the
venv before every command:

```bash
source .venv/bin/activate
```

Stage 2 is **non-deformable** (`num_feature_levels: 1`), so you never need to compile `models/ops`.
The model's ResNet backbone weights may still download on first use if not cached.

## How Stage-1 checkpoints are saved (and which one to use)

The training loop in `main.py:218-231` writes to `output_dir` after **every** epoch:

| File | Written when | Contents |
| ---- | ------------ | -------- |
| `checkpoint.pth` | every epoch (overwritten) | `{'model', 'optimizer', 'lr_scheduler', 'epoch', 'args'}` |
| `checkpoint{epoch:04}.pth` | every epoch (the condition `(epoch + 1) % 1 == 0` is always true) | same dict |
| `config.json` | once at startup | the fully resolved `args` (argparse defaults + config JSON) |
| `log.txt` | one JSON line per epoch | train losses / errors / lr / epoch / `n_parameters` |

For `data/ckpts/vidorsmall_stage1`, the Stage-1 config uses `epochs: 2`, so epoch indices are `0` and `1`:

| File | Epoch |
| ---- | ----- |
| `checkpoint0000.pth` | 0 (first epoch) |
| `checkpoint0001.pth` | 1 (final epoch) |
| `checkpoint.pth` | 1 (same as above; just overwritten each epoch) |

### Best checkpoint to use

**Use `data/ckpts/vidorsmall_stage1/checkpoint0001.pth`.** Stage 1 has **no validation/eval step**
(see `CLAUDE.md`: "Stage 1 can train but has no evaluation implementation"), so there is no
metric-based "best" checkpoint — the **final epoch is the only sensible choice**. `checkpoint.pth`
also holds epoch 1, but the numbered per-epoch file is the canonical one to reference.
`configs/vidorsmall_stage2.json` already sets `pretrain` to that exact path.

You can confirm a checkpoint's epoch (and that the Stage-1 files look right) with:

```bash
python - <<'EOF'
import torch
for f in ['checkpoint0000.pth', 'checkpoint0001.pth', 'checkpoint.pth']:
    c = torch.load('data/ckpts/vidorsmall_stage1/' + f, map_location='cpu', weights_only=False)
    print(f, '-> epoch', c['epoch'], '| keys', sorted(c.keys()))
EOF
```

## Data and checkpoint prerequisites

Before training Stage 2, the pod must have the VidOR-small files referenced by
`configs/vidorsmall_stage2.json`. Prepare them per [`DATA.md`](DATA.md) and confirm each exists:

- `data/vidorsmall/videos/`
- `data/vidorsmall/annotations/train/` and `data/vidorsmall/annotations/val/`
- `data/vidorsmall/action.txt`
- `data/metadata/vidorsmall_annotations.pkl`
- `data/metadata/vidorsmall_train_frames_stage2.json` (note: the **stage2** train-frames file, not stage1)
- `data/metadata/vidorsmall_val_frames.json` (needed because Stage 2 evaluates every epoch)
- `data/ckpts/vidorsmall_stage1/checkpoint0001.pth` (the Stage-1 weights from above)

Unlike Stage 1, Stage 2 does **not** need `data/weights/detr-r101-2c7b67e5.pth` — its `pretrain`
is the Stage-1 checkpoint, not the DETR backbone.

### Move the Stage-1 checkpoint onto the pod

The checkpoints were downloaded to the local machine at the end of `RUNPOD_STAGE1.md`. Upload the
final-epoch file (≈695 MB) to the pod before training:

```bash
scp -P PORT -i ~/.ssh/id_ed25519 \
  /home/samuel/projects/VRDFormer_VRD/data/ckpts/vidorsmall_stage1/checkpoint0001.pth \
  root@POD_IP:/root/VRDFormer_VRD/data/ckpts/vidorsmall_stage1/
```

(Adjust the remote path if the pod's checkout is elsewhere.) If the pod already has the whole
`vidorsmall_stage1/` folder, skip this.

## Launch Stage-2 training

From the repository root, activate the environment and run the wrapper. It defaults to one process;
set `NPROC_PER_NODE` to the number of GPUs:

```bash
source .venv/bin/activate
NPROC_PER_NODE=1 sh scripts/stage2/train_vidorsmall.sh
```

The wrapper uses `configs/vidorsmall_stage2.json` with `--num_queries 200`, `--lr 5e-5`, and
`--lr_backbone 1e-5`. It also derives `--accumulate_steps` from `NPROC_PER_NODE` to keep the
effective batch at ~4 clips/optimizer step (`NPROC=1 → 4`, `NPROC=2 → 2`, `NPROC≥4 → 1`).
Remember the config JSON is loaded over the command-line defaults (`main.py:150-155`), so edit the
JSON — not the flags — to change `batch_size`, `epochs`, `seq_len`, `pretrain`, `output_dir`, etc.

Key settings in `configs/vidorsmall_stage2.json`:

| Key | Value | Meaning |
| --- | ----- | ------- |
| `stage` | `2` | relation classification over time |
| `pretrain` | `data/ckpts/vidorsmall_stage1/checkpoint0001.pth` | Stage-1 weights to initialize from |
| `output_dir` | `data/ckpts/vidorsmall_stage2` | new checkpoint directory |
| `seq_len` | `8` | frames sampled per clip |
| `batch_size` | `1` | clips per forward/backward (8 frames each) |
| `accumulate_steps` | derived from `NPROC_PER_NODE` | gradient accumulation; keeps effective batch ~4 clips/optimizer step |
| `epochs` / `lr_drop` | `2` / `1` | 2 epochs, LR drops after every epoch |
| `train_clip_sample_ratio` | `0.25` | trains on every 4th clip |
| `amp` / `benchmark` | `true` | bf16 autocast on Ampere+, cudnn autotune |

Notes:

- Stage 2 keeps the autograd graph alive across **all** `seq_len` frames of a clip (the per-pair
  `memory` embeddings are concatenated without `detach()` in training), so peak memory scales with
  `batch_size × seq_len`. `batch_size: 1` (8 image-equivalents at once) is what fits a 48 GB GPU;
  `--accumulate_steps` (derived from `NPROC_PER_NODE`) restores an effective batch of ~4 clips per
  optimizer step. Raising `batch_size` without lowering `seq_len` or `resolution` (default `"large"`)
  will OOM again.
- Because `stage == 2`, `main.py` also calls `eval_stage2` **after every epoch** (`main.py:233-234`)
  and logs it. That eval scans the full train annotations to compute `zeroshot_triplets`
  (`dataloader_initializer`), so the val-loader startup is slow — minutes of no iteration output is
  normal, not a hang.
- Run everything **from the repository root**: `eval_stage2` opens `data/vidorsmall/action.txt`
  relative to the current working directory (`engine.py:233`).
- Detach long runs from your SSH session (e.g. `tmux`) so training survives disconnects.

## Stage-2 checkpoints

Stage 2 saves with the same scheme under `data/ckpts/vidorsmall_stage2/`: `checkpoint.pth` plus
`checkpoint{epoch:04}.pth` each epoch (`checkpoint0000.pth`, `checkpoint0001.pth` for 2 epochs),
alongside `config.json` and `log.txt`. **The final epoch (`checkpoint0001.pth`, also the current
`checkpoint.pth`) is the one to keep and evaluate.**

## Evaluate Stage 2

Stage 2 is the only stage with an evaluation path. Point the eval wrapper at a trained Stage-2
checkpoint (edit the `--resume` path in the script, or run the command directly):

```bash
source .venv/bin/activate
python -m torch.distributed.launch \
    --master_port 47746 \
    --nproc_per_node=1 \
    main.py \
    --eval \
    --dataset_config configs/vidorsmall_stage2.json \
    --resume data/ckpts/vidorsmall_stage2/checkpoint.pth
```

`scripts/stage2/eval_vidorsmall.sh` runs exactly this. The `--eval` path asserts `stage == 2`
(`main.py:200-201`).

Reported metrics (`util/evaluate.py`) are printed for three settings — **overall**, **zero-shot**,
and **generalized zero-shot**:

- **detection mean AP** and **detection recall@50 / @100** — relation *detection*
  (`triplet` + temporal IoU of subject/object trajectories ≥ 0.5), and
- **tagging precision@1 / @5 / @10** — relation *tagging* (triplet only, no boxes).

> Use `--resume` with the **Stage-2** checkpoint, never `--pretrain`. Passing Stage-1 weights via
> `--pretrain` loads them into the Stage-2 model and produces meaningless metrics.

## Inference on an mp4

**Important:** Stage 2 is an *oracle* relation classifier — it does **not** detect objects. It
initializes its queries by `roi_align` on ground-truth subject/object boxes
(`models/transformer.py:147-148`, using `targets["unscaled_sub_boxes"]` / `["unscaled_obj_boxes"]`)
and, at eval, `relation_classifier(..., is_eval=True)` predicts a verb for each **ground-truth**
`(subject_tid, object_tid)` pair (`models/vrdformer_stage2.py:82-95`). It never enumerates predicted
pairs.

That means a Stage-2 checkpoint **cannot** turn an arbitrary, unlabeled internet video into triplets
on its own. The required input is a video that already has object-track annotations, i.e.:

- the clip at `data/vidorsmall/videos/<video_id>.mp4`,
- a VidOR-format annotation at `data/vidorsmall/annotations/val/<video_id>.json`,
- an entry for it in `data/metadata/vidorsmall_annotations.pkl` and
  `data/metadata/vidorsmall_val_frames.json`.

The practical way to "do inference with an mp4" is therefore to run the model on one of the
annotated val videos and read the predicted triplets. You can adapt the eval loop
(`engine.py:eval_stage2`) or `data/dev_bs4_eval_smoke.py` into a single-video script:

```python
# run from the repository root, e.g. tools/infer_vidorsmall.py
import torch
from main import get_args_parser
from datasets import build_dataset
from models import build_model

args = get_args_parser().parse_args([
    '--dataset_config', 'configs/vidorsmall_stage2.json',
    '--resume', 'data/ckpts/vidorsmall_stage2/checkpoint.pth',
])
args.eval = True            # do not use argparse bool; set directly
dataset = build_dataset('val', args)   # val dataset, one video per item
# then replicate the eval_stage2 loop for dataset[i] and print
# model.relation_classifier(memory[0], gt=targets[0]['groundtruth'], is_eval=True)
```

For a genuinely new mp4 with no annotations, you would need to supply object tracks yourself first
(e.g. annotate the subject/object trajectories in VidOR format), because the object-detection /
tracking half of the pipeline that would produce those boxes from raw pixels is **not wired into an
inference path in this repo** (Stage 1 has no eval/inference driver, and `TrackingBase.forward`
returns nothing when `targets is None`). Plan for that annotation step before expecting triplets
from a raw video.

## Transfer Stage-2 checkpoints back

Before the pod is terminated, pull the Stage-2 outputs back to the local machine (from local shell):

```bash
scp -rv -P PORT -i ~/.ssh/id_ed25519 \
  root@POD_IP:/root/VRDFormer_VRD/data/ckpts/vidorsmall_stage2/ \
  /home/samuel/projects/VRDFormer_VRD/data/ckpts/
```

That folder contains `checkpoint.pth`, `checkpoint0000.pth`, `checkpoint0001.pth`, `config.json`,
and `log.txt`.
