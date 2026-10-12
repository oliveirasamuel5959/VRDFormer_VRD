# RunPod Single-Video Inference (new, unlabeled videos)

This guide describes how to run the trained VRDFormer on an **arbitrary video that is not part of
VidOR/VidVRD** and produce relation-triplet predictions. It follows the
`video → Stage 1 (detection) → Stage 2 (relation) → triplets` pipeline.

> **Status:** design only — `inference.py` does not exist yet. This document is the plan to implement
> it. It continues from [`RUNPOD_STAGE2.md`](RUNPOD_STAGE2.md), which trains and evaluates Stage 2.

## Why a new script is needed

The repo currently has **no inference path**. The only evaluation entry is
`engine.eval_stage2` (`engine.py:225-272`), and it is an **oracle** evaluator:

- **Stage 2 does not detect objects.** Its decoder queries are initialized by `roi_align` on
  **ground-truth** subject/object boxes (`models/transformer.py:145-153`), and
  `relation_classifier(..., is_eval=True)` (`models/vrdformer_stage2.py:82-95`) predicts a **verb for
  each ground-truth `(subject_tid, object_tid)` pair**. It never enumerates pairs.
- The eval loop also requires ground-truth triplets plus the dataset loader (`.pkl`, frames JSON, raw
  annotations) — none of which exist for a new video.

So a Stage-2 checkpoint **cannot** turn a raw `.mp4` into triplets by itself. The missing pieces are:

1. a **detector** that produces subject/object boxes per frame,
2. **association** of those detections into persistent tracklets, and
3. a **Stage-2 pass** that treats those tracklets as pseudo-ground-truth.

This was already scoped in [`../spec/train_pipeline/05_inference.md`](../spec/train_pipeline/05_inference.md)
under "Phase 5: Inference on New Videos"; this document is the concrete implementation plan.

## Environment

Inference needs the same runtime as Stage 1/2 — run
`scripts/runpod/install_stage1_deps.sh` and activate the venv before every command:

```bash
source .venv/bin/activate
```

It also needs `decord`, `scipy`, `opencv-python-headless`, `timm`, and `pycocotools`, all of which are
in `requirements-runpod-stage1.txt`. The **non-deformable** vidorsmall configs do **not** require the
compiled `models/ops` extension.

> The project `.venv` on a plain CPU workstation is CPU-only and missing several of these packages —
> run inference on the RunPod pod, where Stage 1/2 training already works.

## Checkpoints and inputs

| Input | Path |
| ----- | ---- |
| Stage-1 weights | `data/ckpts/vidorsmall_stage1/checkpoint0001.pth` (final epoch; Stage 1 has no eval) |
| Stage-2 weights | `data/ckpts/vidorsmall_stage2/checkpoint.pth` |
| Stage-1 config | `configs/vidorsmall_stage1.json` |
| Stage-2 config | `configs/vidorsmall_stage2.json` |
| Class names | `data/vidorsmall/obj.txt` (80 object classes) and `data/vidorsmall/action.txt` (50 verbs) |
| Video | any `.mp4` (for the smoke test, use a `data/vidorsmall/videos/<id>.mp4`) |

Both configs are read as-is and merged over the argparse defaults (JSON wins, exactly as
`main.py:150-155` does), because Stage 1 and Stage 2 need **different** model settings
(`stage`, `multi_frame_attention`, `tracking`, …).

## Architecture

```
INPUT: .mp4
   │  decode with decord at --every_n_frames stride, apply the val transform
   ▼
Frames: (3, T, H', W') normalized tensor  +  resized size (h', w')
   │
   ▼  Stage 1 — object detector (base VRDFormer.forward, targets=None)
   │    • 200 static queries → per query: sub_box, obj_box, sub_cls, obj_cls, verb
   │    • decode the focal-loss heads with sigmoid (80 classes, no no-object slot)
   │    • keep confident, dedup
   ▼
Per-frame detections: [{sub_box(px), obj_box(px), sub_cls, obj_cls, verb, verb_score}, ...]
   │  associate across frames by IoU + class match (scipy Hungarian)
   ▼
Tracklets: [{track_id, frames:[{fid, sub_box, obj_box}], sub_cls, obj_cls}, ...]
   │  synthetic targets (unscaled_*_boxes, so_track_ids, num_inst, ...) → pseudo-GT
   ▼  Stage 2 — relation classifier (VRDFormer_S2.forward, memory accumulated)
   │    • per frame: roi_align → so_linear → decoder → rel_embed
   │    • end: mean-pool per pair → verb_class_embed → verb
   ▼
OUTPUT: results.json  (per-tracklet triplets)
```

## `inference.py` design (to be implemented)

Suggested location: **`inference.py` at the repository root** (single file; the pipeline is short
enough to keep in one place, and the repo has no `tools/` convention beyond one-off repro scripts).

### CLI

```bash
python inference.py \
    --video data/vidorsmall/videos/0000000000.mp4 \
    --stage1_ckpt data/ckpts/vidorsmall_stage1/checkpoint0001.pth \
    --stage2_ckpt data/ckpts/vidorsmall_stage2/checkpoint.pth \
    --stage1_config configs/vidorsmall_stage1.json \
    --stage2_config configs/vidorsmall_stage2.json \
    --output results.json \
    --every_n_frames 5 \
    --conf_sub 0.5 --conf_obj 0.5 --conf_verb 0.1 \
    --iou_thresh 0.3 \
    --dataset vidorsmall \
    --device cuda
```

### Functions

| Function | Responsibility |
| -------- | -------------- |
| `load_models(args)` | Build Stage-1 and Stage-2 via `models.build_model`, load checkpoints, `.eval()`, `torch.no_grad()` |
| `extract_frames(video_path, stride, args)` | decord decode at stride + val transform → `(3,T,H',W')`, `(h',w')` |
| `run_stage1(model_s1, frames, args)` | Base `VRDFormer.forward` per frame with `prev_features`; sigmoid-decode; filter; dedup |
| `form_tracklets(per_frame_dets, args)` | IoU + class association (scipy Hungarian) → tracklets |
| `run_stage2(model_s2, frames, tracklets, args)` | Synthetic targets per frame; memory accumulation; `relation_classifier` |
| `save_results(triplets, path)` | Write JSON |

### 0. Args and model construction

Reuse `main.get_args_parser()`, then merge each config JSON into `vars(args)` exactly as `main` does.
Build each model with `build_model(args)` (`models/__init__.py:9`), which derives
`num_obj_classes=80` / `num_verb_classes=50` from the dataset name and, with `focal_loss=true`, sizes
the classification heads to **80** (no extra no-object slot).

Load weights with a strict `load_state_dict(checkpoint['model'])` — the same shape as `--resume`
(`util/checkpoints.py:173-183`). Use `--stage2_ckpt` for Stage 2 (never Stage-1 weights via
`--pretrain`).

**Stage-1 forward must call the base class**, not the tracking wrapper: `VRDFormerTracking.forward`
returns `None` when `targets is None` (`models/vrdformer_track.py:158-191`). Call
`VRDFormer.forward(model_s1, samples, targets=None, prev_features=prev_feat)` directly.

### 1. Frames

- Decode with **decord** (as `datasets/dataset.py:183-188` does), sampling every `--every_n_frames`-th
  frame into a list of `H×W×3 uint8` numpy arrays.
- Apply the **val** transform:
  `make_video_transforms('val', cautious=True, by_ratio=False, resolution=args.resolution)` →
  `[RandomResize([800], max_size=1333), ToTensor + ImageNet Normalize]`, producing `(3, T, H', W')`.
- Stage 1 uses multi-frame attention, so iterate frames in order and keep the previous frame's
  backbone features; frame 0 is called with `prev_features=None` (the model falls back to the current
  frame, `models/vrdformer.py:148-149`).

### 2. Stage-1 detection (per frame)

```python
out, _, feats_all, _, _ = VRDFormer.forward(
    model_s1, frame_nt, targets=None, prev_features=prev_feat)
```

`out` keys: `pred_sub_logits` / `pred_obj_logits` `[1, 200, 80]`, `pred_verb_logits` `[1, 200, 50]`,
`pred_sub_boxes` / `pred_obj_boxes` `[1, 200, 4]` (normalized **cxcywh**), `hs_embed`.

- Decode classes with **`sigmoid`** (the heads are trained with `sigmoid_focal_loss`, so each class is
  an independent binary). With `focal_loss=true` there is **no no-object class** — do *not* use
  softmax and do *not* drop a last slot.
  *Confirmed against `checkpoint0001.pth`: `sub_class_embed.*.weight` is `(80, 256)`,
  `verb_class_embed.*` is `(50, 256)`, `query_embed.weight` is `(200, 512)`.*
- Convert boxes to xyxy; keep a copy scaled to **resized pixel** space for Stage-2 `roi_align`.
- Filter with `--conf_sub`/`--conf_obj`/`--conf_verb`; optionally IoU-dedup overlapping queries of the
  same class.
- Return per frame: `{sub_box(px), obj_box(px), sub_cls, obj_cls, verb, verb_score, query_idx}`.

### 3. Tracklet formation

- Between consecutive frames, run `scipy.optimize.linear_sum_assignment` on `-IoU` of **subject** boxes
  and of **object** boxes independently; accept a match only when IoU > `--iou_thresh` **and** the
  class matches. A subject-object pair joins an existing tracklet only when **both** sides match;
  otherwise it starts a new tracklet.
- Assign synthetic ids `track_id = "<sub_tid>-<obj_tid>"`.
- Return tracklets: `{track_id, frames: [{fid, sub_box, obj_box, ...}], sub_cls, obj_cls}`.

*(Alternative, not chosen: run Stage 1 in TrackFormer tracking mode with recurrent track queries and
`matcher` association. It is closer to how Stage 1 was trained, but the fork never wires that path
into an inference driver and it is more complex, so start with static-query detection.)*

### 4. Stage-2 relation classification

For each frame, build **one** pseudo-target dict covering all tracklets active in that frame:

| Key | Value |
| --- | ----- |
| `unscaled_sub_boxes`, `unscaled_obj_boxes` | xyxy in **resized pixel** space (roi_align input, scale 1/32) |
| `sub_boxes`, `obj_boxes` | cxcywh **normalized** (read by `memory_update`) |
| `so_track_ids` | `[sub_tid, obj_tid]` per row |
| `num_inst` | number of rows |
| `sub_labels`, `obj_labels` | predicted class tensors |
| `verb_labels` | zeros `[N, 50]` (read by `memory_update`, unused by non-eval `relation_classifier`) |
| `frame_id` | frame index |

These are exactly the keys `VRDFormer_S2.forward` / `memory_update` / `prepare_tag_query` read
(`models/vrdformer_stage2.py:44-78`, `models/transformer.py:145-153`). Then:

```python
memory = None
for fid in range(len(frames)):
    memory = model_s2.select_frame_forward(frames, fid, [pseudo_target[fid]], memory,
                                           eos=(fid + 1) == len(frames), is_eval=False)
sub_logits, verb_logits, obj_logits, sub_labels, verb_labels, obj_labels = \
    model_s2.relation_classifier(memory[0])          # NON-eval branch
```

The **non-eval** branch (`models/vrdformer_stage2.py:97-116`) iterates **all** memory keys (one per
tracklet), mean-pools each pair's `rel_embed` over its frames, and returns the three logit tensors —
exactly the per-tracklet predictions we want. Map `verb_logits.argmax(-1)` → `action.txt` names, and
`sub/obj_logits.argmax(-1)` → `obj.txt` names.

For **per-frame** verb predictions, apply `verb_class_embed[-1]` to each frame's `rel_embed[track_id]`
(memory stores the per-frame concatenation) instead of the pooled vector.

### 5. Output JSON

```json
{
  "video": "input.mp4",
  "num_frames": 120,
  "stride": 5,
  "num_tracklets": 15,
  "predictions": [
    {
      "triplet": ["adult", "watch", "cellphone"],
      "score": 0.85,
      "sub_cls": "adult", "obj_cls": "cellphone", "verb": "watch",
      "sub_boxes": [[100,150,300,400], ...],
      "obj_boxes": [[350,200,380,250], ...],
      "frame_range": [0, 45],
      "track_id": "0-1",
      "per_frame_verbs": ["watch", "watch", "hold", ...]
    }
  ]
}
```

`score` is the verb logit maximum (matching `engine.eval_stage2`'s convention).

## Build order (verifiable checkpoints)

1. **`load_models` + frame extraction** — print shapes; confirm the frame tensor matches what a dataset
   sample would produce (same transform).
2. **Stage-1 on one annotated val video** — dump boxes and compare to the GT boxes in
   `data/vidorsmall/annotations/val/<vid>.json` (IoU / recall). This checks detection alone.
3. **Tracklets + Stage 2** on the same val video — score predicted triplets against GT using
   `util/evaluate.py`. Adding `sub_traj`/`obj_traj`/`duration` to each prediction also enables the
   detection mAP/recall branch (`util/evaluate.py:98`), which is otherwise skipped.
4. **Run on the new unlabeled mp4** — write `results.json`.

## Verification

The real test is steps 2–3: an annotated val video has GT boxes *and* GT triplets, so we can (a) check
Stage 1 finds the right objects, and (b) score predicted triplets against GT with the existing
`util/evaluate.evaluate`. Only once that passes do we trust it on the unlabeled video, where we
sanity-check instead: triplets are non-empty, class names are valid, boxes track objects across
frames, and scores are ranked sensibly.

## Risks / gotchas

- **Focal-loss head decoding.** Use `sigmoid`, not softmax, and remember there is no no-object class
  when `focal_loss=true`; a softmax + "drop last" decoding would give wrong confidences.
- **Static-query-only detection** differs from Stage 1's training regime (which injects recurrent
  track queries), so box quality may be lower than training implies — expect to sweep
  `--conf_sub`/`--conf_obj`.
- **Backbone returns only layer4** (`num_feature_levels=1`, `return_interm_layers=False`), stride 32 —
  keep `roi_align` `spatial_scale` at 1/32 and be consistent between Stage-1 output scaling and the
  pseudo-target boxes.
- **Pixel-space consistency** — `roi_align` boxes must be in **resized** pixel coordinates, and the
  same resized size must be used across frames (val transform uses a fixed `[800]` min side).
- **CPU vs CUDA** and `torch.no_grad()` everywhere; the local CPU venv lacks the required packages.
- **VidOR-small is a small subset** — the model is weak; low metrics are expected and are not
  necessarily a bug.
