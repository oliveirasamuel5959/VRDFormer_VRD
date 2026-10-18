# RunPod Single-Video Inference (new, unlabeled videos)

This guide describes how to run the trained VRDFormer on an **arbitrary video that is not part of
VidOR/VidVRD** and produce relation-triplet predictions. It follows the
`video → Stage 1 (detection) → Stage 2 (relation) → triplets` pipeline.

> **Status: implemented.** `inference.py` exists at the repository root; this document was the plan
> for it and is kept as the design record. It continues from [`RUNPOD_STAGE2.md`](RUNPOD_STAGE2.md),
> which trains and evaluates Stage 2. See "Implementation notes" at the end for where the code
> departs from this plan.

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
As built: `inference.py` stays at the root and `scripts/inference/inference.sh` is the launcher,
matching the `.sh`-in-`scripts/` convention of the training wrappers.

### CLI

The planned invocation. The implemented CLI keeps these flag names, adds the flags listed under
"Implementation notes", and drops `--output` in favour of `--output_dir` + a per-video
subdirectory (see "Output" below).

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

## Implementation notes (where the code departs from this plan)

Measured on the CPU workstation against `data/vidorsmall/videos/10042935613.mp4` (a val video with
GT boxes and triplets), 8–12 decoded frames.

1. **The CLI no longer reuses `main.get_args_parser()`'s `--dataset_config`.** It is replaced with
   `conflict_handler="resolve"`, because inference names two configs (`--stage1_config`,
   `--stage2_config`) and loads no dataloader.
2. **Config resolution order is argparse defaults < checkpoint's saved args < config JSON < explicit
   CLI overrides.** The plan said "config JSON wins, exactly as `main.py` does", but neither
   vidorsmall config sets `num_queries`, so the argparse default (100) would win over the
   checkpoint's 200 and the state_dict load would fail. The checkpoint's args are also the only
   record of the *training* `resolution`, which is reported when it disagrees with the config.
3. **Stage 1 keeps the previous frame's backbone features** (`prev_features`) because the checkpoint
   is `multi_frame_attention: true`. Measured effect at `--conf_sub/--conf_obj 0.15`: 34 vs 22
   detections on frame 1.
4. **Class equality is not part of the tracklet association cost** (`--cls_gate motion`, the
   default). The plan's "IoU > iou_thresh AND class matches" was implemented and measured: the
   detector's argmax class flips between frames, so gating on it produced 93 single-frame tracklets
   out of 142 (mean length 1.35). Matching on box motion and voting the classes back
   (`--cls_vote keep`) gives 47 of 111 (mean 2.02). `--cls_gate same` restores the plan's behaviour.
5. **Triplet `score` is the sigmoid verb probability**, not the raw logit maximum
   (`engine.eval_stage2`'s convention), and tracklets below `--min_score` (default 0.05) are dropped.
6. **Threshold defaults are calibrated, not the plan's `--conf_sub/--conf_obj 0.5`.** The checkpoint's
   best pre-threshold pair score is ~0.19, so 0.5 yields *zero* detections on every frame. The
   defaults are 0.15, and each run prints the best pre-threshold subject/object/pair score so the
   value can be set from data. Reference measurements at 0.15: localisation recall 0.41 @IoU 0.50,
   0.59 @IoU 0.30, 0.25 with the correct class at IoU 0.50.
7. **The detector loses objects over time.** Mean best pair score falls from ~0.18 (frame 0) to
   ~0.09 by frame 4 at stride 5, after which some frames have no detections at all. `--max_gap` can
   bridge *missed* detections but not frames with none, so this — not the association — is what
   keeps predicted tracklets short.
8. **`sub_traj`/`obj_traj` are source-frame aligned, with a sidecar for evaluation.**
   `util.evaluate.viou` indexes both trajectories on the same grid, so decoded-axis trajectories
   cannot be scored against ground truth. The JSON carries `sub_boxes`/`obj_boxes` on the decoded
   axis and `sub_traj`/`obj_traj` expanded to every source frame in
   `source_duration = [start, end)`. `write_eval_prediction` emits `eval_prediction.json` in
   `evaluate_visual_relation`'s `prediction` shape. Scores against GT are 0 with these checkpoints,
   but the triplet names are the binding constraint (Stage 2 predicts `next_to` for nearly every
   pair, and GT verbs here are `hold`/`use`/`watch`/`in_front_of`/`behind`/`away`), not the boxes.
9. **Mismatches between a config and its checkpoint are reported, never silently applied.** An
   *architecture* key that differs (`num_queries`, `position_embedding`, …) usually means the
   state_dict load will fail or the network will compute something else; a `resolution` difference
   loads and runs but predicts at a scale the weights never saw. `data/ckpts/vidorsmall_stage2/`
   currently hits the latter: the checkpoint's saved args say `resolution: "small"` while
   `configs/vidorsmall_stage2.json` says `"large"`, so pass `--stage2_resolution small` to match the
   weights.

## Running it

`scripts/inference/inference.sh` is the launcher. It resolves the interpreter (an activated venv,
else the project `.venv`), picks `--device cuda` only when `nvidia-smi` works, and defaults
`--stage2_resolution small` to match the weights. Every flag after the video path goes straight to
`inference.py`.

```bash
source .venv/bin/activate
sh scripts/inference/inference.sh /path/to/video.mp4
```

Then read the output. The script defaults the checkpoints and both configs to the vidorsmall pair,
so the video path is the only required argument.

### Output

For `<video_id>` = the input file's stem, and `<output_dir>` defaulting to
`output/inference/<dataset>`:

```
<output_dir>/<video_id>/
├── results.json          # triplets, tracklets, per-frame detections
├── eval_prediction.json  # util/evaluate-shaped sidecar
└── video.mp4             # the input annotated with boxes and triplet labels
```

**Nothing is written until the whole pipeline finishes** — `results.json` after Stage 2,
`video.mp4` after rendering. On a long clip at a small `--every_n_frames` that is a long wait with
no output on disk, so run a short prefix first (`--max_frames 30`) to confirm the thresholds suit
your video, then launch the full pass. `--output_dir` moves the root.

### Read the `best score before thresholding` line first

This is the one number that decides whether you get any output at all. The vidorsmall Stage-1
checkpoint is weak, and on a video it has never seen its best pair score is typically **0.08–0.35**.
The `--conf_sub`/`--conf_obj` defaults are 0.15. If the run reports `0 detections`, the warning prints
the value that would have worked:

```
[inference] WARNING: no detection cleared the thresholds; the highest pair score reached was
0.104 (subject 0.121, object 0.106). Lower --conf_sub/--conf_obj below those values, e.g.
--conf_sub 0.06 --conf_obj 0.05.
```

A random video that does not look like VidOR (animation, screencast, gameplay, a test pattern) scores
lower than real footage — synthetic `ffmpeg testsrc` clips peak around 0.10–0.16 and produce nothing
at the defaults. That is the model, not the pipeline. Confirm the pipeline works on a vidorsmall clip
first (`data/vidorsmall/videos/*.mp4`), then point it at your download.

### Container formats

decord ships against FFmpeg and decodes **H.264, HEVC, and VP9** fine; **AV1 fails**, and OpenCV
cannot decode AV1 either, so there is no in-script fallback. The script now detects this and tells
you to transcode rather than dumping a decoder traceback:

```bash
ffmpeg -i downloaded.mp4 -c:v libx264 -pix_fmt yuv420p converted.mp4
sh scripts/inference/inference.sh converted.mp4
```

If you download with `yt-dlp`, ask for a compatible codec up front:

```bash
yt-dlp -f "bv*[vcodec^=avc1]+ba/b[ext=mp4]" -o clip.mp4 <url>
```

Verified working (on CPU, so a GPU run should be strictly faster): landscape and portrait aspect
ratios, 23.976 fps, 4K, 3-frame clips, and rotation metadata. Odd fps only affects the output
video's playback rate; rotation metadata is already applied to the decoded pixels by decord.

### Cost

On this CPU workstation one frame costs roughly 2.6 s in Stage 1 at `--resolution large`, so a
36-frame run takes ~2.5 minutes; Stage 2 is ~1.1 fps. On the pod's GPU, expect the same count of
forward passes (2 per decoded frame) plus one extra decode pass if you keep the annotated video
(`--no_video` skips it). A 1-minute clip at the default `--every_n_frames 5` is ~360 decoded frames
if it is 30 fps.

### Expected results on a new video

Low. The checkpoints are a 100-video POC subset trained for 2 epochs, Stage 1 is run in a
static-query regime it was not trained in, and Stage 2 predicts a verb for pairs the detector
proposed. On vidorsmall's own val split the measured Stage-1 box recall is 0.41 @IoU 0.50 / 0.59
@IoU 0.30; Stage 2 dominates the verb distribution with a small number of classes, so triplets are
mostly `adult next_to <thing>`. This is the pipeline working, not the model working.

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
