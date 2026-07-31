# Phase 5: Inference on New Videos

## Problem

There is **no existing inference-only code path** in VRDFormer. The eval loop (`engine.py:eval_stage2`) requires:
1. Ground-truth per-frame boxes and track IDs (for ROI Align query initialization)
2. Ground-truth relation triplets (for `relation_classifier` which only predicts the verb, not subject/object)
3. The DataLoader pipeline which needs annotation files

For inference on an unannotated `.mp4`, we must build a custom pipeline.

## Inference Architecture

```
INPUT: .mp4 video
    │
    ▼  (extract frames at every_n_frames interval)
Frames: [frame_0, frame_5, frame_10, ...]
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 1: Detection                                   │
│                                                      │
│ For each frame:                                      │
│   1. Backbone → encoder → decoder                    │
│   2. 200 learned queries predict:                    │
│      sub_box(4), obj_box(4), sub_cls(81),            │
│      obj_cls(81), verb_cls(50), hs_embed(256)        │
│   3. Filter: keep queries where                      │
│      sub_cls ≠ no-obj AND obj_cls ≠ no-obj           │
│      AND max(verb_sigmoid) > conf_threshold          │
│   4. Extract predictions per query                   │
│                                                      │
│ Output per frame: list of {                          │
│   sub_box, obj_box, sub_cls, obj_cls,                │
│   verb_cls, verb_score, hs_embed                     │
│ }                                                    │
└─────────────────────────────────────────────────────┘
    │
    ▼  (associate across frames via IoU + class match)
┌─────────────────────────────────────────────────────┐
│ TRACKLET FORMATION                                   │
│                                                      │
│ For each frame pair (t, t+1):                        │
│   Match subject boxes by IoU + same class             │
│   Match object boxes by IoU + same class              │
│   If both match → same tracklet (persistent SO pair)  │
│   Otherwise → new tracklet                            │
│                                                      │
│ Output: list of tracklets {                          │
│   track_id,                                          │
│   frames: [{fid, sub_box, obj_box, hs_embed}, ...],  │
│   sub_cls, obj_cls                                   │
│ }                                                    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ STAGE 2: Relation Classification                     │
│                                                      │
│ For each tracklet (pseudo-GT):                       │
│   1. Extract per-frame sub/obj boxes                  │
│   2. ROI Align → s_embed, o_embed                    │
│   3. so_linear(cat(s,o)) → query init                │
│   4. Decoder → rel_embed per frame                   │
│   5. memory_update across frames                     │
│   6. At end: mean_pool → verb_class_embed            │
│   7. verb = argmax(logits), score = softmax(max)     │
│                                                      │
│ Output per tracklet: {                               │
│   triplet: (sub_cls_name, verb_name, obj_cls_name),  │
│   score: float,                                      │
│   sub_boxes: [...], obj_boxes: [...],                │
│   frame_range: (start, end)                          │
│ }                                                    │
└─────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT: results.json
```

## `inference.py` Design (to be implemented)

### CLI Interface
```bash
python inference.py \
    --video input.mp4 \
    --stage1_ckpt data/ckpts/vidor_stage1/checkpoint0004.pth \
    --stage2_ckpt data/ckpts/vidor_stage2/checkpoint.pth \
    --output results.json \
    --conf_threshold 0.3 \
    --every_n_frames 5 \
    --dataset vidor
```

### Core Functions

1. **`load_models(args)`** — Build Stage 1 + Stage 2 via `build_model()`, load checkpoints, set to eval mode

2. **`extract_frames(video_path, every_n)`** — Use decord to load frames at interval, return tensor `(T, 3, H, W)`

3. **`run_stage1(model_s1, frames)`** — For each frame, call `VRDFormer.forward(frame, targets=None)` (bypasses TrackingBase). Filter predictions by confidence. Return per-frame detections.

4. **`form_tracklets(per_frame_detections)`** — Match predictions across frames using IoU + class matching. Assign synthetic track IDs. Return list of tracklets.

5. **`run_stage2(model_s2, tracklets, frames)`** — For each tracklet, collect per-frame boxes as pseudo-GT, run through Stage 2 forward with memory accumulation, classify at end. Return relation triplets.

6. **`save_results(triplets, output_path)`** — Write JSON with per-video predictions.

### Key Implementation Notes

**Bypassing TrackingBase for inference:**
`VRDFormerTracking.forward()` requires `targets` with `prev_target` (for training). For pure inference, call the base class directly:
```python
from models.vrdformer import VRDFormer
outputs, targets_out, features, memory, hs = VRDFormer.forward(model_s1.module, frame_tensor, targets=None)
```

**Tracklet matching algorithm:**
```python
def match_boxes(boxes_t, boxes_t1, iou_threshold=0.3):
    """Match boxes between consecutive frames by IoU."""
    iou_matrix = box_iou(box_cxcywh_to_xyxy(boxes_t), box_cxcywh_to_xyxy(boxes_t1))
    # Hungarian matching on -IoU (cost = -iou)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix.numpy())
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] > iou_threshold:
            matches.append((r, c))
    return matches
```

**Using class label dictionaries:**
Load from `data/vidor/action.txt` and `data/vidor/obj.txt` to convert class indices to human-readable labels.

### Output Format

```json
{
  "video": "input.mp4",
  "num_frames": 120,
  "num_tracklets": 15,
  "predictions": [
    {
      "triplet": ["adult", "watch", "cellphone"],
      "score": 0.85,
      "sub_boxes": [[100, 150, 300, 400], [102, 148, 302, 398], ...],
      "obj_boxes": [[350, 200, 380, 250], [348, 198, 382, 252], ...],
      "frame_range": [0, 45],
      "track_id": 0
    },
    ...
  ]
}
```

## Planned Files to Create

| File | Purpose |
|------|---------|
| `inference.py` | Main inference script |
| Will call existing: `models/__init__.py:build_model` | Model construction |
| Will call existing: `util/box_ops.py:box_cxcywh_to_xyxy` | Box format conversion |
| Will call existing: `util/checkpoints.py` | Checkpoint loading |
