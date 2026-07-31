# Phase 4: Stage 2 Training — Relation Classification

## What Stage 2 Learns

Stage 2 takes the detection outputs from Stage 1 and classifies the **temporal relation** (predicate) by aggregating features across the video clip. It does NOT learn box regression — boxes come from ground truth.

**Key differences from Stage 1:**
- Batch size = 1 (processes one video clip at a time)
- No learned queries — queries initialized from GT boxes via ROI Align
- No Hungarian matching — GT track IDs provide direct correspondence
- No box loss — only classification losses
- `seq_len=8` — processes 8 frames per clip instead of 2
- Frames processed **sequentially**, accumulating memory

## Stage 2 Architecture (per-frame)

```
Input: (1, 3, H, W) single frame
    │
    ▼
ResNet-101 → features + pos encoding
    │
    ▼
Transformer Encoder → memory
    │
    ▼
ROI Align at GT sub/obj boxes → s_embed, o_embed
    │
    ▼
so_linear(cat(s_embed, o_embed)) → query init (padded to 200 queries)
    │
    ▼
Transformer Decoder → rel_embed, s_embed, o_embed per SO pair
    │
    ▼
memory_update(so_track_id, {rel_embed, s_embed, o_embed, verb_labels})
    │
    ▼  (after all 8 frames)
relation_classifier:
    mean_pool per tracklet → sub_class_embed / verb_class_embed / obj_class_embed
    → pred_sub_logits, pred_verb_logits, pred_obj_logits
```

## Step 4.1: Train Stage 2

```bash
python -m torch.distributed.launch \
    --master_port 47745 \
    --nproc_per_node=1 \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidor_colab_stage2.json
```

**Training details:**
- **GPUs:** 1 × T4
- **Batch size:** 1
- **Epochs:** 2 (LR drops at epoch 1)
- **Seq length:** 8 frames
- **Loss:** Focal loss only (sub/obj/verb classification, no box loss)
- **Pretrain:** Loads `checkpoint0004.pth` from Stage 1 via `resume_stage2()`

**Checkpoint loading:**
`util/checkpoints.py:resume_stage2()` (line 163-174):
- Only loads keys present in BOTH state dicts
- When shapes differ, takes the first half (`checkpoint_value[:shape[0]//2]`)
- This handles the Stage 1 → Stage 2 architecture change (discards tracking-query parameters)

**Expected time:** ~1-2 hours for 2 epochs

**Output:** `data/ckpts/vidor_stage2/checkpoint.pth`

## Step 4.2: Evaluate Stage 2

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    main.py \
    --eval \
    --dataset_config configs/vidor_colab_stage2.json \
    --resume data/ckpts/vidor_stage2/checkpoint.pth
```

**Metrics reported (per setting: overall / zero-shot / generalized-zero-shot):**

| Metric | Description |
|--------|-------------|
| **Detection mAP** | Mean Average Precision (vIoU ≥ 0.5 for subject + object trajectories) |
| **recall@50** | Fraction of GT triplets found in top-50 predictions |
| **recall@100** | Fraction of GT triplets found in top-100 predictions |
| **Tagging pre@1** | Precision@1 (top prediction correct, ignoring trajectory) |
| **Tagging pre@5** | Precision@5 |
| **Tagging pre@10** | Precision@10 |

**Zero-shot evaluation:**
Computed as `val_triplets - train_triplets` (triplets never seen during training). This is computed in `datasets/__init__.py:73` and can be slow on first run.

## Verification

```bash
# Check checkpoint
ls -lh data/ckpts/vidor_stage2/checkpoint.pth

# Reload test
python -c "
from argparse import Namespace
import json, torch
args = Namespace(**json.load(open('configs/vidor_colab_stage2.json')))
args.device = 'cpu'; args.distributed = False; args.gpu = 0
from models import build_model
model, _, _, _ = build_model(args)
ckpt = torch.load('data/ckpts/vidor_stage2/checkpoint.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
print('Stage 2 model loaded successfully')
"
```
