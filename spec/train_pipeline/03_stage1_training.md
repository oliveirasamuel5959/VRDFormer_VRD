# Phase 3: Stage 1 Training — Pair Detection + Tracking

## What Stage 1 Learns

Each of the 200 learned queries predicts a **subject-object pair**:
- `sub_class` (80 VidOR object classes + 1 no-object)
- `obj_class` (80 VidOR object classes + 1 no-object)
- `verb_class` (50 VidOR predicate classes, multi-label sigmoid)
- `sub_box` (cx, cy, w, h normalized to [0,1])
- `obj_box` (cx, cy, w, h normalized to [0,1])

**Tracking mechanism:** The previous frame is run with `no_grad`, Hungarian matching links its predictions to previous targets via track IDs, and matched query embeddings (`hs_embed`) are prepended to learned static queries. Track queries get zero position encoding → attend by content similarity.

**Loss:** Focal loss (classification) + L1 + GIoU (boxes). Weight coefficients: cls=1, verb=1, bbox=5, giou=2.

**Training data:** Frame pairs (current + previous). Each sample loads 2 frames from a video and packs the earlier as `prev_image`/`prev_target`.

## Model Architecture

```
Input: (B, 3, H, W) NestedTensor
    │
    ▼
ResNet-101 (FrozenBN) → (B, 2048, H/32, W/32) + pos encoding
    │
    ▼
1×1 Conv (2048 → 256) → (B, 256, H/32, W/32)
    │
    ▼
Flatten + level_embed → (H/32*W/32, B, 256)
    │
    ▼
6× TransformerEncoder (self-attn, 8 heads, FFN=2048) → memory
    │
    ▼
6× TransformerDecoder (self-attn + cross-attn + FFN)
    queries: [M track queries | 200 static queries]
    │
    ▼
5× prediction heads (per decoder layer):
    sub_class_embed:  Linear(256 → 81)
    obj_class_embed:  Linear(256 → 81)
    verb_class_embed: Linear(256 → 50)
    sub_bbox_embed:   MLP(256 → 256 → 4)
    obj_bbox_embed:   MLP(256 → 256 → 4)
```

## Step 3.1: Local Smoke Test (100 videos)

For quick verification that everything works before spending time on Colab:

```bash
python -m torch.distributed.launch \
    --master_port 47749 \
    --nproc_per_node=1 \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidorpart_local_stage1.json \
    --epochs 2
```

**What to watch:**
1. `[info] loading processed annotations...` — loads pickle
2. Training begins with loss around 2-3
3. Loss decreases each batch
4. Sub/obj/verb class accuracy increases
5. Checkpoints saved to `data/ckpts/vidorpart_stage1/`

**Known issues:**
- Stage 1 has **no eval path** — `NameError` at epoch end when it tries to call `eval_one_epoch`. This is harmless; training continues.
- If NaN loss occurs, `engine.py:18` may crash (calls `datetime.now()` instead of `datetime.datetime.now()`). Just restart training.

## Step 3.2: Full Training on Colab (T4 GPU)

```bash
python -m torch.distributed.launch \
    --master_port 47749 \
    --nproc_per_node=1 \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidor_colab_stage1.json
```

**Training details:**
- **GPUs:** 1 × T4 (15GB VRAM)
- **Batch size:** 2 per GPU (effective batch size = 2)
- **Epochs:** 5 (LR drops at epoch 4)
- **Optimizer:** AdamW, backbone LR = 1e-5, rest = 5e-5
- **Seq length:** 2 frames (current + previous)

**Expected time:** ~4-6 hours for 5 epochs with 7,000 videos

**Output checkpoints:**
- DataLoader saves key: `checkpoint.pth` (latest, every epoch)
- `checkpoint0000.pth` through `checkpoint0004.pth` (epoch-end snapshots)
- Stage 2 will use `checkpoint0004.pth` (the last epoch)

**Monitoring training:**
```
Epoch 0: train_loss ~2.5 → ~1.8, sub_class_acc ~30% → ~60%
Epoch 1: train_loss ~1.8 → ~1.4, sub_class_acc ~60% → ~75%
Epoch 2: train_loss ~1.4 → ~1.2, sub_class_acc ~75% → ~82%
Epoch 3: train_loss ~1.2 → ~1.0, sub_class_acc ~82% → ~86%
Epoch 4: train_loss ~1.0 → ~0.9, sub_class_acc ~86% → ~88%
```

## Verification

```bash
# Check checkpoint exists
ls -lh data/ckpts/vidor_stage1/checkpoint0004.pth

# Check checkpoint content
python -c "
import torch
ckpt = torch.load('data/ckpts/vidor_stage1/checkpoint0004.pth', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Model keys: {len(ckpt[\"model\"])}')
print(f'Optimizer keys: {len(ckpt[\"optimizer\"][\"param_groups\"])}')
"

# Quick model reload test
python -c "
from argparse import Namespace
import json, torch
args = Namespace(**json.load(open('configs/vidor_colab_stage1.json')))
args.device = 'cpu'; args.distributed = False; args.gpu = 0
from models import build_model
model, _, _, _ = build_model(args)
ckpt = torch.load('data/ckpts/vidor_stage1/checkpoint0004.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
print('Model loaded successfully')
"
```
