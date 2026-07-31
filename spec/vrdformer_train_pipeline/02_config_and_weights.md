# Phase 2: Configuration & Weights Setup

## Step 2.1: Download DETR Pretrained Weights

VRDFormer Stage 1 starts from a COCO-pretrained DETR checkpoint (ResNet-101 backbone + transformer).

```bash
mkdir -p data/weights
wget https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth \
     -O data/weights/detr-r101-2c7b67e5.pth
```

**Size:** ~164 MB

**How it works:** `util/checkpoints.py:resume_value()` maps DETR's single-object heads to VRDFormer's duplicated `sub_*`/`obj_*`/`verb_*` heads:
- `class_embed.weight` → `sub_class_embed`, `obj_class_embed`, `verb_class_embed` (with class count adjustment)
- `bbox_embed` → `sub_bbox_embed`, `obj_bbox_embed`
- Shape mismatches (e.g., norm layers) handled via repeat/truncation
- Missing keys initialized randomly (from scratch)

The backbone is always ImageNet-pretrained via torchvision (`pretrained=True` is hardcoded in `backbone.py:101`), so no separate backbone weights needed.

## Step 2.2: Config Files

The original configs hardcode `/home/zhengsipeng/data/vidor` and have several bugs. We create clean local versions.

### Config for Local Testing (smoke test)

**`configs/vidorpart_local_stage1.json`:**
```json
{
    "dataset": "vidorpart",
    "vidor_path": "data/vidor",
    "pretrain": "data/weights/detr-r101-2c7b67e5.pth",
    "output_dir": "data/ckpts/vidorpart_stage1",
    "batch_size": 4,
    "epochs": 5,
    "lr_drop": 4,
    "num_workers": 4,
    "max_duration": 24,
    "seq_len": 2,
    "cautious": true,
    "by_ratio": false,
    "stage": 1,
    "multi_frame_attention": true,
    "multi_frame_encoding": true,
    "merge_frame_features": false,
    "multi_frame_attention_separate_encoder": true,
    "position_embedding": "sine_3d_v2",
    "num_feature_levels": 1,
    "tracking": true,
    "track_prev_frame_range": 8,
    "focal_loss": true,
    "aux_loss": false,
    "debug": true
}
```

Key: `"dataset": "vidorpart"` caps at 100 videos, `"debug": true` sets `num_workers=0` and skips raw annotation loading.

### Config for Colab Training

**`configs/vidor_colab_stage1.json`:**
```json
{
    "dataset": "vidor",
    "vidor_path": "/content/drive/MyDrive/VRDFormer/data/vidor",
    "pretrain": "/content/drive/MyDrive/VRDFormer/data/weights/detr-r101-2c7b67e5.pth",
    "output_dir": "/content/drive/MyDrive/VRDFormer/data/ckpts/vidor_stage1",
    "batch_size": 2,
    "epochs": 5,
    "lr_drop": 4,
    "num_workers": 2,
    "max_duration": 24,
    "seq_len": 2,
    "cautious": true,
    "by_ratio": false,
    "stage": 1,
    "multi_frame_attention": true,
    "multi_frame_encoding": true,
    "merge_frame_features": false,
    "multi_frame_attention_separate_encoder": true,
    "position_embedding": "sine_3d_v2",
    "num_feature_levels": 1,
    "tracking": true,
    "track_prev_frame_range": 8,
    "focal_loss": true,
    "aux_loss": false
}
```

Adjustments for Colab:
- `batch_size: 2` (T4 has 16GB VRAM; 4 may OOM with ResNet-101 + frame pairs)
- `num_workers: 2` (Colab limits)
- All paths use `/content/drive/MyDrive/VRDFormer/` prefix

**`configs/vidor_colab_stage2.json`:**
```json
{
    "dataset": "vidor",
    "vidor_path": "/content/drive/MyDrive/VRDFormer/data/vidor",
    "pretrain": "/content/drive/MyDrive/VRDFormer/data/ckpts/vidor_stage1/checkpoint0004.pth",
    "output_dir": "/content/drive/MyDrive/VRDFormer/data/ckpts/vidor_stage2",
    "epochs": 2,
    "lr_drop": 1,
    "num_workers": 2,
    "max_duration": 24,
    "seq_len": 8,
    "cautious": true,
    "by_ratio": false,
    "stage": 2,
    "multi_frame_attention": false,
    "multi_frame_encoding": false,
    "merge_frame_features": false,
    "multi_frame_attention_separate_encoder": false,
    "position_embedding": "sine_3d_v2",
    "num_feature_levels": 1,
    "focal_loss": true,
    "batch_size": 1
}
```

**CRITICAL FIX:** Original `vidor_stage2.json` uses `"vidvrd_path"` instead of `"vidor_path"`. The VidOR dataset class (`datasets/vidor.py`) reads `args.vidor_path`, so `vidvrd_path` = empty path = crash. This is fixed above.

## Step 2.3: Bug Fixes in Existing Configs

These existing config files need manual fixes (or use our new ones instead):

| File | Bug | Fix |
|------|-----|-----|
| `configs/vidor_stage2.json` | `"vidvrd_path"` field | Change to `"vidor_path"` |
| `configs/vidor_stage1_deform.json` | `"dataset": "vidvrd"` + `"vidvrd_path"` | Change to `"vidor"` + `"vidor_path"` |
| `configs/vidor_stage2_deform.json` | `"vidvrd_path"` field | Change to `"vidor_path"` |

## Step 2.4: Colab Notebook Structure

Create `plan/vrdformer_train_pipeline/colab_train.ipynb` with these cells:

1. **Mount Drive + install dependencies:**
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install decord timm scipy lap pycocotools
# PyTorch comes pre-installed on Colab with CUDA
```

2. **Clone/setup repo:**
```python
%cd /content/drive/MyDrive/VRDFormer
# Verify GPU
!nvidia-smi
import torch; print(torch.cuda.is_available())
```

3. **Run Stage 1 training** (see Phase 3)

4. **Run Stage 2 training** (see Phase 4)

5. **Run evaluation** (see Phase 4)

6. **Save checkpoints to Drive** (automatic if `output_dir` is on Drive)

## Verification

```bash
# Check configs are valid JSON
python -c "import json; json.load(open('configs/vidorpart_local_stage1.json'))"
python -c "import json; json.load(open('configs/vidor_colab_stage1.json'))"
python -c "import json; json.load(open('configs/vidor_colab_stage2.json'))"

# Check DETR weights downloaded
ls -lh data/weights/detr-r101-2c7b67e5.pth

# Quick model build test (CPU, local)
python -c "
from argparse import Namespace
import json
args = Namespace(**json.load(open('configs/vidorpart_local_stage1.json')))
args.device = 'cpu'; args.distributed = False; args.gpu = 0
from models import model_initializer
model, _, criterion, n = model_initializer(args, args.device)
print(f'Model built: {n/1e6:.1f}M params')
"
```
