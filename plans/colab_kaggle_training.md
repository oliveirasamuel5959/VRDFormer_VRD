# Plan: Colab T4 Training with Kaggle-Hosted Data

## Context

The user has no Google Drive space. The 16GB `data/vidor/` (13GB videos + 2.7GB annotations) will be hosted on Kaggle as a dataset. The code repo will be pushed to a GitHub fork so Colab can clone it directly. At runtime, the Kaggle dataset is downloaded and extracted into the cloned repo.

**Critical pre-requisite:** The current git repo is at `/mnt/c/Users/Samuel Oliveira/` (home directory level) with zero tracked files — broken. A new git repo must be initialized inside `VRDFormer_VRD/` before anything can be pushed.

## What Exists (accurate sizes)

| Resource | Size |
|----------|------|
| `data/vidor/videos/` (3,799 mp4) | 13 GB |
| `data/vidor/annotations/` (7,000 train + 835 val JSONs) | 2.7 GB |
| `data/vidor/*.txt` + `train_files.json` | ~80 KB |
| `data/metadata/` (sharded batches + frame JSONs) | 1.3 GB |
| `data/weights/detr-r101-2c7b67e5.pth` | 232 MB |
| **Kaggle dataset total** | **~17.5 GB** |
| Code (all .py, configs, docs, scripts) | ~2 MB |

## Implementation Steps

### Step 1: Init Git Repo + Push to GitHub Fork

**1a.** Create a fork of `zhengsipeng/VRDFormer_VRD` on GitHub (web UI: click "Fork").

**1b.** Initialize a proper git repo inside VRDFormer_VRD:
```bash
cd /mnt/c/Users/Samuel\ Oliveira/Desktop/CS/VRDFormer_VRD
git init
```

**1c.** Create `.gitignore`:
```
data/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.vscode/
.idea/
*.egg-info/
build/
dist/
```

**1d.** Add all code and commit:
```bash
git add .gitignore
git add main.py engine.py models/ datasets/ util/ configs/ scripts/ docs/ spec/ notebooks/
git commit -m "VRDFormer codebase with Kaggle-Colab configs and bug fixes"
```

**1e.** Add fork remote and push:
```bash
git remote add origin https://github.com/<YOUR_USERNAME>/VRDFormer_VRD.git
git branch -M main
git push -u origin main
```

### Step 2: Bundle + Upload the Kaggle Dataset (local, one-time)

```bash
cp -r data/metadata data/vidor/metadata_bundle
cp data/weights/detr-r101-2c7b67e5.pth data/vidor/
```

Create `data/vidor/kaggle_dataset_metadata/dataset-metadata.json`:
```json
{"title": "VRDFormer-VidOR-Training", "id": "vrdformer-vidor", "licenses": [{"name": "CC0-1.0"}]}
```

Upload via Kaggle web UI at `https://www.kaggle.com/datasets/create` (zip upload, max 20GB — 17.5GB fits). Or use CLI:
```bash
pip install kaggle
# Place kaggle.json API key from kaggle.com/settings → API → Create New Token
kaggle datasets create -p data/vidor --dir-mode zip
```

### Step 3: Create Relative-Path Configs

**`configs/vidor_kaggle_stage1.json`:**
- `vidor_path: "data/vidor"`, `pretrain: "data/weights/detr-r101-2c7b67e5.pth"`, `output_dir: "data/ckpts/vidor_stage1"`
- All other fields same as `vidor_colab_stage1.json`

**`configs/vidor_kaggle_stage2.json`:**
- `vidor_path: "data/vidor"`, `pretrain: "data/ckpts/vidor_stage1/checkpoint0004.pth"`, `output_dir: "data/ckpts/vidor_stage2"`
- All other fields same as `vidor_colab_stage2.json`

### Step 4: Create the Colab Notebook

**`notebooks/colab_kaggle_train.ipynb`** — 8 cells:

| Cell | Purpose | Content |
|------|---------|---------|
| 1 | Clone & install | `git clone` fork, `pip install decord timm scipy lap`, `pip install` custom pycocotools fork |
| 2 | Download Kaggle data | Upload kaggle.json, `kaggle datasets download`, unzip, move metadata/weights into place |
| 3 | Verify | `nvidia-smi`, check GPU, verify all paths exist |
| 4 | Smoke test (optional) | 1 epoch with `vidorpart_local_stage1.json` to verify pipeline |
| 5 | Stage 1 train | 5 epochs, ~4-6 hours, `vidor_kaggle_stage1.json` |
| 6 | Stage 2 train | 2 epochs, ~1-2 hours, `vidor_kaggle_stage2.json` |
| 7 | Evaluate | Detection mAP, recall@50/100, tagging precision |
| 8 | Download checkpoints | Zip `data/ckpts/` and download |

### Step 5: Create Shell Wrappers + Guide

- `scripts/stage1/train_kaggle.sh`
- `scripts/stage2/train_kaggle.sh`
- `spec/vrdformer_train_pipeline/KAGGLE_COLAB_GUIDE.md`

### Step 6: Commit Everything and Push

All new files + bug fixes committed and pushed to the GitHub fork so Colab can clone.

## Files to Create

| File | Purpose |
|------|---------|
| `.gitignore` | Exclude data/, caches, IDE files |
| `configs/vidor_kaggle_stage1.json` | Stage 1 config, relative paths |
| `configs/vidor_kaggle_stage2.json` | Stage 2 config, relative paths |
| `notebooks/colab_kaggle_train.ipynb` | Complete Colab notebook |
| `scripts/stage1/train_kaggle.sh` | Stage 1 convenience wrapper |
| `scripts/stage2/train_kaggle.sh` | Stage 2 convenience wrapper |
| `data/vidor/kaggle_dataset_metadata/dataset-metadata.json` | Kaggle upload metadata |
| `spec/vrdformer_train_pipeline/KAGGLE_COLAB_GUIDE.md` | Complete step-by-step guide |

## Already Fixed (will be included in the push)

| File | Fix |
|------|-----|
| `engine.py:18` | `datetime.now()` → `datetime.datetime.now()` |
| `util/checkpoints.py:36-44` | Removed 4 `pdb.set_trace()` calls |
| `models/vrdformer.py:159` | `pdb.set_trace()` → `RuntimeError` |
| `scripts/stage1/train_vidorpart_smoke.sh` | Already created in previous step |

## Colab Runtime Summary

| Phase | Time | VRAM |
|-------|------|------|
| Download + unzip Kaggle dataset (17.5GB) | 15–30 min | — |
| Stage 1 training (5 epochs, batch=2) | 4–6 hours | ~13–15 GB |
| Stage 2 training (2 epochs, batch=1) | 1–2 hours | ~10–12 GB |
| Evaluation | ~20 min | ~8 GB |

## Updating the Kaggle Dataset (e.g. regenerated PKL)

When `data/metadata/vidor_annotations.pkl` is regenerated via `prepare.py`, push a new version of the Kaggle dataset:

### Option A — Kaggle Web UI (fastest, 30s)

1. Go to your dataset page: `https://www.kaggle.com/datasets/samuelpatricio/vrdformer-vidor-training`
2. Click **New Version**
3. Delete the old `metadata/vidor_annotations.pkl` and drag in the new one
4. Version note: `"Regenerated vidor_annotations.pkl with updated prepare.py"`
5. Click **Create Version**

### Option B — Kaggle CLI (scriptable)

First-time setup:
```bash
sudo apt install python3-pip -y
pip install kaggle
# kaggle.json API key already at ~/.kaggle/kaggle.json
```

Then push the update:
```bash
mkdir -p /tmp/kaggle_update/metadata
cp data/metadata/vidor_annotations.pkl /tmp/kaggle_update/metadata/
cat > /tmp/kaggle_update/dataset-metadata.json << 'EOF'
{"title": "VRDFormer-VidOR-Training", "id": "samuelpatricio/vrdformer-vidor-training", "licenses": [{"name": "CC0-1.0"}]}
EOF
kaggle datasets version -p /tmp/kaggle_update -m "Update vidor_annotations.pkl"
```

### Option C — Download, merge, re-upload

If you need to update multiple files while preserving others:
```bash
kaggle datasets download samuelpatricio/vrdformer-vidor-training -p /tmp/kaggle_full/
unzip /tmp/kaggle_full/vrdformer-vidor-training.zip -d /tmp/kaggle_full/
cp data/metadata/vidor_annotations.pkl /tmp/kaggle_full/metadata/
kaggle datasets version -p /tmp/kaggle_full -m "Update metadata PKL"
```

Last PKL generated: 2026-08-03 (88 KB, from `data/prepare.py --func get_anno --dbname vidor`)

## Verification

1. New configs are valid JSON
2. `git init` + commit + push succeeds
3. Clone fork in Colab → Cell 3 passes (GPU + data verification)
4. Optional: 1-epoch smoke test before full 5-epoch run
