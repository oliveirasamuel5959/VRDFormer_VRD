# Kaggle + Colab T4 Training Guide

How to train VRDFormer on a Colab T4 GPU with data hosted on Kaggle (no Google Drive space needed).

## Overview

```
Local WSL2                        Kaggle                     Colab T4
─────────────                    ─────────                  ──────────
data/vidor/ ───upload──►  Kaggle Dataset  ───download──►  data/vidor/
(16GB)                   (17.5GB zip)                     (cloned repo)

GitHub Fork
────────────
VRDFormer_VRD ───git clone──►  /content/VRDFormer_VRD/
(+ configs, notebook, fixes)
```

## Part A: Upload Data to Kaggle (one-time, from WSL2)

### 1. Bundle metadata and weights into data/vidor

```bash
cd /mnt/c/Users/Samuel\ Oliveira/Desktop/CS/VRDFormer_VRD
cp -r data/metadata data/vidor/metadata_bundle
cp data/weights/detr-r101-2c7b67e5.pth data/vidor/
```

### 2. Upload to Kaggle

**Option A — Web UI (recommended for 17.5GB):**
1. Go to https://www.kaggle.com/datasets/create
2. Zip `data/vidor/` (right-click → Send to → Compressed folder on Windows, or `zip -r vidor.zip data/vidor/` on WSL)
3. Upload the zip, set title to "VRDFormer-VidOR-Training"
4. Make it **Private** (or Public if you prefer)

**Option B — Kaggle CLI:**
```bash
pip install kaggle
# 1. Get API key: kaggle.com/settings → API → Create New Token → downloads kaggle.json
# 2. Place at ~/.kaggle/kaggle.json, chmod 600
kaggle datasets create -p data/vidor --dir-mode zip
```

The dataset already has `kaggle_dataset_metadata/dataset-metadata.json` with the required metadata.

## Part B: Push Code to GitHub Fork (one-time, from WSL2)

### 1. Fork the repo

Go to https://github.com/zhengsipeng/VRDFormer_VRD → click **Fork**.

### 2. Init git and push

```bash
cd /mnt/c/Users/Samuel\ Oliveira/Desktop/CS/VRDFormer_VRD
git init
git add .gitignore
git add main.py engine.py models/ datasets/ util/ configs/ scripts/ docs/ spec/ notebooks/ plans/
git commit -m "VRDFormer codebase with Kaggle-Colab configs and bug fixes"
git remote add origin https://github.com/YOUR_USERNAME/VRDFormer_VRD.git
git branch -M main
git push -u origin main
```

## Part C: Train on Colab T4

### 1. Open the notebook

Upload `notebooks/colab_kaggle_train.ipynb` to Colab, or open it directly from your GitHub fork:
```
https://colab.research.google.com/github/YOUR_USERNAME/VRDFormer_VRD/blob/main/notebooks/colab_kaggle_train.ipynb
```

### 2. Get your Kaggle API key

- Go to https://www.kaggle.com/settings
- Click **API** → **Create New Token**
- Downloads `kaggle.json`

### 3. Run the cells in order

| Cell | What it does | Time |
|------|-------------|------|
| 1 | Clone repo from YOUR GitHub fork, pip install deps | ~2 min |
| 2 | Upload kaggle.json, download + unzip dataset | 15-30 min |
| 3 | Verify GPU, VRAM, all data paths | <1 min |
| 4 | (Optional) Smoke test — 1 epoch, 100 videos | ~20 min |
| 5 | **Stage 1 training** — 5 epochs | **4-6 hours** |
| 6 | **Stage 2 training** — 2 epochs | **1-2 hours** |
| 7 | Evaluation — mAP, recall, precision | ~20 min |
| 8 | Download checkpoints | ~5 min |

### 4. Edit before running

In the notebook, replace:
- `YOUR_USERNAME` → your GitHub username (Cells 1)
- `YOUR_KAGGLE_USERNAME` → your Kaggle username (Cell 2)

## Runtime Reference

| Phase | VRAM | Batch Size | Output |
|-------|------|-----------|--------|
| Stage 1 | ~13-15 GB | 2 | `data/ckpts/vidor_stage1/checkpoint0004.pth` |
| Stage 2 | ~10-12 GB | 1 | `data/ckpts/vidor_stage2/checkpoint.pth` |

## Known Issues

### Stage 1 eval NameError (harmless)
At the end of each Stage 1 epoch, you'll see:
```
NameError: name 'eval_one_epoch' is not defined
```
This is harmless — Stage 1 has no evaluation path. Training continues normally.

### NaN loss
If loss becomes NaN, `engine.py` raises `RuntimeError`. The `datetime.now()` bug has been fixed in this fork. Just restart the cell — it usually doesn't happen twice.

### Colab disconnects after ~90 min idle
Paste this in your browser's developer console (F12):
```javascript
function ClickConnect(){
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

### Deformable attention not needed
This pipeline uses the standard DETR architecture (`num_feature_levels=1`). You do NOT need to compile `models/ops/` — skip that step from the original INSTALL.md.

## After Training

Downloaded `checkpoints.zip` contains:
- `data/ckpts/vidor_stage1/checkpoint0004.pth` — Stage 1 model (~500MB)
- `data/ckpts/vidor_stage2/checkpoint.pth` — Stage 2 model (~500MB)

The Stage 2 checkpoint is what you use for inference on new videos (Phase 5).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named 'decord'" | Re-run `pip install decord` |
| OOM during Stage 1 | Reduce `batch_size` to 1 in `vidor_kaggle_stage1.json` |
| Kaggle dataset download fails | Check kaggle.json permissions (`chmod 600`) and API key validity |
| "vidor_path does not exist" | Verify Kaggle dataset extracted correctly: `ls data/vidor/videos/` |
| Checkpoint loading fails Stage 2 | Make sure Stage 1 completed all 5 epochs (needs `checkpoint0004.pth`) |
