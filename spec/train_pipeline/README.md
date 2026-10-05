# VRDFormer Full Training Pipeline Plan

## Overview

```
Phase 1: Data Prep     Phase 2: Config        Phase 3: Stage 1 Train      Phase 4: Stage 2 Train      Phase 5: Inference

  Videos + JSONs  ──►  Fix paths + DETR wt ──► Smoke test ──► Full S1 ──► S2 train ──► Eval ──► .mp4 ──► triplets
```

This directory contains the full step-by-step plan to go from raw VidOR data to a trained VRDFormer model capable of inference on new videos.

## Quick Reference

| Phase | What | Time (T4 GPU) | Output |
|-------|------|---------------|--------|
| 1 | Data preparation | ~30 min (local) | `data/metadata/vidor_*.pkl`, frame JSONs |
| 2 | Config + weights | ~5 min | `data/weights/detr-r101-*.pth` (164MB), config JSONs |
| 3 | Stage 1 training | ~4-6 hours | `data/ckpts/vidor_stage1/checkpoint0004.pth` |
| 4 | Stage 2 training | ~1-2 hours | `data/ckpts/vidor_stage2/checkpoint.pth` |
| 5 | Inference | ~seconds per video | `results.json` with `<S,P,O>` triplets |

## Where to run

- **Phase 1-2:** Local machine (WSL2/CPU, data reorganization only)
- **Phase 3-4:** Google Colab with T4 GPU
- **Phase 5:** Any GPU machine (Colab or local with CUDA)

## Files in this plan

- `01_data_preparation.md` — Reorganize data, run prepare.py
- `02_config_and_weights.md` — Fix config paths, download DETR weights
- `03_stage1_training.md` — Smoke test + full Stage 1 training
- `04_stage2_training.md` — Stage 2 training + evaluation
- `05_inference.md` — Inference design for new .mp4 videos
- `colab_train.ipynb` — Colab notebook template for training
- `inference.py` — (to be implemented) Inference script
