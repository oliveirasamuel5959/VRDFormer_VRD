# RunPod 4× RTX 4090 — VidOR Stage-2 DDP Training Runbook

**Goal:** Train VRDFormer VidOR stage 2 for **2 epochs** on a RunPod pod with **4× RTX 4090** (96 GB VRAM total = 4×24 GB) using 4-GPU PyTorch DDP, monitor GPUs over SSH, run a full-val eval, and download the checkpoints to the local machine.

**Decisions (confirmed with user):**
- Data: Kaggle dataset `samuelpatricio/vrdformer-vidor` downloaded on the pod (mirrors `notebooks/colab_kaggle_train.ipynb` cells 6–7); stage-1 checkpoint scp'd from local `data/ckpts/vidor_stage1`.
- Repo changes committed to the GitHub fork, pod clones it.
- `batch_size: 1` per GPU (global batch 4 via DDP; gradient scale identical to previous single-GPU bs-1 runs → `lr 5e-5` stays valid).
- In-training eval left as-is (partial metrics under DDP); authoritative eval run separately on 1 GPU after epoch 2.

**Repo facts this plan relies on:**
- `main.py:187-188` calls `torch.set_deterministic(True)` when torch minor ≤ 8 — crashes (`AttributeError`) on torch 2.0–2.8. **Must patch** (RunPod template is 2.5.x).
- DDP: `util/dist.py` reads `RANK/WORLD_SIZE/LOCAL_RANK` env (set by `torch.distributed.launch`); per-rank losses are summed over B and DDP averages across ranks → gradient scale ∝ per-GPU batch_size only. Nothing else needs changing for 4 GPUs.
- Stage-2 checkpoints (~728 MB each) are written by rank 0 after **every** epoch to `output_dir`: `checkpoint.pth` + `checkpoint{epoch:04}.pth` (`main.py:218-231`). Epoch 0 → `checkpoint0000.pth`, epoch 1 → `checkpoint0001.pth`.
- Eval reads `data/vidor/action.txt` relative to CWD (`engine.py:233`) → **run everything from the repo root**; `vidor_path: "data/vidor"` is relative and works from repo root.
- `models/ops` (compiled extension) is **not** needed — non-deformable stage 2.

---

## 1. Local: commit RunPod config, script, and torch-2.x fix to the fork

Create and commit exactly **3 files** (add only these — the working tree has CRLF noise):

**`configs/vidor_runpod_stage2.json`** (new — based on `configs/vidor_kaggle_stage2.json`, which has the proven `train_clip_sample_ratio/amp/benchmark` choices):

```json
{
    "dataset": "vidor",
    "vidor_path": "data/vidor",
    "pretrain": "data/ckpts/vidor_stage1/checkpoint0000.pth",
    "output_dir": "data/ckpts/vidor_runpod_stage2",
    "epochs": 2,
    "lr_drop": 1,
    "num_workers": 4,
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
    "batch_size": 1,
    "train_clip_sample_ratio": 0.25,
    "amp": true,
    "benchmark": true
}
```

- `num_workers: 4` (16 was meant for 1 rank; 4 ranks × 4 = 16 total).
- `train_clip_sample_ratio: 0.25` trains on every 4th clip → ~3–6 h/epoch instead of days. Set to `1.0` only if full data per epoch is required (≈13–16 h/epoch on 1×A100 in the Colab estimate — expect ~10× the cost for 2 epochs).
- `amp: true` → bf16 on 4090 (Ampere+, `engine.py:44`); `benchmark: true` → cudnn autotune.

**`scripts/stage2/train_vidor_runpod.sh`** (new):

```bash
# VRDFormer Stage 2 — RunPod 4x RTX 4090 DDP (relative paths)
# Requires Stage 1 checkpoint at data/ckpts/vidor_stage1/checkpoint0000.pth
python -m torch.distributed.launch \
    --master_port 47745 \
    --nproc_per_node=4 \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidor_runpod_stage2.json
```

**`main.py:187-188`** (patch — fix for torch 2.x):

```python
    if int(torch.__version__.split(".")[1]) <= 8:  # for torch version<=1.8
        torch.set_deterministic(True)
```

→ replace with:

```python
    if hasattr(torch, 'set_deterministic'):  # removed in torch 2.0; newer torch skips this
        torch.set_deterministic(True)
```

Then:

```bash
git add configs/vidor_runpod_stage2.json scripts/stage2/train_vidor_runpod.sh main.py
git commit -m "RunPod: 4-GPU DDP stage2 config/script; fix torch.set_deterministic on torch 2.x"
git push origin main
```

---

## 2. RunPod: deploy the pod

1. **runpod.io → Pods → Deploy**:
   - GPU: **RTX 4090**, quantity **4** (shows 96 GB VRAM total). RAM: 344 GB (or ≥100 GB). **Container disk: 120 GB** (27.4 GB zip + ~34 GB extracted + ~1.5 GB checkpoints + code).
   - Template: `runpod/pytorch:2.5.1-py3.11-cuda12.4-devel` (matches repo pin torch 2.5.1+cu121; devel image ships `nvidia-smi` + build tools). Any `runpod/pytorch:2.5.x ... cuda12.x` devel image works. If only newer torch exists, the §1 patch makes any 2.x/3.x safe.
   - Keep **Jupyter Notebook** enabled (default) — used as backup access.
2. **Before/while deploying**: add your public SSH key in **Settings → SSH Public Keys** (`~/.ssh/id_ed25519.pub`), so the pod accepts passwordless SSH.
3. Once **Running**, open the pod page:
   - **SSH** (primary): `ssh root@<pod-ip> -p <exposed-ssh-port> -i ~/.ssh/id_ed25519` — the exact command is in the pod's Connect → "SSH over exposed TCP" dropdown.
   - **Jupyter** (backup): `https://<pod-id>-8888.proxy.runpod.net/?token=<token>` — token in pod logs.

---

## 3. Pod: system tools + repo + dependencies

```bash
apt-get update && apt-get install -y nvtop tmux htop rsync unzip
pip install -U pip
pip install 'numpy<2' decord==0.6.0 timm scipy lap opencv-python-headless tqdm
pip install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'
pip install -U kaggle gpustat
```

(`numpy<2` is required by decord 0.6.x; `cv2` comes from opencv-python-headless — the notebook relied on Colab's preinstalls.)

```bash
cd /workspace
git clone https://github.com/oliveirasamuel5959/VRDFormer_VRD.git
cd VRDFormer_VRD
```

---

## 4. Pod: Kaggle dataset + stage-1 checkpoint

From the **local WSL2 machine** (uploads credentials + stage-1 ckpts, ~1.4 GB):

```bash
scp -P <exposed-ssh-port> -i ~/.ssh/id_ed25519 ~/.kaggle/kaggle.json root@<pod-ip>:/root/.kaggle/
scp -P <exposed-ssh-port> -i ~/.ssh/id_ed25519 -r \
  "/mnt/c/Users/Samuel Oliveira/Desktop/CS/VRDFormer_VRD/data/ckpts/vidor_stage1" \
  root@<pod-ip>:/workspace/VRDFormer_VRD/data/ckpts/
```

On the **pod** (mirrors notebook cells 6–7; ~12 min at ~40 MB/s):

```bash
chmod 600 ~/.kaggle/kaggle.json
mkdir -p data/metadata data/weights data/ckpts
kaggle datasets download samuelpatricio/vrdformer-vidor
unzip -q vrdformer-vidor.zip -d data && rm vrdformer-vidor.zip
mv data/vidor/metadata/* data/metadata/ 2>/dev/null || true
mv data/vidor/detr-r101-2c7b67e5.pth data/weights/ 2>/dev/null || true
```

---

## 5. Pod: verify GPUs and data before launching

```bash
nvidia-smi   # expect 4x RTX 4090, driver + CUDA 12.x
```

```python
python - <<'EOF'
import torch, os
print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
checks = [
    'data/vidor/videos', 'data/vidor/annotations/train', 'data/vidor/annotations/val',
    'data/vidor/action.txt',
    'data/metadata/vidor_annotations.pkl',
    'data/metadata/vidor_train_frames_stage2.json', 'data/metadata/vidor_val_frames.json',
    'data/ckpts/vidor_stage1/checkpoint0000.pth',
]
for p in checks:
    print(('  OK ' if os.path.exists(p) else 'MISS'), p)
EOF
```

All must print OK (stage-2 train-frames JSON is `vidor_train_frames_stage2.json` — the notebook checked the stage-1 name).

---

## 6. Pod: launch the 2-epoch DDP training (in tmux, not a Jupyter cell)

```bash
tmux new -s train
cd /workspace/VRDFormer_VRD
sh scripts/stage2/train_vidor_runpod.sh 2>&1 | tee train_runpod.log
```

- Detach with `Ctrl-b d`; reattach anytime with `tmux a -t train`. tmux keeps training alive if SSH/Jupyter disconnects (a Jupyter cell would die with the browser session).
- Startup silence: the val loader scans the full train annotations to build `zeroshot_triplets` (`dataloader_initializer`) — several minutes before iteration logs appear. **Not a hang.**
- After the first ~50 iterations, confirm: loss finite and decreasing, `ps aux | grep main.py` shows **4 python processes**, all 4 GPUs busy (see §7). Then let it run.
- Checkpoint per epoch appears in `data/ckpts/vidor_runpod_stage2/` (`checkpoint.pth` + `checkpoint0000.pth`, then `checkpoint0001.pth`) plus `config.json` and `log.txt`.

**ETA math** (check against the log's it/s): ≈282,389 train clips × 0.25 ratio ≈ 70,600 clips ÷ 4 GPUs ≈ **17,650 steps/epoch**; 2 epochs ≈ 3–6 h at a healthy 0.5–1 it/s per GPU.

---

## 7. GPU monitoring over SSH

- **nvtop** (best overview): `nvtop` — per-GPU utilization, VRAM, temp, power, per-process bars.
- Live samples: `watch -n 2 nvidia-smi`
- Time series: `nvidia-smi dmon -s pucvmet -d 2` (power/util/clock/mem/temp/enc every 2 s) or `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 2`
- Compact: `gpustat -i 1`
- RAM/CPU: `htop`
- **RunPod web UI**: pod page shows per-GPU utilization/VRAM/temp graphs without SSH.
- Healthy signs: 4 python procs on 4 distinct GPUs; util ≥ ~85%; temps ≤ 85 °C; power 250–450 W. Stage 2 at bs-1 is light on VRAM (expect ~8–16 GB/GPU) — that's normal.
- Progress: `tail -f train_runpod.log`.

---

## 8. After training: authoritative eval on 1 GPU

The per-epoch in-training eval prints metrics computed on each rank's ¼ of the val set — ignore those. Run the real eval after epoch 2:

```bash
cd /workspace/VRDFormer_VRD
python -m torch.distributed.launch --nproc_per_node=1 main.py --eval \
    --dataset_config configs/vidor_runpod_stage2.json \
    --resume data/ckpts/vidor_runpod_stage2/checkpoint0001.pth
```

Reports relation detection (mean AP, recall@50/100) and tagging (precision@1/5/10) — overall / zero-shot / generalized-zero-shot. (`eval_stage2` runs without `no_grad`, so 24 GB VRAM at batch 1 is fine.)

---

## 9. Download checkpoints to the local machine

**Do this before terminating the pod** — pod disk is destroyed on terminate. From local WSL2:

```bash
mkdir -p ~/vrdformer_runpod_stage2
rsync -avP -e "ssh -p <exposed-ssh-port> -i ~/.ssh/id_ed25519" \
  root@<pod-ip>:/workspace/VRDFormer_VRD/data/ckpts/vidor_runpod_stage2/ \
  ~/vrdformer_runpod_stage2/
```

Downloads ~1.5 GB (2 × 728 MB checkpoints + `config.json` + `log.txt`; epoch-0 `checkpoint0000.pth` included, epoch-1 final `checkpoint0001.pth` is the one to keep). Alternatives: `scp -P <port> -i key root@<ip>:<path> .` per file, or the pod's web file browser.

Optionally copy into the local repo for dev/eval: `cp -r ~/vrdformer_runpod_stage2/* data/ckpts/vidor_runpod_stage2/`.

---

## 10. Cleanup

- Terminate the pod (Stop → Delete) to stop billing. If more runs are planned, consider a **network volume** for the dataset next time (avoids the 12-min Kaggle download per pod).

---

## Risks & gotchas

1. **`torch.set_deterministic` crash** on torch 2.0–2.8 — fixed by the §1 patch. If the pod template ships torch 2.9+, it's a no-op anyway.
2. **NaN loss**: `engine.py:22` exits only that rank; the other 3 hang on the 10-day NCCL timeout (`util/dist.py:155`). Detect via `ps aux | grep main.py` (expect exactly 4 procs) — if fewer, `pkill -f main.py`, fix, relaunch.
3. **Colab's `weights_only` unpickling crash** (notebook cell-20) is already fixed in the current local `util/checkpoints.py:191` (`weights_only=False`) — confirm the pushed commit includes it.
4. **CRLF noise**: the local working tree shows every tracked file as modified — `git add` only the 3 intended files from §1.
5. **Disk**: 120 GB container disk minimum (zip + extraction + checkpoints). Don't pick the default small disk.
6. **`--eval` must use `--resume`** (stage-2 checkpoint), never `--pretrain` (stage-1 weights) — the latter loads stage-1 weights into the stage-2 model and produces garbage metrics.
7. **Val loader init is slow** (zeroshot triplet diff) — minutes of silence at startup is expected.
