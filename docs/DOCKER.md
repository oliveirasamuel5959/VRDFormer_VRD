# Docker

Container packaging for **training and evaluation only** — no code changes. The image carries
the Python 3.12 + PyTorch 2.8.0+cu128 (CUDA 12.8) environment; the repository and datasets are
bind-mounted in at runtime, so checkpoints and outputs land on the host.

The image targets the **vanilla (non-deformable) transformer** path. The `*_deform.json` configs
need a compiled `models/ops` CUDA extension and are out of scope (see
[Out of scope](#out-of-scope)).

## Host prerequisites

- NVIDIA driver **≥ 570.x** (CUDA 12.8) and `nvidia-container-toolkit`.
- Docker (with Compose v2 for the `docker-compose.yml` path).

## Build

```bash
docker build -t vrdformer:cu128 .
```

## Run

The scripts assume the **repo root** as their working directory, so mount the repo at `/app`
and run with `-w /app`.

### `docker run`

```bash
docker run --rm --gpus all \
  -v "$PWD:/app" \
  --shm-size=16g \
  -w /app \
  -e NPROC_PER_NODE=2 \
  vrdformer:cu128 \
  sh scripts/stage1/train_vidorsmall.sh
```

Drop into a shell instead (then run any `sh scripts/...` command interactively):

```bash
docker run --rm -it --gpus all -v "$PWD:/app" --shm-size=16g -w /app \
  -e NPROC_PER_NODE=2 vrdformer:cu128
```

### `docker compose`

```bash
docker compose run --rm vrdformer                                    # interactive shell
docker compose run --rm vrdformer sh scripts/stage2/eval_vidorsmall.sh
NPROC_PER_NODE=2 docker compose run --rm vrdformer sh scripts/stage2/train_vidorsmall.sh
```

## Mount contract

| Path in container | Purpose | Mode |
| --- | --- | --- |
| `/app` (whole repo) | `configs/`, `main.py`, `models/`, `datasets/`, `util/`, and `data/` | bind-mounted (`$PWD`) |
| `/app/data/weights/` | DETR pretrained backbone (`detr-r101-2c7b67e5.pth`) | read (part of `/app`) |
| `/app/data/metadata/` | `*_annotations.pkl`, `*_frames*.json` | read (part of `/app`) |
| `/app/data/<dataset>/` | `videos/*.mp4`, `annotations/`, `action.txt`, `obj.txt` | read (part of `/app`) |
| `/app/data/ckpts/` | training checkpoints + stage-2 `pretrain` source | **read-write** (persists on host) |
| `/app/output/` | inference artifacts | **read-write** (persists on host) |
| `/cache/torch` (`TORCH_HOME`) | ImageNet backbone weights downloaded on first run | named volume, persists |

## First-run backbone download

`models/backbone.py` builds the ResNet backbone with `pretrained=True`, downloading ImageNet
weights into `TORCH_HOME` (`/cache/torch`) on the first model build. Either allow outbound
network, or pre-populate the `vrdformer_torch_cache` volume. After the first successful run the
weights are cached and no network is needed.

## Notes for the 2-GPU box

- `scripts/stage2/train_vidorsmall.sh` reads `NPROC_PER_NODE` (default 4) and scales
  `--accumulate_steps` inversely — set `NPROC_PER_NODE=2`.
- `scripts/stage1/train_vidorsmall.sh` hardcodes `torchrun --standalone --nproc_per_node=4`
  in its active line, which would exceed 2 GPUs. Run the equivalent command directly instead:

  ```bash
  torchrun --standalone --nproc_per_node=2 main.py \
      --accumulate_steps 1 --lr_backbone 1e-5 --lr 5e-5 --num_queries 200 \
      --dataset_config configs/vidorsmall_stage1.json
  ```

- The full-dataset scripts (`scripts/stage1/train.sh`, `train_vidor.sh`, …) hardcode
  `--nproc_per_node=8` and won't fit; use the `vidorsmall` scripts on this box.
- `--shm-size=16g` is a safe default for NCCL + DataLoader workers; the repo's RunPod notes
  (`scripts/runpod/diagnose_hang.sh`) document `/dev/shm` and NCCL P2P tuning if NCCL hangs.

## Verify the image

```bash
# 1. Build
docker build -t vrdformer:cu128 .

# 2. Environment sanity (fast, no data, no GPU needed)
docker run --rm vrdformer:cu128 python -c \
  "import torch, torchvision, decord, pycocotools, timm, scipy, cv2; \
   print(torch.__version__, torch.cuda.is_available())"
# -> 2.8.0+cu128 True   (torch.cuda.is_available() reports True only with --gpus)

# 3. GPU count (on the target box)
docker run --rm --gpus all vrdformer:cu128 python -c \
  "import torch; print(torch.cuda.device_count())"
# -> 2
```

## Out of scope

- **CPU-only training** — the train/eval scripts hardcode the NCCL backend (`util/dist.py`),
  so they need CUDA. CPU is only viable for `data/prepare.py` and
  `inference.py --device cpu`, both unchanged by this packaging.
- **Deformable-attention extension** (`models/ops`) — requires nvcc/CUDA devel; needed only by
  the `*_deform.json` configs.
- Pre-existing repo issues left untouched: the `epython` typo in
  `scripts/stage2/train_deform.sh` / `train_vidor_deform.sh`, and the
  `/home/zhengsipeng/data/...` absolute paths in the non-small configs.
