# RunPod Stage-1 Setup

This guide moves the setup recorded in [`notebooks/vrdformer.ipynb`](../notebooks/vrdformer.ipynb) from notebook cells to a RunPod pod's Linux shell. It installs and checks the Python runtime for the **standard Stage-1 VidOR-small config**; it does not download the dataset/checkpoints or start training.

## Tested baseline

The notebook recorded this working GPU environment:

| Component | Version observed |
| --- | --- |
| CPython | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Torchvision | 0.23.0+cu128 |
| PyTorch CUDA build | 12.8 |
| NumPy | 1.26.4 |
| Cython | 3.0.12 |
| decord | 0.6.0 |
| timm | 1.0.30 |
| SciPy | 1.17.1 |
| lap | 0.5.13 |
| pycocotools | 2.0.11 |
| OpenCV (headless) | 4.11.0.86 |
| Pillow | 11.0.0 |
| tqdm | 4.70.1 |

The installer requires the matching Python/Torch/Torchvision/CUDA versions above and checks that a RunPod GPU is visible. Use a RunPod image that already contains this CUDA-enabled PyTorch build and a compatible NVIDIA driver. It intentionally does **not** install or replace PyTorch. The Python dependency pins are in [`requirements-runpod-stage1.txt`](../requirements-runpod-stage1.txt).

> These are the versions observed in the notebook, not a guarantee that every RunPod template has the same image. The shell installer will stop on a mismatch rather than silently switching the CUDA stack.

## Install from the pod shell

Open a terminal/SSH session **inside the running RunPod pod** (not the local workstation or `runpodctl` management CLI). Clone the repository if it is not already on the pod, then run the setup script from the checkout:

```bash
git clone https://github.com/oliveirasamuel5959/VRDFormer_VRD.git
cd VRDFormer_VRD
bash scripts/runpod/install_stage1_deps.sh
```

The script creates `.venv` with access to the image's installed packages, installs the pinned Stage-1 dependencies, and validates:

- Python, PyTorch, Torchvision and PyTorch's CUDA build match the supported baseline;
- core imports including `decord`, `timm`, SciPy, `lap`, OpenCV, and `pycocotools`;
- the compiled COCO mask extension can encode a small mask (this catches a NumPy binary-ABI mismatch);
- CUDA is available and a small matrix multiplication runs on the GPU; and
- the repository modules `main`, `models`, and `datasets` import successfully.

If your repository checkout or branch is already present, enter that directory and run the script there. To select a particular Python executable or venv location:

```bash
PYTHON=/usr/bin/python3.12 VRDFORMER_VENV="$HOME/venvs/vrdformer-stage1" \
  bash scripts/runpod/install_stage1_deps.sh
```

After installation, activate the venv before running project commands:

```bash
source .venv/bin/activate
```

To rerun the checks without installing anything, use:

```bash
bash scripts/runpod/install_stage1_deps.sh --check-only
```

## Important NumPy / COCO API note

Keep NumPy at `1.26.4` (below 2) with this tested dependency set. The notebook recorded `ValueError: numpy.dtype size changed` after a COCO API native extension and NumPy were installed/replaced in an incompatible order. The installer uses the pinned PyPI `pycocotools` wheel and validates both `pycocotools.coco` and the native mask extension. Avoid manually installing the GitHub COCO API fork or upgrading NumPy afterward; if dependencies change, rerun the setup checks in the same environment.

The root [`requirements.txt`](../requirements.txt) is the historical Python 3.7 dependency set, not this RunPod/Python 3.12 environment. Do not install it for this setup.

## Data and checkpoint prerequisites

The dependency script does not fetch data, weights, or train. Before training, prepare the files referenced by `configs/vidorsmall_stage1.json`:

- `data/vidorsmall/videos/`
- `data/vidorsmall/annotations/train/` and `data/vidorsmall/annotations/val/`
- `data/vidorsmall/action.txt`
- `data/metadata/vidorsmall_annotations.pkl`
- `data/metadata/vidorsmall_train_frames_stage1.json`
- `data/metadata/vidorsmall_val_frames.json`
- `data/weights/detr-r101-2c7b67e5.pth`

Use [`docs/DATA.md`](DATA.md) for the data preparation flow and confirm all dataset/checkpoint paths in the JSON match the pod. The config's `vidor_path`, `pretrain`, and `output_dir` are relative to the repository root. The config JSON is loaded over command-line defaults, so edit the JSON for settings represented there. The model uses Torchvision's pretrained ResNet backbone, which may download its ImageNet weights on the first training start if they are not already cached.

This non-deformable config uses `num_feature_levels: 1`; it does not need to compile `models/ops`. Deformable configs are a separate setup and require building the CUDA extension against the exact installed Torch/CUDA toolchain.

## Launch Stage-1 training

From the repository root, activate the environment and choose the number of available GPUs. The checked-in wrapper defaults to one process; to use the notebook's two-GPU run:

```bash
source .venv/bin/activate
NPROC_PER_NODE=2 sh scripts/stage1/train_vidorsmall.sh
```

The wrapper uses `configs/vidorsmall_stage1.json`, `--num_queries 200`, `--lr 5e-5`, and `--lr_backbone 1e-5`. Start with `NPROC_PER_NODE=1` if only one GPU is allocated. Training requires the data and checkpoint files listed above; successful dependency checks alone do not mean the run is ready.
