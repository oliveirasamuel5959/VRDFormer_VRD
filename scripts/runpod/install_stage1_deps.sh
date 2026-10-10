#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS="$REPO_ROOT/requirements-runpod-stage1.txt"
VENV_DIR="${VRDFORMER_VENV:-$REPO_ROOT/.venv}"
CHECK_ONLY=0

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
else
    PYTHON_BIN=python
fi

usage() {
    printf 'Usage: %s [--check-only] [--help]\n\n' "$0"
    printf 'Install and verify Stage-1 dependencies in a repo-local venv.\n'
    printf 'The venv inherits the RunPod image PyTorch stack; this script never installs torch.\n\n'
    printf 'Environment overrides:\n'
    printf '  PYTHON             Python 3.12 executable (default: python3.12, then python)\n'
    printf '  VRDFORMER_VENV     venv path (default: <repo>/.venv)\n'
    printf '\nOptions:\n  --check-only       Verify the existing venv without installing packages\n  -h, --help         Show this help\n'
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --check-only) CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; fail "Unknown option: $1" ;;
    esac
    shift
done

[[ -f "$REPO_ROOT/main.py" && -f "$REQUIREMENTS" ]] \
    || fail "Could not locate the repository. Expected main.py and $REQUIREMENTS."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"

printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Base Python: %s\n' "$PYTHON_BIN"
printf 'Virtual environment: %s\n' "$VENV_DIR"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"This setup is pinned to the tested Python 3.12 baseline; found {sys.version.split()[0]}"
    )

try:
    import torch
    import torchvision
except Exception as exc:
    raise SystemExit(
        "Could not import the RunPod-provided torch/torchvision. Start from a CUDA-enabled "
        f"PyTorch image. Original error: {exc}"
    ) from exc

expected = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "CUDA build": "12.8",
}
actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "CUDA build": str(torch.version.cuda),
}
for name, wanted in expected.items():
    if actual[name] != wanted:
        raise SystemExit(f"Expected {name} {wanted}; found {actual[name]}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the RunPod GPU allocation and driver.")

print(f"Base Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"Visible GPUs: {torch.cuda.device_count()}")
PY

if ((CHECK_ONLY)); then
    [[ -x "$VENV_DIR/bin/python" ]] || fail "No venv found at $VENV_DIR; run without --check-only first."
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    fi

    "$VENV_DIR/bin/python" - <<'PY'
import sys
import torch
import torchvision

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"The virtual environment must use Python 3.12; found {sys.version.split()[0]}")
if torch.__version__ != "2.8.0+cu128" or torchvision.__version__ != "0.23.0+cu128":
    raise SystemExit("The virtual environment does not expose the expected RunPod torch/torchvision stack")
PY

    # Refresh build/install tooling inside the venv, not the shared base image.
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

    # Constrain transitive dependencies (notably timm) to the exact CUDA build in the image.
    CONSTRAINTS="$(mktemp)"
    trap 'rm -f "$CONSTRAINTS"' EXIT
    "$PYTHON_BIN" - <<'PY' > "$CONSTRAINTS"
import torch
import torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY

    "$VENV_DIR/bin/python" -m pip install \
        --constraint "$CONSTRAINTS" \
        --prefer-binary \
        --only-binary=pycocotools \
        --upgrade-strategy only-if-needed \
        -r "$REQUIREMENTS"
fi

"$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import version

import cv2
import decord
import lap
import numpy as np
import PIL
import pycocotools
import sys
import scipy
import timm
import torch
import torchvision
from tqdm import tqdm
from pycocotools import mask
from pycocotools.coco import COCO

expected = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "numpy": "1.26.4",
    "Cython": "3.0.12",
    "decord": "0.6.0",
    "timm": "1.0.30",
    "scipy": "1.17.1",
    "lap": "0.5.13",
    "pycocotools": "2.0.11",
    "Pillow": "11.0.0",
    "opencv-python-headless": "4.11.0.86",
    "tqdm": "4.70.1",
}
actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    **{
        name: version(name)
        for name in expected
        if name not in ("torch", "torchvision", "numpy")
    },
}
for name, wanted in expected.items():
    if actual[name] != wanted:
        raise SystemExit(f"Expected {name} {wanted}; found {actual[name]}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected PyTorch CUDA build 12.8; found {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

# Exercise the compiled COCO mask extension to catch NumPy ABI mismatches.
mask.encode(np.asfortranarray(np.zeros((4, 4, 1), dtype=np.uint8)))

# Exercise the GPU runtime before spending time on data setup or training.
x = torch.randn((256, 256), device="cuda")
y = x @ x
torch.cuda.synchronize()
assert y.is_cuda

print("Runtime package versions:")
for name, found in actual.items():
    print(f"  {name}: {found}")
print(f"OpenCV: {cv2.__version__}")
print(f"CUDA build: {torch.version.cuda}; visible GPUs: {torch.cuda.device_count()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("NumPy/COCO ABI, imports, and CUDA operation: OK")
PY

(
    cd "$REPO_ROOT"
    "$VENV_DIR/bin/python" - <<'PY'
import datasets
import main
import models

print("Project imports (main, models, datasets): OK")
PY
)

printf '\nDependency setup and checks passed. Activate with:\n  source %q/bin/activate\n' "$VENV_DIR"
printf 'Data and pretrained checkpoints must be prepared separately; no training was started.\n'
