# VRDFormer training/eval image — CUDA 12.8 + Python 3.12 (vanilla transformer path).
#
# The repo itself is bind-mounted at /app at runtime (see docs/DOCKER.md); this image
# only carries the Python environment. It mirrors the validated RunPod stage-1 baseline:
#   scripts/runpod/install_stage1_deps.sh + requirements-runpod-stage1.txt
#
# No source build: the deformable-attention CUDA extension (models/ops) is intentionally
# not built here — it is only needed for the *_deform.json configs, which are out of scope.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/cache/torch \
    PATH=/opt/venv/bin:$PATH

# Python 3.12 (Ubuntu 24.04 default), a compiler + headers for any source-built wheel
# (e.g. lap), and git for util/dist.py:get_sha().
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-venv python3-pip python3-dev build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Self-contained virtualenv, sidestepping Ubuntu 24.04's PEP-668 externally-managed Python.
RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

# PyTorch stack from the CUDA 12.8 wheel index (the tested baseline). The exact
# `==X.Y.Z+cu128` pins guarantee the CUDA wheel is chosen over PyPI's CPU build.
RUN python -m pip install \
        torch==2.8.0+cu128 \
        torchvision==0.23.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

# Remaining project deps, constrained to the exact CUDA build so transitive deps (timm)
# never pull a different torch. requirements-runpod-stage1.txt intentionally omits torch.
COPY requirements-runpod-stage1.txt /tmp/requirements-runpod-stage1.txt
RUN printf 'torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\n' > /tmp/torch-constraints.txt \
    && python -m pip install \
        --constraint /tmp/torch-constraints.txt \
        --prefer-binary \
        --only-binary=pycocotools \
        --upgrade-strategy only-if-needed \
        -r /tmp/requirements-runpod-stage1.txt \
    && rm -f /tmp/torch-constraints.txt /tmp/requirements-runpod-stage1.txt

WORKDIR /app
CMD ["bash"]
