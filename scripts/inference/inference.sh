#!/usr/bin/env bash
# VRDFormer single-video inference.
#
#   sh scripts/inference/inference.sh stage1 clip.mp4
#   sh scripts/inference/inference.sh stage2 clip.mp4 --every_n_frames 5 --max_frames 120
#
# stage1: detect and annotate only. stage2: detect, associate, classify into relation triplets, and
# annotate. Both write an annotated mp4 plus results.json under
# output/inference/<mode>/<video_id>/.
#
# Every flag after the two positional arguments is passed straight through to inference.py, so the
# full option list in `python inference.py --help` is available. Later argparse flags win, so a
# pass-through flag overrides anything this script sets first.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: sh scripts/inference/inference.sh <stage1|stage2> <video.mp4> [inference.py flags...]" >&2
    exit 2
fi

MODE=$1
VIDEO=$2
shift 2

case "$MODE" in
    stage1|stage2) ;;
    *)
        echo "[inference.sh] mode must be stage1 or stage2 (got '$MODE')" >&2
        exit 2
        ;;
esac

# Pick the interpreter. An activated venv (VIRTUAL_ENV) always wins, since that is the documented
# way to run inference; otherwise fall back to the project's .venv so the script works without
# `source .venv/bin/activate` first. A bare `python` is the last resort and is checked below,
# because it is usually the system interpreter without numpy/torch.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PY=python
elif [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python
fi

if ! "$PY" -c "import numpy, torch" >/dev/null 2>&1; then
    echo "[inference.sh] $PY cannot import numpy/torch." >&2
    echo "[inference.sh] Create the environment and activate it first:" >&2
    echo "[inference.sh]   bash scripts/runpod/install_stage1_deps.sh && source .venv/bin/activate" >&2
    exit 1
fi

# Pick the device rather than defaulting to CUDA: an explicit --device cuda on a CPU-only box
# stops with an error, which is a confusing first run on a workstation. An explicit --device
# from the caller still wins.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE=cuda
else
    DEVICE=cpu
    echo "[inference.sh] no usable GPU found; falling back to --device cpu (slow)" >&2
fi

# No --stage2_resolution here: the stage-2 checkpoint's saved "small" outranks
# configs/vidorsmall_stage2.json's "large" now, so stage 2 uses the scale its weights were trained
# at without being told. Override explicitly if you retrain at another resolution.
"$PY" inference.py \
    --video "$VIDEO" \
    --mode "$MODE" \
    --device "$DEVICE" \
    "$@"
