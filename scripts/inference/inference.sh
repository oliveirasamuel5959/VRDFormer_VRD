#!/usr/bin/env bash
# VRDFormer inference — one video: Stage 1 detections -> Stage 2 relation triplets.
#
# Run from the repo root with the Stage-1/2 venv active (see docs/RUNPOD_INFERENCE.md):
#
#   sh scripts/inference/inference.sh data/vidorsmall/videos/10042935613.mp4
#   sh scripts/inference/inference.sh clip.mp4 --every_n_frames 5 --max_frames 120
#   sh scripts/inference/inference.sh clip.mp4 --no_video --output_dir /tmp/out
#
# Writes, under <output_dir>/<video_id>/ (default output/inference/<dataset>/):
#
#   results.json           triplets, tracklets and per-frame detections
#   eval_prediction.json   sidecar in util/evaluate's prediction shape
#   video.mp4              the input annotated with boxes and triplet labels
#
# Every flag after the video path is passed straight through to inference.py, so the full
# option list in `python inference.py --help` is available. Flags given here come first, so a
# pass-through flag overrides the default below (later argparse flags win).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sh scripts/inference/inference.sh <video.mp4> [inference.py flags...]" >&2
    exit 2
fi

VIDEO=$1
shift

# Pick the interpreter. An activated venv (VIRTUAL_ENV) always wins, since that is the
# documented way to run inference; otherwise fall back to the project's .venv so the script
# works without `source .venv/bin/activate` first. A bare `python` is the last resort and is
# checked below, because it is usually the system interpreter without numpy/torch.
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

# --stage2_resolution small matches the weights in data/ckpts/vidorsmall_stage2: its saved args
# say "small" while configs/vidorsmall_stage2.json says "large", and a Stage-2 pass at a spatial
# scale the checkpoint never saw degrades its roi_align features. Stage 1 stays at "large" via
# its own config. Override with --stage2_resolution large if you retrain at that resolution.
"$PY" inference.py \
    --video "$VIDEO" \
    --device "$DEVICE" \
    --stage2_resolution small \
    "$@"
