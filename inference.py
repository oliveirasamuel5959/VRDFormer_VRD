"""Single-video inference for VRDFormer. Entry point; the pipeline lives in ``vrdf_inference``.

    python inference.py --video clip.mp4 --mode stage1   # detection + annotated video
    python inference.py --video clip.mp4 --mode stage2   # + tracklets, triplets, annotations

Both modes always write an annotated ``video.mp4`` and a ``results.json`` under
``output/inference/<mode>/<video_id>/``. See ``docs/RUNPOD_INFERENCE.md`` for the guides per mode
and ``docs/superpowers/specs/2026-10-18-inference-modularization-design.md`` for the design.
"""

import sys
from pathlib import Path

# The package imports the repository's own top-level modules (models, util, datasets, main), which
# only resolve with the repo root on sys.path. Adding it here means inference.py works regardless
# of the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vrdf_inference.cli import main

if __name__ == "__main__":
    main()
