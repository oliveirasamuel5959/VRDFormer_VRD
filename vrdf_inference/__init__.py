"""Single-video inference for VRDFormer, split by responsibility.

``inference.py`` at the repository root is the entry point; every module here is importable on its
own. See ``docs/superpowers/specs/2026-10-18-inference-modularization-design.md``.
"""

__all__ = [
    "checkpoints",
    "cli",
    "console",
    "render",
    "results",
    "stage1",
    "stage2",
    "tracklets",
    "video",
]
