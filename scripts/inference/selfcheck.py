"""Pure-logic checks for the inference package.

No checkpoint, no GPU, no decord decode: every check here exercises a pure function on synthetic
input, so it runs in a second on any machine. This is the repo's stand-in for a unit-test suite
(there is no pytest here -- see CLAUDE.md).

Run: .venv/bin/python scripts/inference/selfcheck.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def console_prefix_is_stable():
    from vrdf_inference import console

    assert console.LOG == "[inference]", console.LOG


@check
def frame_selection_matches_stride():
    from vrdf_inference.video import selected_frame_ids

    assert selected_frame_ids(100, 5) == list(range(0, 100, 5))
    assert selected_frame_ids(100, 5, max_frames=3) == [0, 5, 10]
    assert selected_frame_ids(7, 1) == [0, 1, 2, 3, 4, 5, 6]


@check
def frame_selection_clamps_bad_stride():
    from vrdf_inference.video import selected_frame_ids

    # stride 0 would be a range() ValueError; it must clamp to 1 rather than crash.
    assert selected_frame_ids(3, 0) == [0, 1, 2]


@check
def transform_frame_returns_chw():
    from datasets.dataset import make_video_transforms
    from vrdf_inference.video import transform_frame

    transform = make_video_transforms("val", True, False, "small", overflow_boxes=False)
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    tensor, height, width = transform_frame(transform, frame)
    assert tensor.ndim == 3, tensor.shape
    assert tensor.shape[0] == 3, tensor.shape
    assert (height, width) == tuple(tensor.shape[1:]), (height, width, tensor.shape)


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
