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


@check
def pair_iou_matches_box_iou_and_guards_degenerate():
    import torch

    from vrdf_inference.stage1 import pair_iou

    a = torch.tensor([0.0, 0.0, 10.0, 10.0])
    b = torch.tensor([5.0, 5.0, 15.0, 15.0])
    assert abs(pair_iou(a, b) - 1.0 / 7.0) < 1e-6, pair_iou(a, b)
    assert pair_iou(a, a) == 1.0, pair_iou(a, a)
    # Two degenerate boxes: util's box_iou gives NaN here, the wrapper must give 0.0.
    z = torch.tensor([5.0, 5.0, 5.0, 5.0])
    assert pair_iou(z, z) == 0.0, pair_iou(z, z)
    # Detections carry numpy boxes, so the wrapper must accept those too.
    assert abs(pair_iou(a.numpy(), b.numpy()) - 1.0 / 7.0) < 1e-6
    assert pair_iou(a.numpy(), a.numpy()) == 1.0


@check
def decode_frame_thresholds_and_scales():
    import argparse

    import torch

    from vrdf_inference.stage1 import decode_frame

    num_queries, num_obj, num_verb = 4, 80, 50
    out = {
        # Query 0 clearly above threshold, the rest at sigmoid(0) = 0.5.
        "pred_sub_logits": [torch.zeros(num_queries, num_obj)],
        "pred_obj_logits": [torch.zeros(num_queries, num_obj)],
        "pred_verb_logits": [torch.zeros(num_queries, num_verb)],
        # cxcywh, normalized: a 0.2-wide box centred at 0.5.
        "pred_sub_boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * num_queries)],
        "pred_obj_boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * num_queries)],
    }
    out["pred_sub_logits"][0][0, 5] = 4.0  # sigmoid ~0.98
    out["pred_obj_logits"][0][0, 7] = 4.0

    args = argparse.Namespace(
        conf_sub=0.9, conf_obj=0.9, conf_verb=0.5, dedup_iou=0.9, max_dets_per_frame=50
    )
    detections, ceilings = decode_frame(out, (200, 300), (100, 150), args)

    assert len(detections) == 1, (
        f"expected only the confident query, got {len(detections)}"
    )
    det = detections[0]
    assert det["sub_cls"] == 5, det["sub_cls"]
    assert det["obj_cls"] == 7, det["obj_cls"]
    # 0.2 normalized width on a 300px-wide resized frame is 60px, halved again on the source.
    width_resized = det["box_resized_sub"][2] - det["box_resized_sub"][0]
    width_px = det["box_px_sub"][2] - det["box_px_sub"][0]
    assert abs(width_resized - 60.0) < 1e-3, det["box_resized_sub"]
    assert abs(width_px - 30.0) < 1e-3, det["box_px_sub"]
    assert ceilings["sub"] > 0.9, ceilings


@check
def decode_frame_dedups_same_class_pairs():
    import argparse

    import torch

    from vrdf_inference.stage1 import decode_frame

    num_queries, num_obj, num_verb = 2, 80, 50
    out = {
        "pred_sub_logits": [torch.zeros(num_queries, num_obj)],
        "pred_obj_logits": [torch.zeros(num_queries, num_obj)],
        "pred_verb_logits": [torch.zeros(num_queries, num_verb)],
        "pred_sub_boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * num_queries)],
        "pred_obj_boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * num_queries)],
    }
    # Both queries fire the same classes on the same boxes -> one survives NMS.
    for q in range(num_queries):
        out["pred_sub_logits"][0][q, 5] = 4.0
        out["pred_obj_logits"][0][q, 7] = 4.0

    args = argparse.Namespace(
        conf_sub=0.5, conf_obj=0.5, conf_verb=0.5, dedup_iou=0.9, max_dets_per_frame=50
    )
    detections, _ = decode_frame(out, (200, 300), (100, 150), args)
    assert len(detections) == 1, f"duplicates were not merged: {len(detections)}"


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
