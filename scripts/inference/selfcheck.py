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


@check
def weighted_vote_prefers_confidence():
    from vrdf_inference.tracklets import weighted_vote

    # 1 appears twice but with tiny weight; 2 appears once with a large weight.
    assert weighted_vote([1, 1, 2], [0.01, 0.01, 5.0]) == 2
    assert weighted_vote([1, 1, 2], [5.0, 5.0, 0.01]) == 1
    # Ties break deterministically on the value, not on insertion order.
    assert weighted_vote([3, 1], [1.0, 1.0]) == 1


def _synthetic_detection(frame_shift, cls=5):
    """A detection whose subject box slides right with ``frame_shift``; object box fixed."""
    return {
        "sub_cls": cls,
        "obj_cls": 7,
        "pair_score": 0.2,
        "box_n_sub": [0.1 + frame_shift, 0.1, 0.3 + frame_shift, 0.3],
        "box_n_obj": [0.5, 0.5, 0.7, 0.7],
        "box_px_sub": [0.0, 0.0, 1.0, 1.0],
        "box_px_obj": [0.0, 0.0, 1.0, 1.0],
        "box_resized_sub": [0.0, 0.0, 1.0, 1.0],
        "box_resized_obj": [0.0, 0.0, 1.0, 1.0],
        "query_idx": 0,
        "sub_score": 0.2,
        "obj_score": 0.2,
        "verb_cls": 0,
        "verb_score": 0.2,
    }


def _assoc_args(**overrides):
    import argparse

    base = {
        "iou_thresh": 0.3,
        "max_gap": 2,
        "min_tracklet_frames": 1,
        "cls_gate": "motion",
        "cls_vote": "keep",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@check
def tracklets_join_boxes_that_overlap():
    from vrdf_inference.tracklets import form_tracklets

    # Frame 0 and frame 1 overlap heavily -> one tracklet spanning both frames.
    tracklets = form_tracklets(
        [[_synthetic_detection(0.0)], [_synthetic_detection(0.02)]], _assoc_args()
    )
    assert len(tracklets) == 1, f"expected one joined tracklet, got {len(tracklets)}"
    assert tracklets[0]["frame_indices"] == [0, 1], tracklets[0]["frame_indices"]


@check
def tracklets_split_on_no_overlap():
    from vrdf_inference.tracklets import form_tracklets

    # Frame 1's subject box moved far away -> no overlap -> two tracklets.
    tracklets = form_tracklets(
        [[_synthetic_detection(0.0)], [_synthetic_detection(0.6)]], _assoc_args()
    )
    assert len(tracklets) == 2, f"expected two tracklets, got {len(tracklets)}"


@check
def tracklets_bridge_a_missed_frame_within_max_gap():
    from vrdf_inference.tracklets import form_tracklets

    # A gap at frame 1, then an overlapping box at frame 2, is inside max_gap=2.
    tracklets = form_tracklets(
        [[_synthetic_detection(0.0)], [], [_synthetic_detection(0.02)]], _assoc_args()
    )
    assert len(tracklets) == 1, f"the gap was not bridged: {len(tracklets)} tracklets"
    assert tracklets[0]["frame_indices"] == [0, 2], tracklets[0]["frame_indices"]


@check
def tracklets_respect_min_frames():
    from vrdf_inference.tracklets import form_tracklets

    # The single-frame tracklet is dropped once min_tracklet_frames rises above 1.
    per_frame = [[_synthetic_detection(0.0)], [_synthetic_detection(0.02)]]
    assert len(form_tracklets(per_frame, _assoc_args())) == 1
    assert len(form_tracklets(per_frame, _assoc_args(min_tracklet_frames=3))) == 0


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
