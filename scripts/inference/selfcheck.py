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


@check
def pseudo_target_uses_the_given_size():
    """The stage-2 pixel boxes must follow stage 2's own input size, not stage 1's."""
    import torch

    from vrdf_inference.stage2 import build_pseudo_target

    def tracklet(sub_box_n):
        frame = {
            "box_n_sub": sub_box_n,
            "box_n_obj": [0.4, 0.4, 0.8, 0.8],
            "box_resized_sub": [0.0, 0.0, 1.0, 1.0],
            "box_resized_obj": [0.0, 0.0, 1.0, 1.0],
            "pair_score": 0.2,
        }
        return {
            "sub_tid": 0,
            "obj_tid": 0,
            "sub_cls": 5,
            "obj_cls": 7,
            "frames": {0: frame},
        }

    active = [tracklet([0.25, 0.25, 0.75, 0.75])]
    cpu = torch.device("cpu")

    # A half-width box: 0.5 * 600 = 300 wide, 0.5 * 800 = 400 tall.
    target = build_pseudo_target(active, 0, cpu, 50, (800, 600))
    box = target["unscaled_sub_boxes"][0]
    assert abs(float(box[2] - box[0]) - 300.0) < 1e-3, box
    assert abs(float(box[3] - box[1]) - 400.0) < 1e-3, box

    # The same normalized box at a different stage-2 size must scale differently. This is the bug
    # the task fixes, so a size-insensitive implementation fails here.
    other = build_pseudo_target(active, 0, cpu, 50, (600, 800))
    other_box = other["unscaled_sub_boxes"][0]
    assert abs(float(other_box[2] - other_box[0]) - 400.0) < 1e-3, other_box

    # memory_update reads shape[0] of sub_boxes, so it must exist and match the row count.
    assert target["sub_boxes"].shape[0] == len(active), target["sub_boxes"].shape
    assert target["num_inst"] == len(active), target["num_inst"]


@check
def label_style_scales_with_frame_height():
    from vrdf_inference.render import label_style

    # 240/400 = 0.6, inside the 0.45..1.2 band; 2160 clamps at the top.
    assert label_style(240) == (0.6, 1), label_style(240)
    assert label_style(2160)[0] == 1.2, label_style(2160)
    assert label_style(2160)[1] >= 1, label_style(2160)


@check
def renderers_always_write_a_playable_video():
    """Both renderers, including the zero-detection path, must produce a real video."""
    import argparse
    import tempfile
    from pathlib import Path

    import cv2

    from vrdf_inference.render import render_detections, render_triplets

    def frame_count(path):
        reader = cv2.VideoCapture(str(path))
        assert reader.isOpened(), f"{path} is not readable"
        count = 0
        while True:
            ok, _ = reader.read()
            if not ok:
                break
            count += 1
        reader.release()
        return count

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "in.mp4"
        writer = cv2.VideoWriter(
            str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
        )
        for i in range(5):
            writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
        writer.release()

        meta = {
            "video_id": "synthetic",
            "fps": 10.0,
            "source_width": 64,
            "source_height": 48,
            "num_decoded_frames": 5,
        }
        args = argparse.Namespace(
            every_n_frames=1,
            max_frames=None,
            decode_chunk=8,
            max_labels_per_frame=2,
            obj_names=["a"] * 80,
        )
        det = {
            "sub_cls": 5,
            "obj_cls": 7,
            "sub_score": 0.2,
            "obj_score": 0.15,
            "pair_score": 0.15,
            "box_px_sub": [4.0, 4.0, 30.0, 30.0],
            "box_px_obj": [20.0, 10.0, 50.0, 40.0],
        }
        prediction = {
            "track_id": "0-0",
            "subject": "adult",
            "verb": "next_to",
            "object": "table",
            "score": 0.3,
            "frames": [0, 1, 2, 3, 4],
            "observed_frames": [0, 1, 2, 3, 4],
            "sub_boxes": [[4.0, 4.0, 30.0, 30.0]] * 5,
            "obj_boxes": [[20.0, 10.0, 50.0, 40.0]] * 5,
        }

        cases = {
            "detections": lambda p: render_detections(
                str(src), p, [[det]] * 5, meta, args
            ),
            "detections-empty": lambda p: render_detections(
                str(src), p, [[]] * 5, meta, args
            ),
            "triplets": lambda p: render_triplets(
                str(src), p, [prediction], meta, args
            ),
            "triplets-empty": lambda p: render_triplets(str(src), p, [], meta, args),
        }
        for name, run in cases.items():
            out = tmp / f"{name}.mp4"
            run(out)
            assert out.exists() and out.stat().st_size > 0, f"{name} wrote no video"
            assert frame_count(out) == 5, f"{name}: wrong frame count"


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
