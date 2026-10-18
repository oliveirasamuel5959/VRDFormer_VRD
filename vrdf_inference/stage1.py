"""Stage-1 detection: decode one frame's output, or drive the whole video through the detector.

Two entry points, because two callers need different halves:

* :func:`detect_frame` runs one decoded frame and returns that frame's detections. This is what
  ``--mode stage1`` uses.
* :func:`run_detection` drives the whole decode loop and calls an optional ``on_frame`` hook, which
  lets ``--mode stage2`` consume detections as they are produced instead of decoding twice.
"""

import time

import numpy as np
import torch

import util.misc as utils
from models.vrdformer import VRDFormer
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from vrdf_inference.console import log, warn
from vrdf_inference.video import iter_decoded_frames, transform_frame


def pair_iou(box_a, box_b):
    """IoU between two single xyxy boxes, as a float.

    Thin wrapper over util's pairwise ``box_iou`` (``util/box_ops.py:55``), which takes ``[N,4]``
    tensors and returns an ``(N,M)`` matrix. Detections carry their boxes as numpy arrays, so this
    accepts either. That function divides by the union without a guard, so two degenerate
    (zero-area) boxes give NaN where the scalar helper this replaced gave 0.0; ``nan_to_num``
    restores the old behaviour.
    """
    a = torch.as_tensor(box_a, dtype=torch.float32).unsqueeze(0)
    b = torch.as_tensor(box_b, dtype=torch.float32).unsqueeze(0)
    return float(torch.nan_to_num(box_iou(a, b)[0][0, 0]))


def decode_frame(out, resized_hw, source_hw, args):
    """Filter one frame's Stage-1 output into detections in resized *and* source pixels.

    Each of the 200 static queries predicts a full (subject, object, verb) tuple. Classes are
    decoded with sigmoid (no softmax, no no-object slot), the query is kept when both the
    subject and the object class clear their threshold, then queries of the same
    (sub_cls, obj_cls) group are NMS'd on ``min(iou_sub, iou_obj)`` so one object pair is not
    reported several times.
    """

    def to_np(t):
        return t.detach().float().cpu().numpy()

    sub_prob = to_np(out["pred_sub_logits"][0].sigmoid())  # (Q, num_obj)
    obj_prob = to_np(out["pred_obj_logits"][0].sigmoid())
    verb_prob = to_np(out["pred_verb_logits"][0].sigmoid())  # (Q, num_verb)
    sub_boxes = to_np(
        box_cxcywh_to_xyxy(out["pred_sub_boxes"][0])
    )  # (Q, 4) normalized xyxy
    obj_boxes = to_np(box_cxcywh_to_xyxy(out["pred_obj_boxes"][0]))

    sub_cls = sub_prob.argmax(-1)
    obj_cls = obj_prob.argmax(-1)
    sub_score = sub_prob.max(-1)
    obj_score = obj_prob.max(-1)
    verb_cls = verb_prob.argmax(-1)
    verb_score = verb_prob.max(-1)

    resized_h, resized_w = resized_hw
    source_h, source_w = source_hw
    scale = np.array([resized_w, resized_h, resized_w, resized_h], dtype=np.float32)
    to_source = np.array(
        [
            source_w / resized_w,
            source_h / resized_h,
            source_w / resized_w,
            source_h / resized_h,
        ],
        dtype=np.float32,
    )

    # Report the pre-filter ceiling as well: when a threshold is too high the kept set is empty
    # and the only useful diagnostic is how high the scores got before filtering.
    ceilings = {
        "sub": float(sub_score.max()),
        "obj": float(obj_score.max()),
        "pair": float(np.minimum(sub_score, obj_score).max()),
    }

    keep = np.flatnonzero((sub_score >= args.conf_sub) & (obj_score >= args.conf_obj))
    dets = []
    for q in keep:
        sub_px = sub_boxes[q] * scale
        obj_px = obj_boxes[q] * scale
        dets.append(
            {
                "query_idx": int(q),
                "sub_cls": int(sub_cls[q]),
                "obj_cls": int(obj_cls[q]),
                "sub_score": float(sub_score[q]),
                "obj_score": float(obj_score[q]),
                "verb_cls": int(verb_cls[q]),
                "verb_score": float(verb_score[q]),
                "box_n_sub": sub_boxes[q].copy(),
                "box_n_obj": obj_boxes[q].copy(),
                "box_px_sub": (sub_px * to_source).copy(),
                "box_px_obj": (obj_px * to_source).copy(),
                "box_resized_sub": sub_px.copy(),
                "box_resized_obj": obj_px.copy(),
            }
        )

    # Score for ordering/NMS: the weaker of the two boxes decides how much to trust the pair.
    for det in dets:
        det["pair_score"] = min(det["sub_score"], det["obj_score"])
    dets.sort(key=lambda d: d["pair_score"], reverse=True)
    dets = dets[: args.max_dets_per_frame]

    # Suppression is on ``min(iou_sub, iou_obj)`` within a (sub_cls, obj_cls) group. torchvision's
    # nms/batched_nms compare a single box list on plain IoU, so they cannot express this.
    kept = []
    for det in dets:
        duplicate = False
        for prev in kept:
            if (det["sub_cls"], det["obj_cls"]) != (prev["sub_cls"], prev["obj_cls"]):
                continue
            overlap = min(
                pair_iou(det["box_n_sub"], prev["box_n_sub"]),
                pair_iou(det["box_n_obj"], prev["box_n_obj"]),
            )
            if overlap >= args.dedup_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept, ceilings


def report_stage1_scores(per_frame):
    """Describe the kept detections' confidence so thresholds can be picked from data."""
    scores = [d["pair_score"] for dets in per_frame for d in dets]
    if not scores:
        return
    arr = np.asarray(scores)
    log(
        f"stage 1 kept-pair scores: min {arr.min():.3f} median {np.median(arr):.3f} "
        f"max {arr.max():.3f} ({len(arr)} pairs)"
    )


@torch.no_grad()
def detect_frame(model, frame_rgb, prev_features, transform, device, args, source_hw):
    """Run the detector on one decoded frame.

    Returns ``(detections, ceilings, features, (height, width))`` where the size is the
    *transformed* frame's, i.e. the resized input the model actually saw. ``features`` is this
    frame's backbone output, to be handed back as ``prev_features`` on the next call: Stage 1 was
    trained with ``multi_frame_attention``, so consecutive frames must be fed in order.
    """
    frame_tensor, height, width = transform_frame(transform, frame_rgb)
    samples = utils.NestedTensor.from_tensor_list([frame_tensor]).to(device)
    out, _, features, _, _ = VRDFormer.forward(
        model, samples, targets=None, prev_features=prev_features
    )
    detections, ceiling = decode_frame(out, (height, width), source_hw, args)
    return detections, ceiling, features, (height, width)


def report_stage1_summary(per_frame, ceilings, args, started):
    """Log the detection totals and the pre-threshold score ceiling.

    The ceiling is the number that tells you how to set ``--conf_sub``/``--conf_obj`` on a new
    video: when nothing clears the thresholds it is the only useful diagnostic, so it is printed
    either way.
    """
    total = sum(len(d) for d in per_frame)
    best_sub = max((c["sub"] for c in ceilings), default=0.0)
    best_obj = max((c["obj"] for c in ceilings), default=0.0)
    best_pair = max((c["pair"] for c in ceilings), default=0.0)
    elapsed = time.time() - started
    log(
        f"stage 1: {total} detections over {len(per_frame)} frames at "
        f"--conf_sub {args.conf_sub} / --conf_obj {args.conf_obj} "
        f"in {elapsed:.1f}s ({len(per_frame) / max(elapsed, 1e-6):.2f} fps)"
    )
    log(
        f"stage 1: best score before thresholding -- subject {best_sub:.3f}, "
        f"object {best_obj:.3f}, pair {best_pair:.3f}"
    )
    if total == 0:
        warn(
            f"no detection cleared the thresholds; the highest pair score reached was "
            f"{best_pair:.3f} (subject {best_sub:.3f}, object {best_obj:.3f}). Lower "
            f"--conf_sub/--conf_obj below those values, e.g. "
            f"--conf_sub {max(best_sub * 0.5, 0.05):.2f} --conf_obj {max(best_obj * 0.5, 0.05):.2f}."
        )


@torch.no_grad()
def run_detection(model, video_path, transform, device, args, on_frame=None):
    """Detect on every selected frame, carrying the previous frame's backbone features.

    Returns ``(per_frame, resized_hw, source_hw)``. ``per_frame`` is a list of detection lists, one
    entry per decoded frame, in decode order.

    ``on_frame(index, source_fid, detections)`` is called after each frame if given, which lets a
    caller consume detections as they are produced without a second decode pass.
    """
    per_frame = []
    ceilings = []
    prev_features = None
    resized_hw = None
    source_hw = None
    started = time.time()
    for index, source_fid, frame_rgb in iter_decoded_frames(
        video_path, args.every_n_frames, args.max_frames, args.decode_chunk
    ):
        source_hw = frame_rgb.shape[:2]
        detections, ceiling, prev_features, resized_hw = detect_frame(
            model, frame_rgb, prev_features, transform, device, args, source_hw
        )
        per_frame.append(detections)
        ceilings.append(ceiling)
        if on_frame is not None:
            on_frame(index, source_fid, detections)
        if index % 20 == 0:
            elapsed = time.time() - started
            rate = (index + 1) / elapsed if elapsed > 0 else 0.0
            log(
                f"stage 1: frame {index} ({source_fid}) -> {len(detections)} detections "
                f"[{rate:.1f} fps, {elapsed:.1f}s elapsed]"
            )
    report_stage1_summary(per_frame, ceilings, args, started)
    return per_frame, resized_hw, source_hw
