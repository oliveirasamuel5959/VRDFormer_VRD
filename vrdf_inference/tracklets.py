"""Association of per-frame detections into tracklets.

Used only by the full pipeline: this is the module ``--mode stage1`` never imports, because a
detection-only run has no relations to track.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_iou
from vrdf_inference.console import log, warn


def weighted_vote(values, weights):
    """Most common value, ties broken by summed confidence (higher wins)."""
    totals = {}
    for value, weight in zip(values, weights):
        totals[value] = totals.get(value, 0.0) + weight
    return max(sorted(totals), key=lambda v: totals[v])


def form_tracklets(per_frame_dets, args):
    """Associate per-frame detections into tracklets (IoU, scipy Hungarian).

    Subject and object boxes are matched *together*: the cost is ``min(iou_sub, iou_obj)`` and
    a candidate is accepted only when that cost clears ``--iou_thresh``. A subject-object pair
    therefore joins an existing tracklet only if both sides track, which is what keeps
    ``track_id`` meaning one relation over time.

    Class labels are **not** part of the cost by default (``--cls_gate motion``): the detector's
    argmax class flips between frames at the confidence levels this checkpoint produces, and
    gating on it shatters relations into one tracklet per label wobble. Box motion is the more
    reliable signal, and the classes are recovered afterwards as a confidence-weighted vote
    over the tracklet (``--cls_vote keep``). ``--cls_gate same`` restores strict class equality
    --- the variant RUNPOD_INFERENCE.md describes.

    Unmatched detections open new tracklets. A tracklet that misses a detection stays
    ``--max_gap`` frames in the pending pool (its last known boxes are kept for matching) so a
    single dropped frame does not split one relation into two.
    """
    tracklets = []
    pending = []
    next_sub_tid = 0
    next_obj_tid = 0

    for frame_index, dets in enumerate(per_frame_dets):
        pending = [p for p in pending if frame_index - p["last_frame"] <= args.max_gap]

        matched_pairs = []
        if pending and dets:
            # Two pairwise IoU matrices replace the former per-pair Python loop. The cost is the
            # min of the subject and object overlap, so both sides must track for a join.
            pending_sub = torch.as_tensor(
                np.stack([p["box_n_sub"] for p in pending]), dtype=torch.float32
            )
            pending_obj = torch.as_tensor(
                np.stack([p["box_n_obj"] for p in pending]), dtype=torch.float32
            )
            det_sub = torch.as_tensor(
                np.stack([d["box_n_sub"] for d in dets]), dtype=torch.float32
            )
            det_obj = torch.as_tensor(
                np.stack([d["box_n_obj"] for d in dets]), dtype=torch.float32
            )
            iou_sub = box_iou(pending_sub, det_sub)[0]
            iou_obj = box_iou(pending_obj, det_obj)[0]
            # nan_to_num for the same reason as pair_iou: two degenerate boxes give NaN, and a
            # NaN in the cost matrix would make linear_sum_assignment return nonsense.
            cost = torch.nan_to_num(torch.minimum(iou_sub, iou_obj)).numpy()
            cost = cost.astype(np.float64)
            allowed = cost >= args.iou_thresh
            rows, cols = linear_sum_assignment(-cost)
            matched_pairs = [(r, c) for r, c in zip(rows, cols) if allowed[r, c]]
            # The class labels are deliberately NOT part of the cost. The detector's argmax
            # class flips between frames at these confidence levels, so gating on it splits a
            # relation every time the label wobbles. Box motion is the more reliable signal;
            # classes come back as a per-tracklet vote over its frames ("keep" mode) or from
            # the first frame ("first" mode).
            if args.cls_gate == "same":
                matched_pairs = [
                    (r, c)
                    for r, c in matched_pairs
                    if pending[r]["sub_cls"] == dets[c]["sub_cls"]
                    and pending[r]["obj_cls"] == dets[c]["obj_cls"]
                ]

        used_dets = set()
        for row, col in matched_pairs:
            entry = pending[row]
            det = dets[col]
            tracklets[entry["tracklet"]]["frames"][frame_index] = det
            entry["last_frame"] = frame_index
            entry["box_n_sub"] = det["box_n_sub"]
            entry["box_n_obj"] = det["box_n_obj"]
            used_dets.add(col)

        for col, det in enumerate(dets):
            if col in used_dets:
                continue
            if len(tracklets) >= 4096:
                warn("tracklet limit (4096) reached; dropping further new tracklets")
                break
            tracklet = {
                "sub_tid": next_sub_tid,
                "obj_tid": next_obj_tid,
                "sub_cls": det["sub_cls"],
                "obj_cls": det["obj_cls"],
                "frames": {frame_index: det},
            }
            next_sub_tid += 1
            next_obj_tid += 1
            tracklets.append(tracklet)
            pending.append(
                {
                    "tracklet": len(tracklets) - 1,
                    "last_frame": frame_index,
                    "box_n_sub": det["box_n_sub"],
                    "box_n_obj": det["box_n_obj"],
                    "sub_cls": det["sub_cls"],
                    "obj_cls": det["obj_cls"],
                }
            )

    for tracklet in tracklets:
        tracklet["so_id"] = f"{tracklet['sub_tid']}-{tracklet['obj_tid']}"
        tracklet["frame_indices"] = sorted(tracklet["frames"].keys())
        # Majority class over the tracklet's observed frames (the association ignores class, so
        # the opening frame's label would otherwise stand for the whole tracklet). Used by
        # Stage 2 only as auxiliary context -- the class actually reported comes from Stage 2's
        # own heads, not from here.
        if args.cls_vote == "keep" and len(tracklet["frame_indices"]) > 1:
            votes_sub = [
                tracklet["frames"][i]["sub_cls"] for i in tracklet["frame_indices"]
            ]
            votes_obj = [
                tracklet["frames"][i]["obj_cls"] for i in tracklet["frame_indices"]
            ]
            weights = [
                tracklet["frames"][i]["pair_score"] for i in tracklet["frame_indices"]
            ]
            tracklet["sub_cls"] = weighted_vote(votes_sub, weights)
            tracklet["obj_cls"] = weighted_vote(votes_obj, weights)

    kept = [t for t in tracklets if len(t["frame_indices"]) >= args.min_tracklet_frames]
    log(
        f"tracklets: {len(kept)} kept of {len(tracklets)} built "
        f"(iou_thresh {args.iou_thresh}, max_gap {args.max_gap}, "
        f"min_tracklet_frames {args.min_tracklet_frames}, cls_gate {args.cls_gate})"
    )
    if tracklets:
        mean_len = sum(len(t["frame_indices"]) for t in kept) / max(len(kept), 1)
        log(f"tracklets: mean length {mean_len:.2f} decoded frames")
    return kept
