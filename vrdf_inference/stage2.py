"""Stage-2 relation classification over the tracklets Stage 1 proposed.

Stage 2 initialises its decoder queries from ``roi_align`` on boxes it is given, so it cannot
enumerate pairs itself -- that is what ``tracklets.py`` is for. Here the tracklets are treated as
pseudo-ground-truth: one pseudo-target per frame, then one classification per tracklet at the end.
"""

import time

import numpy as np
import torch

import util.misc as utils
from util.box_ops import box_xyxy_to_cxcywh
from vrdf_inference.console import log, warn
from vrdf_inference.video import iter_decoded_frames, transform_frame


def build_pseudo_target(active, frame_index, device, num_verb_classes, resized_hw):
    """Assemble the one pseudo-GT dict ``VRDFormer_S2.forward`` reads for a single frame.

    Keys and conventions follow ``datasets/dataset.py:prepare_data_stage2`` and
    ``models/transformer.py``: ``unscaled_{sub,obj}_boxes`` are xyxy in **resized** pixels
    (``roi_align`` input at spatial scale 1/32), while ``{sub,obj}_boxes`` are normalized
    cxcywh (read by ``memory_update``).

    ``resized_hw`` is *this stage's* transform output size, not stage 1's. Deriving the pixel boxes
    from the normalized ones here, rather than passing stage 1's pixel boxes through, is what keeps
    the stages independent when ``--stage1_resolution`` and ``--stage2_resolution`` differ: the
    resolution changes ``max_size``, which clamps the short side too, so stage 1's 800x1066 boxes
    are not stage 2's 600x800 boxes.
    """
    height, width = resized_hw
    scale = torch.tensor(
        [width, height, width, height], dtype=torch.float32, device=device
    )

    sub_norm = torch.as_tensor(
        np.stack([t["frames"][frame_index]["box_n_sub"] for t in active]),
        dtype=torch.float32,
    )
    obj_norm = torch.as_tensor(
        np.stack([t["frames"][frame_index]["box_n_obj"] for t in active]),
        dtype=torch.float32,
    )

    target = {
        "unscaled_sub_boxes": (sub_norm * scale).to(device),
        "unscaled_obj_boxes": (obj_norm * scale).to(device),
        "sub_boxes": box_xyxy_to_cxcywh(sub_norm).to(device),
        "obj_boxes": box_xyxy_to_cxcywh(obj_norm).to(device),
        "so_track_ids": torch.as_tensor(
            [[t["sub_tid"], t["obj_tid"]] for t in active],
            dtype=torch.long,
            device=device,
        ),
        "num_inst": len(active),
        "sub_labels": torch.as_tensor(
            [t["sub_cls"] for t in active], dtype=torch.long, device=device
        ),
        "obj_labels": torch.as_tensor(
            [t["obj_cls"] for t in active], dtype=torch.long, device=device
        ),
        # Read by memory_update but only consumed by the criterion, which inference never runs.
        "verb_labels": torch.zeros(
            (len(active), num_verb_classes), dtype=torch.float32, device=device
        ),
        "frame_id": frame_index,
    }
    return target


@torch.no_grad()
def classify_tracklets(
    model, video_path, transform, tracklets, device, args, num_verb_classes
):
    """Accumulate per-tracklet ``rel_embed`` over the video, then classify each pair once.

    Mirrors ``engine.eval_stage2`` but with ``is_eval=False``: the non-eval branch of
    ``relation_classifier`` walks every memory key, mean-pools that pair's ``rel_embed`` over
    the frames it was seen in, and returns the sub/verb/obj logits -- exactly the
    per-tracklet predictions we want. ``eos`` stays False so the classification happens here,
    once, with an explicit key ordering to map back to tracklets.

    Returns ``(stage2_out, resized_hw)``; ``resized_hw`` is the stage-2 input size, reported in
    ``results.json`` so a stage-1/stage-2 resolution difference is visible rather than implicit.
    """
    by_frame = {}
    for tracklet in tracklets:
        for frame_index in tracklet["frame_indices"]:
            by_frame.setdefault(frame_index, []).append(tracklet)

    memory = None
    frames_run = 0
    resized_hw = None
    started = time.time()
    for index, _source_fid, frame_rgb in iter_decoded_frames(
        video_path, args.every_n_frames, args.max_frames, args.decode_chunk
    ):
        active = by_frame.get(index)
        if not active:
            continue
        if len(active) > model.num_queries:
            active = sorted(
                active, key=lambda t: t["frames"][index]["pair_score"], reverse=True
            )
            warn(
                f"frame {index}: {len(by_frame[index])} active tracklets exceeds "
                f"num_queries={model.num_queries}; keeping the strongest {model.num_queries}"
            )
            active = active[: model.num_queries]
        frame_tensor, height, width = transform_frame(transform, frame_rgb)
        resized_hw = (height, width)
        samples = utils.NestedTensor.from_tensor_list([frame_tensor]).to(device)
        target = build_pseudo_target(
            active, index, device, num_verb_classes, resized_hw
        )
        memory = model(samples, [target], memory, eos=False, is_eval=False)
        frames_run += 1
        if frames_run % 20 == 0:
            elapsed = time.time() - started
            log(
                f"stage 2: processed {frames_run} frames "
                f"[{frames_run / max(elapsed, 1e-6):.1f} fps, {elapsed:.1f}s elapsed]"
            )

    if memory is None:
        warn("stage 2: no frame had an active tracklet; nothing to classify")
        return None, resized_hw
    log(
        f"stage 2: accumulated memory over {frames_run} frames, "
        f"{len(memory[0]['rel_embed'])} tracklet keys"
    )

    sub_logits, verb_logits, obj_logits, *_ = model.relation_classifier(memory[0])
    return {
        "memory": memory[0],
        "sub_logits": sub_logits,
        "verb_logits": verb_logits,
        "obj_logits": obj_logits,
    }, resized_hw


def det_box(tracklet, frame_index, key):
    det = tracklet["frames"].get(frame_index)
    return [round(float(v), 2) for v in det[key]] if det is not None else None


@torch.no_grad()
def build_predictions(
    stage2_out, tracklets, obj_names, action_names, args, meta, model_s2
):
    """Turn the Stage-2 logits into the per-tracklet prediction records.

    ``relation_classifier``'s non-eval branch returns one row per memory key, and
    ``classify_tracklets`` inserts keys in ``tracklets`` order, so ``so_keys`` maps back to
    tracklets by id rather than by position.
    """
    memory = stage2_out["memory"]
    sub_logits = stage2_out["sub_logits"].float().cpu()
    verb_logits = stage2_out["verb_logits"].float().cpu()
    obj_logits = stage2_out["obj_logits"].float().cpu()

    so_keys = list(memory["rel_embed"].keys())
    by_so_id = {t["so_id"]: t for t in tracklets}
    frame_ids = meta["frame_ids"]
    verb_head = model_s2.verb_class_embed[-1]

    predictions = []
    for row, so_id in enumerate(so_keys):
        tracklet = by_so_id.get(so_id)
        if tracklet is None:
            # Should not happen: memory keys come from the pseudo-targets built from tracklets.
            warn(f"stage 2 produced key {so_id} with no matching tracklet; skipping")
            continue

        verb_id = int(verb_logits[row].argmax())
        score = float(verb_logits[row].sigmoid()[verb_id])
        if score < args.min_score:
            continue

        sub_id = int(sub_logits[row].argmax())
        obj_id = int(obj_logits[row].argmax())

        observed = sorted(tracklet["frame_indices"])
        first_frame, last_frame = observed[0], observed[-1]

        # ``sub_boxes``/``obj_boxes`` carry one entry per *observed* frame -- dense, in step with
        # ``frames``/``observed_frames``, and never None. Iterating a contiguous
        # ``range(first, last + 1)`` here would be wrong: ``--max_gap`` is what lets a tracklet
        # skip decoded frames, so a contiguous span reintroduces the skipped frames as None
        # entries and desynchronises this list from ``observed_frames``.
        sub_boxes = [det_box(tracklet, i, "box_px_sub") for i in observed]
        obj_boxes = [det_box(tracklet, i, "box_px_obj") for i in observed]
        assert None not in sub_boxes and None not in obj_boxes, (
            f"tracklet {tracklet['so_id']} has a gap in observed frames {observed}"
        )

        # ``sub_traj``/``obj_traj`` are the other axis: every source frame the tracklet spans,
        # which is what ``util.evaluate.viou`` compares under ``source_duration``. Missed source
        # frames carry the last observed box forward. ``observed[0]`` is the earliest frame, so
        # the first iteration always assigns and the carry is never a leading None.
        source_first, source_last = frame_ids[first_frame], frame_ids[last_frame]
        observed_sources = [frame_ids[i] for i in observed]
        sub_traj, obj_traj = [], []
        last_sub = last_obj = None
        pos = 0
        for source_fid in range(source_first, source_last + 1):
            if pos < len(observed_sources) and observed_sources[pos] == source_fid:
                last_sub, last_obj = sub_boxes[pos], obj_boxes[pos]
                pos += 1
            sub_traj.append(last_sub)
            obj_traj.append(last_obj)
        assert len(sub_traj) == source_last + 1 - source_first

        # memory["rel_embed"][so_id] holds one row per frame the pair was active in, in
        # ascending frame order, so the head can be applied per frame for a temporal view.
        per_frame_rows = memory["rel_embed"][so_id].float()
        per_frame_verbs = [
            action_names[int(i)] for i in verb_head(per_frame_rows).argmax(-1)
        ]

        predictions.append(
            {
                "track_id": so_id,
                "sub_track_id": tracklet["sub_tid"],
                "obj_track_id": tracklet["obj_tid"],
                "triplet": [
                    obj_names[sub_id],
                    action_names[verb_id],
                    obj_names[obj_id],
                ],
                "subject": obj_names[sub_id],
                "verb": action_names[verb_id],
                "object": obj_names[obj_id],
                "verb_id": verb_id,
                "sub_cls_id": sub_id,
                "obj_cls_id": obj_id,
                "score": score,
                "verb_logit": float(verb_logits[row][verb_id]),
                # What the Stage-1 detector called the pair, for comparison with Stage 2's guess.
                "detector_classes": {
                    "subject": obj_names[tracklet["sub_cls"]],
                    "object": obj_names[tracklet["obj_cls"]],
                    "subject_id": tracklet["sub_cls"],
                    "object_id": tracklet["obj_cls"],
                },
                "frames": observed,
                "source_frames": [frame_ids[i] for i in observed],
                "frame_range": [first_frame, last_frame],
                "source_frame_range": [source_first, source_last],
                # Half-open [start, end) over the decoded axis and over the source axis, the
                # convention util/evaluate.py expects.
                "duration": [first_frame, last_frame + 1],
                "source_duration": [source_first, source_last + 1],
                "observed_frames": observed,
                # Decoded-axis boxes, one entry per ``frames`` entry (None on missed frames).
                "sub_boxes": sub_boxes,
                "obj_boxes": obj_boxes,
                # Source-axis trajectories: len(sub_traj) == source_duration[1] - source_duration[0],
                # so ``util.evaluate.viou(sub_traj, source_duration, gt_traj, gt_duration)`` is
                # well-formed against dataset ground truth.
                "sub_traj": sub_traj,
                "obj_traj": obj_traj,
                "per_frame_verbs": per_frame_verbs,
                "frames_in_memory": int(per_frame_rows.shape[0]),
            }
        )

    predictions.sort(key=lambda p: p["score"], reverse=True)
    log(
        f"triplets: {len(predictions)} kept at --min_score {args.min_score} "
        f"(of {len(so_keys)} tracklets)"
    )
    return predictions
