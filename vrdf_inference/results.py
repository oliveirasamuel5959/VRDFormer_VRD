"""Result payloads written next to the annotated video.

Two payload shapes, one per mode: a detection-only run has no tracklets and no triplets, so it does
not carry those keys at all rather than carrying them empty.
"""

from vrdf_inference.console import log


def save_results(path, payload):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


def summarise_detections(per_frame, obj_names, action_names, source_hw, args):
    """Compact per-frame Stage-1 annotation for results.json."""
    source_h, source_w = source_hw
    frames = []
    for index, dets in enumerate(per_frame):
        records = []
        for det in dets:
            verb = None
            if det["verb_score"] >= args.conf_verb:
                verb = {
                    "name": action_names[det["verb_cls"]],
                    "id": det["verb_cls"],
                    "score": round(det["verb_score"], 6),
                }
            records.append(
                {
                    "query_idx": det["query_idx"],
                    "subject": {
                        "name": obj_names[det["sub_cls"]],
                        "id": det["sub_cls"],
                        "score": round(det["sub_score"], 6),
                    },
                    "object": {
                        "name": obj_names[det["obj_cls"]],
                        "id": det["obj_cls"],
                        "score": round(det["obj_score"], 6),
                    },
                    "verb": verb,
                    "sub_box": [round(float(v), 2) for v in det["box_px_sub"]],
                    "obj_box": [round(float(v), 2) for v in det["box_px_obj"]],
                }
            )
        frames.append({"frame_index": index, "detections": records})
    return {"image_size": [source_w, source_h], "frames": frames}


def base_payload(video_path, video_id, mode, meta, resized_hw, args, stage_args):
    """The metadata every run records, mode-independent.

    ``num_source_frames``, ``stride`` and ``source_size`` describe the *input*, so a CPU run and a
    GPU run of the same clip must agree on them exactly. That is what makes them the right fields
    to compare across devices.
    """
    return {
        "video": video_path,
        "video_id": video_id,
        "mode": mode,
        "dataset": stage_args.dataset,
        "num_source_frames": meta["num_source_frames"],
        "num_frames": meta["num_decoded_frames"],
        "frame_ids": meta["frame_ids"],
        "stride": args.every_n_frames,
        "fps": round(meta["fps"], 4),
        "output_fps": round(max(1.0, meta["fps"] / max(1, args.every_n_frames)), 4),
        "source_size": [meta["source_width"], meta["source_height"]],
        "resized_size": [resized_hw[1], resized_hw[0]],
        "stage1": {
            "checkpoint": args.stage1_ckpt,
            "config": args.stage1_config,
            "resolution": stage_args.resolution,
        },
        "thresholds": {
            "conf_sub": args.conf_sub,
            "conf_obj": args.conf_obj,
            "conf_verb": args.conf_verb,
            "dedup_iou": args.dedup_iou,
        },
        "num_detections": int(sum(len(d) for d in meta["per_frame"])),
        "detections": summarise_detections(
            meta["per_frame"],
            args.obj_names,
            args.action_names,
            meta["source_hw"],
            args,
        ),
    }


def detection_payload(video_path, video_id, meta, resized_hw, args, stage_args):
    """Payload for ``--mode stage1``: metadata plus per-frame detections, nothing else."""
    return base_payload(
        video_path, video_id, "stage1", meta, resized_hw, args, stage_args
    )


def triplet_payload(
    video_path,
    video_id,
    meta,
    resized_hw,
    args,
    stage_args,
    stage2_args,
    stage2_hw,
    tracklets,
    predictions,
):
    """Payload for ``--mode stage2``: the detection payload plus tracklets and triplets."""
    payload = base_payload(
        video_path, video_id, "stage2", meta, resized_hw, args, stage_args
    )
    payload["stage2"] = {
        "checkpoint": args.stage2_ckpt,
        "config": args.stage2_config,
        "resolution": stage2_args.resolution,
        # Stage 2 resizes to its own size, and the resolution changes it. Recording both sizes
        # makes a stage-1/stage-2 resolution difference visible instead of implicit.
        "input_size": [stage2_hw[1], stage2_hw[0]] if stage2_hw else None,
    }
    payload["thresholds"].update(
        {
            "iou_thresh": args.iou_thresh,
            "max_gap": args.max_gap,
            "min_score": args.min_score,
        }
    )
    payload["num_tracklets"] = len(tracklets)
    payload["num_predictions"] = len(predictions)
    payload["predictions"] = predictions
    return payload


def write_eval_prediction(path, predictions, video_id):
    """Write ``util/evaluate``-shaped predictions next to results.json.

    ``engine.eval_stage2`` grades *ranking* over a whole val split, not one video, so this
    cannot produce that number by itself. What it does produce is a drop-in ``prediction``
    dict for the split, which is the shape-loopable part of the evaluation:

        from util.evaluate import evaluate_visual_relation
        groundtruth = {vid: dataset.get_relation_insts(vid) for vid in wanted}
        prediction = {}
        for f in glob.glob("output/inference/*/*/*_eval_prediction.json"):
            prediction.update(json.load(open(f)))
        evaluate_visual_relation(groundtruth, prediction)

    Only the official detection/tagging fields are emitted (``triplet``, ``score``,
    ``sub_traj``/``obj_traj`` in source pixels, ``duration``/``sub_duration``/``obj_duration``
    in source frame ids). ``min_pred_num`` padding and the zero-shot splits are the caller's
    business. The detection branch of ``evaluate_visual_relation`` only runs when
    ``sub_traj`` is present, which it now is.

    Coordinate caveat: these trajectories are source-pixel xyxy, whereas
    ``datasets/dataset.get_relation_insts`` returns ``[0,1]``-normalized boxes
    (``datasets/dataset.py:290-296`` with ``normalize_coords = True``). ``util.compute.viou``
    compares them without rescaling, so a sidecar scored against dataset ground truth needs one
    side rescaled by the source width/height first. That rescale is only valid when the input
    video is the original full-rate source, which is why nothing here guesses it.

    Frame ids carry the same assumption: ``viou`` indexes ``traj[k]`` as frame
    ``duration[0] + k``, and annotation ``duration`` values count original-video frames, while
    these ``source_duration`` values are decord indices of the input file. The two coincide only
    when the input *is* that source video.
    """
    records = [
        {
            "triplet": tuple(prediction["triplet"]),
            "score": prediction["score"],
            "sub_traj": prediction["sub_traj"],
            "obj_traj": prediction["obj_traj"],
            "duration": prediction["source_duration"],
            # The evaluator falls back to ``duration`` for both unless --allow_misalign; these
            # are informational here and correct either way.
            "sub_duration": prediction["source_duration"],
            "obj_duration": prediction["source_duration"],
            "track_id": prediction["track_id"],
            "traj_coord_space": "source_pixel_xyxy",
        }
        for prediction in predictions
    ]
    save_results(path, {video_id: records})
