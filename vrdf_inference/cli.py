"""Command line, config resolution, and mode dispatch for single-video inference."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from datasets.dataset import make_video_transforms
from main import get_args_parser
from vrdf_inference import results
from vrdf_inference.checkpoints import (
    load_checkpoint,
    load_class_names,
    load_model,
    resolve_device,
)
from vrdf_inference.console import die, log, warn
from vrdf_inference.render import render_detections, render_triplets
from vrdf_inference.stage1 import run_detection
from vrdf_inference.video import video_meta

# Keys that describe the *architecture* of the trained weights. A mismatch between a
# checkpoint's saved args and the effective args either breaks the state_dict load or
# silently changes what the network computes, so they are reported loudly.
ARCH_KEYS = (
    "stage",
    "num_queries",
    "focal_loss",
    "backbone",
    "dilation",
    "hidden_dim",
    "enc_layers",
    "dec_layers",
    "num_feature_levels",
    "deformable",
    "with_box_refine",
    "multi_frame_attention",
    "multi_frame_encoding",
    "merge_frame_features",
    "multi_frame_attention_separate_encoder",
    "position_embedding",
    "pre_norm",
)

# The checkpoint is the only record of the scale the weights actually saw, so for this one key it
# outranks the config JSON. Without that, every stage-2 run needs a remembered
# --stage2_resolution small or it silently predicts at the wrong scale: the checkpoint in
# data/ckpts/vidorsmall_stage2 was trained at "small", but configs/vidorsmall_stage2.json says
# "large". A CLI override still wins, so overriding stays possible.
CKPT_WINS_KEYS = ("resolution",)


def build_arg_parser():
    """``main``'s parser plus the inference-only options.

    ``conflict_handler="resolve"`` lets the re-registered ``--dataset_config`` replace the
    training parser's ``required=True`` version: inference names two configs instead
    (``--stage1_config`` / ``--stage2_config``) and never loads a dataloader.
    """
    parser = argparse.ArgumentParser(
        "VRDFormer single-video inference",
        parents=[get_args_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="unused in inference; kept so main's parser is satisfied",
    )

    parser.add_argument("--video", required=True, help="path to the input .mp4")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("stage1", "stage2"),
        help="stage1: detect and annotate only. stage2: detect, associate, classify, and annotate "
        "with relation triplets. Required: there is no default, so a cheap detection-only run and "
        "an expensive full run can never be confused for one another.",
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="output folder name; defaults to the input file stem",
    )
    parser.add_argument(
        "--stage1_ckpt", default="data/ckpts/vidorsmall_stage1/checkpoint0001.pth"
    )
    parser.add_argument(
        "--stage2_ckpt", default="data/ckpts/vidorsmall_stage2/checkpoint.pth"
    )
    parser.add_argument("--stage1_config", default="configs/vidorsmall_stage1.json")
    parser.add_argument("--stage2_config", default="configs/vidorsmall_stage2.json")

    parser.add_argument(
        "--output_dir",
        default=None,
        help="the per-mode output directory; results go to <output_dir>/<video_id>/ "
        "(default: output/inference/<mode>)",
    )
    parser.add_argument("--results_name", default="results.json")
    parser.add_argument("--video_name", default="video.mp4")
    parser.add_argument(
        "--eval_prediction_name",
        default="eval_prediction.json",
        help="sidecar in util/evaluate's prediction shape, so predicted triplets can be scored "
        "against ground truth; pass an empty string to skip it",
    )

    parser.add_argument(
        "--every_n_frames",
        "--stride",
        dest="every_n_frames",
        type=int,
        default=5,
        help="decode every N-th frame",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="cap the number of decoded frames (debug / quick smoke runs)",
    )
    parser.add_argument(
        "--decode_chunk",
        type=int,
        default=64,
        help="frames decoded per decord batch, keeps peak memory bounded",
    )

    parser.add_argument(
        "--conf_sub",
        default=0.15,
        type=float,
        help="sigmoid threshold on the Stage-1 subject class score. The vidorsmall Stage-1 "
        "checkpoint peaks around 0.2-0.3 on a real video, so anything near 0.5 yields no "
        "detections at all; check the 'best score before thresholding' line to calibrate",
    )
    parser.add_argument(
        "--conf_obj",
        default=0.15,
        type=float,
        help="sigmoid threshold on the Stage-1 object class score (see --conf_sub)",
    )
    parser.add_argument(
        "--conf_verb",
        default=0.15,
        type=float,
        help="report the Stage-1 detection verb only above this sigmoid score; "
        "does NOT drop the detection (the triplet verb comes from Stage 2)",
    )
    parser.add_argument(
        "--dedup_iou",
        default=0.9,
        type=float,
        help="per-frame NMS IoU among Stage-1 queries of the same (sub, obj) class",
    )
    parser.add_argument("--max_dets_per_frame", type=int, default=50)

    parser.add_argument(
        "--iou_thresh",
        default=0.3,
        type=float,
        help="min IoU on BOTH subject and object boxes to extend a tracklet",
    )
    parser.add_argument(
        "--max_gap",
        type=int,
        default=2,
        help="frames a tracklet stays matchable after a missed detection; the vidorsmall Stage-1 "
        "detector is sparse, so 1 splits most relations at every dropped frame",
    )
    parser.add_argument(
        "--min_tracklet_frames",
        type=int,
        default=1,
        help="drop tracklets observed in fewer decoded frames than this",
    )
    parser.add_argument(
        "--cls_gate",
        default="motion",
        choices=("motion", "same"),
        help="'motion': associate on box motion alone, then vote the classes back "
        "('same': require both class labels to match, as RUNPOD_INFERENCE.md describes)",
    )
    parser.add_argument(
        "--cls_vote",
        default="keep",
        choices=("keep", "first"),
        help="how a tracklet's reported classes are chosen when --cls_gate motion: "
        "'keep' = confidence-weighted vote over its frames, 'first' = the opening frame",
    )

    parser.add_argument(
        "--min_score",
        default=0.05,
        type=float,
        help="drop triplets whose sigmoid verb probability is below this",
    )
    parser.add_argument(
        "--max_labels_per_frame",
        type=int,
        default=8,
        help="draw text labels for at most this many detections/tracklets per frame (the most "
        "confident); boxes are still drawn for every one of them. results.json always has all",
    )

    parser.add_argument(
        "--stage1_resolution",
        default=None,
        choices=(None, "small", "middle", "large"),
        help="override the Stage-1 config's resolution",
    )
    parser.add_argument(
        "--stage2_resolution",
        default=None,
        choices=(None, "small", "middle", "large"),
        help="override the Stage-2 config's resolution",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="override the dataset name from both configs (class counts come from it)",
    )
    parser.add_argument(
        "--obj_list", default=None, help="default: data/<dataset>/obj.txt"
    )
    parser.add_argument(
        "--action_list", default=None, help="default: data/<dataset>/action.txt"
    )
    return parser


def resolve_stage_args(base_ns, config_path, ckpt_args, overrides):
    """Merge the layers that describe one stage, most specific last.

    Order: argparse defaults < the checkpoint's own saved args < the config JSON < CLI overrides,
    except for ``CKPT_WINS_KEYS`` where the checkpoint outranks the config.

    The checkpoint sits *above* the argparse defaults because its args are the only record of what
    the weights were actually built with -- ``num_queries`` is absent from
    ``configs/vidorsmall_stage*.json``, so without this a 200-query checkpoint would be loaded into
    a 100-query model and fail.
    """
    merged = dict(vars(base_ns))
    if ckpt_args is not None:
        merged.update(ckpt_args)
    with open(config_path, "r") as f:
        config = json.load(f)
    for key, value in config.items():
        # A CKPT_WINS_KEYS key the checkpoint actually stores keeps the checkpoint's value; every
        # other key, and any ckpt-wins key the checkpoint does not store, takes the config's.
        if key in CKPT_WINS_KEYS and ckpt_args is not None and key in ckpt_args:
            continue
        merged[key] = value
    merged.update({k: v for k, v in overrides.items() if v is not None})
    merged.pop("dataset_config", None)
    return argparse.Namespace(**merged)


def check_stage_args(stage, args, ckpt_args, config_path):
    """Report where the effective args disagree with the checkpoint that will be loaded."""
    if ckpt_args is None:
        warn(
            f"stage {stage}: checkpoint stores no args; cannot cross-check {config_path}"
        )
        return
    arch = [
        k
        for k in ARCH_KEYS
        if k in ckpt_args and k in vars(args) and vars(args)[k] != ckpt_args[k]
    ]
    if arch:
        details = ", ".join(
            f"{k}: ckpt={ckpt_args[k]!r} -> effective={vars(args)[k]!r}" for k in arch
        )
        warn(
            f"stage {stage}: effective config differs from the checkpoint's saved args for "
            f"{details}. Architecture mismatches will fail the state_dict load; the rest change "
            f"what the network computes."
        )
    if (
        "resolution" in ckpt_args
        and getattr(args, "resolution", None) != ckpt_args["resolution"]
    ):
        warn(
            f"stage {stage}: checkpoint was trained at resolution={ckpt_args['resolution']!r} but the "
            f"effective resolution is {getattr(args, 'resolution', None)!r}. Loading still works "
            f"(it only selects the val-transform sizes) but boxes will be predicted at a scale the "
            f"weights did not see; pass --stage{stage}_resolution {ckpt_args['resolution']} or "
            f"re-train to match."
        )


def main():
    args = build_arg_parser().parse_args()

    if args.every_n_frames < 1:
        die(f"--every_n_frames must be >= 1 (got {args.every_n_frames})")
    if not Path(args.video).exists():
        die(f"video not found: {args.video}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # Stage 1 is always loaded; stage 2's checkpoint is read only in stage-2 mode, so a
    # detection-only run works on a machine where no Stage-2 checkpoint exists.
    ckpt1, saved1 = load_checkpoint(args.stage1_ckpt)
    s1_args = resolve_stage_args(
        args,
        args.stage1_config,
        saved1,
        {"dataset": args.dataset, "resolution": args.stage1_resolution},
    )
    check_stage_args(1, s1_args, saved1, args.stage1_config)
    if s1_args.stage != 1:
        die(
            f"stage 1 config {args.stage1_config} resolves to stage={s1_args.stage}, expected 1"
        )
    if not s1_args.focal_loss:
        warn(
            "focal_loss is false for stage 1; class decoding here assumes the sigmoid-focal "
            "heads (80 object classes, no no-object slot)"
        )

    dataset = s1_args.dataset
    obj_list = args.obj_list or f"data/{dataset}/obj.txt"
    action_list = args.action_list or f"data/{dataset}/action.txt"
    obj_names, action_names = load_class_names(obj_list, action_list)
    # Bound here so the renderers and the payload builders can read them without threading the
    # class lists through every call.
    args.obj_names = obj_names
    args.action_names = action_names

    device = resolve_device(args.device)
    log(f"device: {device}")

    video_id = args.video_id or Path(args.video).stem
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output/inference") / args.mode
    )
    video_out_dir = output_dir / video_id

    meta = video_meta(args.video, args.every_n_frames, args.max_frames)
    meta["video_id"] = video_id
    meta["stride"] = args.every_n_frames
    log(
        f"video {args.video}: {meta['num_source_frames']} source frames, "
        f"{meta['num_decoded_frames']} decoded (stride {args.every_n_frames}), "
        f"{meta['fps']:.2f} fps, {meta['source_width']}x{meta['source_height']}"
    )
    log(
        f"mode {args.mode} -> {video_out_dir}; the annotated {args.video_name} is always written"
    )

    model_s1 = load_model(1, s1_args, ckpt1, device)
    transform_s1 = make_video_transforms(
        "val",
        s1_args.cautious,
        s1_args.by_ratio,
        s1_args.resolution,
        overflow_boxes=s1_args.overflow_boxes,
    )

    per_frame, resized_hw, source_hw = run_detection(
        model_s1, args.video, transform_s1, device, args
    )
    if resized_hw is None:
        die("no frame was decoded")
    meta["per_frame"] = per_frame
    meta["source_hw"] = source_hw

    if args.mode == "stage1":
        payload = results.detection_payload(
            args.video, video_id, meta, resized_hw, args, s1_args
        )
        results.save_results(video_out_dir / args.results_name, payload)
        # The annotated video is a required deliverable of both modes, including when nothing was
        # detected -- an empty annotated video is itself the diagnosis.
        render_detections(
            args.video, video_out_dir / args.video_name, per_frame, meta, args
        )
        log("done")
        log(f"  {video_out_dir / args.video_name}")
        log(f"  {video_out_dir / args.results_name}")
        return

    _run_stage2(
        args, meta, per_frame, resized_hw, source_hw, video_out_dir, device, s1_args
    )


def _run_stage2(
    args, meta, per_frame, resized_hw, source_hw, video_out_dir, device, s1_args
):
    """Association, classification, and triplet output. Only reachable from --mode stage2."""
    from vrdf_inference import stage2 as stage2_mod
    from vrdf_inference import tracklets as tracklets_mod

    ckpt2, saved2 = load_checkpoint(args.stage2_ckpt)
    s2_args = resolve_stage_args(
        args,
        args.stage2_config,
        saved2,
        {"dataset": args.dataset, "resolution": args.stage2_resolution},
    )
    check_stage_args(2, s2_args, saved2, args.stage2_config)
    if s2_args.stage != 2:
        die(
            f"stage 2 config {args.stage2_config} resolves to stage={s2_args.stage}, expected 2"
        )
    if not s2_args.focal_loss:
        warn(
            "focal_loss is false for stage 2; class decoding assumes the sigmoid-focal heads"
        )
    if s2_args.dataset != s1_args.dataset:
        warn(
            f"stage 1 uses dataset {s1_args.dataset!r} but stage 2 uses {s2_args.dataset!r}; "
            f"class lists come from {s1_args.dataset!r}"
        )

    model_s2 = load_model(2, s2_args, ckpt2, device)
    transform_s2 = make_video_transforms(
        "val",
        s2_args.cautious,
        s2_args.by_ratio,
        s2_args.resolution,
        overflow_boxes=s2_args.overflow_boxes,
    )

    tracklet_list = tracklets_mod.form_tracklets(per_frame, args)
    if not tracklet_list:
        warn("no tracklet survived; results.json will hold an empty predictions list")

    stage2_out, stage2_hw = (
        stage2_mod.classify_tracklets(
            model_s2,
            args.video,
            transform_s2,
            tracklet_list,
            device,
            args,
            s2_args.num_verb_classes,
        )
        if tracklet_list
        else (None, None)
    )
    predictions = (
        stage2_mod.build_predictions(
            stage2_out,
            tracklet_list,
            args.obj_names,
            args.action_names,
            args,
            meta,
            model_s2,
        )
        if stage2_out
        else []
    )
    if not predictions:
        warn(
            "no triplet cleared --min_score; the annotated video will carry detections only. "
            "Lower --min_score (and possibly --conf_sub/--conf_obj) to see more."
        )

    video_id = meta["video_id"]
    payload = results.triplet_payload(
        args.video,
        video_id,
        meta,
        resized_hw,
        args,
        s1_args,
        s2_args,
        stage2_hw,
        tracklet_list,
        predictions,
    )
    results.save_results(video_out_dir / args.results_name, payload)
    if args.eval_prediction_name:
        results.write_eval_prediction(
            video_out_dir / args.eval_prediction_name, predictions, video_id
        )
    render_triplets(
        args.video, video_out_dir / args.video_name, predictions, meta, args
    )

    log("done")
    log(f"  {video_out_dir / args.video_name}")
    log(f"  {video_out_dir / args.results_name}")
    if args.eval_prediction_name:
        log(f"  {video_out_dir / args.eval_prediction_name}")
    for prediction in predictions[:5]:
        log(
            f"  {prediction['score']:.3f}  {' '.join(prediction['triplet'])} "
            f"[{prediction['track_id']}] frames {prediction['frame_range']}"
        )
