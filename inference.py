"""Single-video inference for VRDFormer: detection (Stage 1) -> relation triplets (Stage 2).

Turns an arbitrary ``.mp4`` into an annotated ``video.mp4`` plus ``results.json``::

    <output_dir>/<video_id>/results.json          # triplets, tracklets, per-frame detections
    <output_dir>/<video_id>/video.mp4             # source video with boxes + triplet labels
    <output_dir>/<video_id>/eval_prediction.json  # sidecar in util/evaluate's shape

Why this file exists
--------------------
The repository has no inference path. ``engine.eval_stage2`` is an *oracle* evaluator: Stage 2
initialises its decoder queries by ``roi_align`` on **ground-truth** subject/object boxes
(``models/transformer.py:extract_roi_feat`` / ``prepare_tag_query``) and predicts a verb for each
ground-truth ``(subject_tid, object_tid)`` pair. It never enumerates pairs, so a Stage-2 checkpoint
alone cannot turn a raw video into triplets. This script supplies the three missing pieces:

1. a detector that produces subject/object boxes per frame (Stage 1, static queries),
2. association of those detections into persistent tracklets (IoU, scipy Hungarian),
3. a Stage-2 pass that treats the tracklets as pseudo-ground-truth.

Usage
-----

.. code-block:: console

    python inference.py \\
        --video data/vidorsmall/videos/10042935613.mp4 \\
        --stage1_ckpt data/ckpts/vidorsmall_stage1/checkpoint0001.pth \\
        --stage2_ckpt data/ckpts/vidorsmall_stage2/checkpoint.pth \\
        --stage1_config configs/vidorsmall_stage1.json \\
        --stage2_config configs/vidorsmall_stage2.json \\
        --every_n_frames 5 --device cuda

The checkpoint paths and both configs already default to the vidorsmall pair, so a minimal run is
``python inference.py --video <path>.mp4``. Read the docstring of :func:`main` for the full list of
thresholds and what each one costs.

Notable behaviour
-----------------
* **Frames** are decoded with decord at ``--every_n_frames`` stride and pushed through the *val*
  transform (``datasets/dataset.make_video_transforms``), the same one the dataset uses.
* **Stage-1 decoding uses ``sigmoid``, not softmax.** The heads are trained with
  ``sigmoid_focal_loss`` and, because ``focal_loss=true``, there is no no-object slot: each of the
  80 object classes is an independent binary. Stage 1 is called through the *base*
  ``VRDFormer.forward`` (with ``targets=None``), never ``VRDFormerTracking.forward``, which returns
  ``None`` without targets.
* **The Stage-1 verb head is not used for the triplet.** It is recorded per detection as
  ``verb``/``verb_score`` for reference, but the reported verb comes from Stage 2 — that is the head
  trained for the subject-object relation. ``--conf_verb`` only decides whether the Stage-1 verb
  annotation is reported at all.
* **``resolution`` matters twice and can differ per stage.** It selects the val-transform sizes, and
  a Stage-2 checkpoint carries the resolution it was trained at. The two stages are independent, so
  ``--stage1_resolution`` / ``--stage2_resolution`` override the per-stage config value; a mismatch
  with a checkpoint's own setting is reported as a warning.
* **Tracklets survive gaps but not association failures.** ``--max_gap`` frames of missed detection
  keep a tracklet matchable, but a frame where the detector produced *no* box at all cannot be
  bridged — at stride 5 the vidorsmall Stage-1 checkpoint loses most objects within a couple of
  seconds, which is the main reason predicted tracklets are short. ``sub_traj``/``obj_traj`` are
  expanded to the source-frame axis so ``util.evaluate.viou`` (which assumes a contiguous
  trajectory) is well-formed; ``observed_frames`` says which entries were real detections.
* **The models are only as good as vidorsmall**: a 100-video POC subset trained for 2 epochs, whose
  Stage-1 confidence peaks near 0.2 and decays to ~0.08 within a second of video. Low scores, thin
  triplets and frame-span mismatches against ground truth are expected and are not necessarily bugs.
  Every run prints the best pre-threshold score so ``--conf_sub``/``--conf_obj`` can be calibrated
  from the data rather than guessed; a measured reference point is 0.59 box recall at IoU 0.30 on
  ``data/vidorsmall/videos/10042935613.mp4`` at ``--conf_sub/--conf_obj 0.15``.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import util.misc as utils
from datasets.dataset import make_video_transforms
from main import get_args_parser
from models import build_model
from models.vrdformer import VRDFormer
from models.vrdformer_track import VRDFormerTracking
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

LOG = "[inference]"


def log(msg):
    print(f"{LOG} {msg}", flush=True)


def warn(msg):
    print(f"{LOG} WARNING: {msg}", flush=True)


def die(msg):
    print(f"{LOG} ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------------------

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
        help="root output dir; per-video results go to <output_dir>/<video_id>/ "
        "(default: output/inference/<dataset>)",
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
        "--no_video", action="store_true", help="skip writing the annotated mp4"
    )
    parser.add_argument(
        "--max_labels_per_frame",
        type=int,
        default=8,
        help="draw text labels for at most this many tracklets per frame (the most confident); "
        "boxes are still drawn for every active tracklet. results.json always has them all",
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

    Order: argparse defaults < the checkpoint's own saved args < the config JSON < CLI
    overrides. The checkpoint sits *above* the argparse defaults because its args are the
    only record of what the weights were actually built with -- ``num_queries`` is absent
    from ``configs/vidorsmall_stage*.json``, so without this a 200-query checkpoint would
    be loaded into a 100-query model and fail.
    """
    merged = dict(vars(base_ns))
    if ckpt_args is not None:
        merged.update(ckpt_args)
    with open(config_path, "r") as f:
        merged.update(json.load(f))
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


# --------------------------------------------------------------------------------------
# Checkpoints, class names, device
# --------------------------------------------------------------------------------------


def load_class_names(obj_path, action_path):
    def read(path):
        # VidOR's class lists are CRLF-terminated.
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    obj_names = read(obj_path)
    action_names = read(action_path)
    log(
        f"classes: {len(obj_names)} objects from {obj_path}, {len(action_names)} verbs from {action_path}"
    )
    return obj_names, action_names


def load_checkpoint(path):
    if not Path(path).exists():
        die(f"checkpoint not found: {path}")
    # weights_only=False: checkpoints pickle an argparse.Namespace (see util/checkpoints.py).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        die(f"{path} has no 'model' key (found {sorted(ckpt.keys())})")
    saved_args = ckpt.get("args")
    saved_args = (
        vars(saved_args) if isinstance(saved_args, argparse.Namespace) else saved_args
    )
    log(f"loaded {path} (epoch {ckpt.get('epoch')}, {len(ckpt['model'])} tensors)")
    return ckpt, saved_args


def resolve_device(requested):
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            die(
                f"--device {requested} was requested but CUDA is unavailable on this machine. "
                f"Run on the RunPod pod, or pass --device cpu (slow but functional)."
            )
        return torch.device(requested)
    return torch.device(requested)


def load_model(stage, args, ckpt, device):
    """Build one stage and load its weights strictly.

    ``ckpt`` is the already-loaded checkpoint dict, so the (large) file is read once per stage.
    """
    args = argparse.Namespace(**vars(args))
    args.device = str(device)
    # build_model dispatches on args.tracking for stage 1; the tracking wrapper needs the
    # matcher and returns None without targets, so inference always builds the base class.
    args.tracking = False
    model, _, _ = build_model(args)
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        # Saved from a DDP-wrapped module; unwrap so the load is not defeated by the prefix.
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"stage {stage}: {type(model).__name__} built and loaded ({n_params / 1e6:.2f}M params)"
    )
    return model


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------


DECODER_HELP = (
    "decord is built against FFmpeg and ships without some newer codecs. AV1 (and some 10-bit "
    "HEVC) commonly fail here; OpenCV cannot read AV1 either, so there is no in-script fallback. "
    "Transcode with ffmpeg first, then re-run -- for example: "
    "ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4"
)


def open_video(video_path):
    """Open a video with decord, turning decode failures into an actionable message."""
    from decord import VideoReader, cpu

    if not Path(video_path).exists():
        die(f"video not found: {video_path}")
    try:
        return VideoReader(video_path, ctx=cpu(0))
    except Exception as exc:  # noqa: BLE001 - decord raises several unrelated types
        die(
            f"could not decode {video_path}: {type(exc).__name__}: {exc}\n{DECODER_HELP}"
        )


def iter_decoded_frames(video_path, stride, max_frames=None, chunk=64):
    """Yield ``(decoded_index, source_frame_id, rgb_uint8_hwc)`` with decord.

    Reads in batches so peak memory stays proportional to ``chunk`` rather than to the
    length of the video; the caller can walk the video as many times as it needs.
    """
    vr = open_video(video_path)
    total = len(vr)
    stride = max(1, int(stride))
    ids = list(range(0, total, stride))
    if max_frames is not None:
        ids = ids[:max_frames]
    if not ids:
        die(f"no frames selected from {video_path} (len={total}, stride={stride})")

    for start in range(0, len(ids), chunk):
        block = ids[start : start + chunk]
        try:
            array = vr.get_batch(block).asnumpy()
        except Exception as exc:  # noqa: BLE001
            die(
                f"could not decode frames {block[0]}-{block[-1]} of {video_path}: "
                f"{type(exc).__name__}: {exc}\n{DECODER_HELP}"
            )
        if array.shape[0] != len(block):
            die(
                f"decord returned {array.shape[0]} frames for a request of {len(block)} from "
                f"{video_path}; the container's frame index is unreliable (variable frame rate "
                f"or a damaged file)."
            )
        for offset, source_fid in enumerate(block):
            yield start + offset, int(source_fid), array[offset]


def video_meta(video_path, stride, max_frames):
    """Cheap metadata pass (opens the container, decodes nothing)."""
    vr = open_video(video_path)
    total = len(vr)
    if total == 0:
        die(f"{video_path} reports zero frames; nothing to run inference on")
    ids = list(range(0, total, max(1, int(stride))))
    if max_frames is not None:
        ids = ids[:max_frames]
    if not ids:
        die(
            f"no frames selected from {video_path} (len={total}, stride={stride}); "
            f"check --every_n_frames and --max_frames"
        )
    try:
        first_frame = vr[0]
    except Exception as exc:  # noqa: BLE001
        die(
            f"could not decode the first frame of {video_path}: "
            f"{type(exc).__name__}: {exc}\n{DECODER_HELP}"
        )
    return {
        "num_source_frames": total,
        "num_decoded_frames": len(ids),
        "frame_ids": [int(i) for i in ids],
        "fps": float(vr.get_avg_fps()) or 30.0,
        "source_width": int(first_frame.shape[1]),
        "source_height": int(first_frame.shape[0]),
    }


def transform_frame(transform, frame_rgb):
    """Apply the dataset's val transform to one frame, returning ``(3, H, W)`` and its size.

    ``make_video_transforms("val", ...)`` is deterministic and per-frame, so running it one
    frame at a time is equivalent to running it on the whole clip -- and it keeps us from
    materialising a ``(3, T, H, W)`` float tensor for a long video. The transform still returns
    the clip axis (here of length 1), hence the squeeze.
    """
    tensor, _ = transform([frame_rgb], None)
    if tensor.ndim == 4:
        tensor = tensor[:, 0]
    channels, height, width = tensor.shape
    assert channels == 3, tensor.shape
    return tensor, height, width


# --------------------------------------------------------------------------------------
# Stage 1: detection
# --------------------------------------------------------------------------------------


def _iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_stage1_frame(out, resized_hw, source_hw, args):
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

    kept = []
    for det in dets:
        duplicate = False
        for prev in kept:
            if (det["sub_cls"], det["obj_cls"]) != (prev["sub_cls"], prev["obj_cls"]):
                continue
            overlap = min(
                _iou_xyxy(det["box_n_sub"], prev["box_n_sub"]),
                _iou_xyxy(det["box_n_obj"], prev["box_n_obj"]),
            )
            if overlap >= args.dedup_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept, ceilings


@torch.no_grad()
def run_stage1(model, video_path, transform, device, args):
    """Detect on every decoded frame, carrying the previous frame's backbone features.

    Stage 1 was trained with ``multi_frame_attention``, so consecutive frames must be fed in
    order with ``prev_features``; frame 0 passes ``None`` and the model falls back to the
    current frame (``models/vrdformer.py``).
    """
    per_frame = []
    prev_features = None
    resized_hw = None
    source_hw = None
    started = time.time()
    # Highest score any query reached in each frame, before thresholding. This is the number
    # that tells you how to set --conf_sub/--conf_obj on a new video.
    ceilings = []
    for index, source_fid, frame_rgb in iter_decoded_frames(
        video_path, args.every_n_frames, args.max_frames, args.decode_chunk
    ):
        frame_tensor, height, width = transform_frame(transform, frame_rgb)
        resized_hw = (height, width)
        source_hw = frame_rgb.shape[:2]
        samples = utils.NestedTensor.from_tensor_list([frame_tensor]).to(device)

        out, _, features_all, _, _ = VRDFormer.forward(
            model, samples, targets=None, prev_features=prev_features
        )
        prev_features = features_all

        dets, ceiling = decode_stage1_frame(out, resized_hw, source_hw, args)
        per_frame.append(dets)
        ceilings.append(ceiling)
        if index % 20 == 0:
            elapsed = time.time() - started
            rate = (index + 1) / elapsed if elapsed > 0 else 0.0
            log(
                f"stage 1: frame {index} ({source_fid}) -> {len(dets)} detections "
                f"[{rate:.1f} fps, {elapsed:.1f}s elapsed]"
            )

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
    return per_frame, resized_hw, source_hw


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


# --------------------------------------------------------------------------------------
# Tracklets
# --------------------------------------------------------------------------------------


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
            n_pending, n_dets = len(pending), len(dets)
            cost = np.zeros((n_pending, n_dets), dtype=np.float64)
            allowed = np.zeros((n_pending, n_dets), dtype=bool)
            for i, p in enumerate(pending):
                for j, det in enumerate(dets):
                    iou_sub = _iou_xyxy(p["box_n_sub"], det["box_n_sub"])
                    iou_obj = _iou_xyxy(p["box_n_obj"], det["box_n_obj"])
                    cost[i, j] = min(iou_sub, iou_obj)
                    allowed[i, j] = cost[i, j] >= args.iou_thresh
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
            tracklet["sub_cls"] = _weighted_vote(votes_sub, weights)
            tracklet["obj_cls"] = _weighted_vote(votes_obj, weights)

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


def _weighted_vote(values, weights):
    """Most common value, ties broken by summed confidence (higher wins)."""
    totals = {}
    for value, weight in zip(values, weights):
        totals[value] = totals.get(value, 0.0) + weight
    return max(sorted(totals), key=lambda v: totals[v])


# --------------------------------------------------------------------------------------
# Stage 2: relation classification
# --------------------------------------------------------------------------------------


def build_pseudo_target(active, frame_index, device, num_verb_classes):
    """Assemble the one pseudo-GT dict ``VRDFormer_S2.forward`` reads for a single frame.

    Keys and conventions follow ``datasets/dataset.py:prepare_data_stage2`` and
    ``models/transformer.py``: ``unscaled_{sub,obj}_boxes`` are xyxy in **resized** pixels
    (``roi_align`` input at spatial scale 1/32), while ``{sub,obj}_boxes`` are normalized
    cxcywh (read by ``memory_update``).
    """
    sub_resized = torch.as_tensor(
        np.stack([t["frames"][frame_index]["box_resized_sub"] for t in active]),
        dtype=torch.float32,
    )
    obj_resized = torch.as_tensor(
        np.stack([t["frames"][frame_index]["box_resized_obj"] for t in active]),
        dtype=torch.float32,
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
        "unscaled_sub_boxes": sub_resized.to(device),
        "unscaled_obj_boxes": obj_resized.to(device),
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
def run_stage2(model, video_path, transform, tracklets, device, args, num_verb_classes):
    """Accumulate per-tracklet ``rel_embed`` over the video, then classify each pair once.

    Mirrors ``engine.eval_stage2`` but with ``is_eval=False``: the non-eval branch of
    ``relation_classifier`` walks every memory key, mean-pools that pair's ``rel_embed`` over
    the frames it was seen in, and returns the sub/verb/obj logits -- exactly the
    per-tracklet predictions we want. ``eos`` stays False so the classification happens here,
    once, with an explicit key ordering to map back to tracklets.
    """
    by_frame = {}
    for tracklet in tracklets:
        for frame_index in tracklet["frame_indices"]:
            by_frame.setdefault(frame_index, []).append(tracklet)

    memory = None
    frames_run = 0
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
        frame_tensor, _height, _width = transform_frame(transform, frame_rgb)
        samples = utils.NestedTensor.from_tensor_list([frame_tensor]).to(device)
        target = build_pseudo_target(active, index, device, num_verb_classes)
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
        return None
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
    }


@torch.no_grad()
def build_predictions(
    stage2_out, tracklets, obj_names, action_names, args, meta, model_s2
):
    """Turn the Stage-2 logits into the per-tracklet prediction records.

    ``relation_classifier``'s non-eval branch returns one row per memory key, and
    ``run_stage2`` inserts keys in ``tracklets`` order, so ``so_keys`` maps back to tracklets
    by id rather than by position.
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


def det_box(tracklet, frame_index, key):
    det = tracklet["frames"].get(frame_index)
    return [round(float(v), 2) for v in det[key]] if det is not None else None


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def save_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


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
        }
        for prediction in predictions
    ]
    save_results(path, {video_id: records})


# A distinct colour per tracklet; BGR, as OpenCV expects.
_PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (49, 210, 207),
    (45, 129, 190),
    (152, 135, 31),
    (60, 190, 255),
    (190, 60, 200),
    (120, 220, 60),
    (255, 190, 60),
    (200, 130, 255),
    (30, 200, 120),
    (240, 100, 100),
    (100, 240, 240),
    (160, 60, 60),
    (60, 60, 160),
    (0, 165, 255),
    (128, 0, 128),
    (0, 100, 0),
    (203, 192, 255),
    (140, 180, 210),
]


def _put_label(image, text, origin, color, scale, thickness):
    import cv2

    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = origin
    height, width = image.shape[:2]
    x = max(0, min(x, width - text_w - 2))
    y = max(text_h + 2, min(y, height - baseline - 2))
    cv2.rectangle(
        image,
        (x - 1, y - text_h - 2),
        (x + text_w + 1, y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_video(video_path, out_path, predictions, meta, args):
    """Draw every triplet's subject/object boxes and label on every frame it is active in.

    Frames are re-decoded rather than held in memory, so the annotated video costs one more
    sequential read of the input instead of a full-resolution frame cache.

    On a small frame (this pipeline is commonly run on 320x240 clips) a box per tracklet plus a
    label per box is unreadable, so only the strongest ``max_labels_per_frame`` tracklets get a
    text label -- boxes are still drawn for all of them, and their colour still identifies the
    tracklet. The full text of every triplet is in ``results.json`` regardless.
    """
    import cv2

    colors = {
        p["track_id"]: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(predictions)
    }
    by_frame = {}
    for prediction in predictions:
        # Position of each decoded frame inside the prediction's own frame list.
        positions = {
            frame_index: i for i, frame_index in enumerate(prediction["frames"])
        }
        for frame_index in prediction["observed_frames"]:
            by_frame.setdefault(frame_index, []).append(
                (prediction, positions[frame_index])
            )

    source_w, source_h = meta["source_width"], meta["source_height"]
    # Legibility is about the drawn pixels, not the frame's nominal size: at 320x240 the old
    # 0.4 floor produced ~5px text. Anchor the scale to the frame's height so both tiny and
    # large inputs get a readable label.
    scale = min(1.2, max(0.45, source_h / 400.0))
    thickness = max(1, round(source_h / 320.0))
    max_labels = max(1, args.max_labels_per_frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    writer_codec = None
    written = 0
    for index, source_fid, frame_rgb in iter_decoded_frames(
        video_path, args.every_n_frames, args.max_frames, args.decode_chunk
    ):
        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        active = by_frame.get(index, [])
        # Label the most confident tracklets only; ties keep the earlier tracklet's colour stable.
        labelled = {
            id(prediction)
            for prediction, _ in sorted(active, key=lambda pair: -pair[0]["score"])[
                :max_labels
            ]
        }

        for prediction, position in active:
            color = colors[prediction["track_id"]]
            tops = []
            for key in ("sub_boxes", "obj_boxes"):
                xyxy = prediction[key][position]
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = [round(v) for v in xyxy]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                tops.append((y1, x1))
            if not tops or id(prediction) not in labelled:
                continue
            label = (
                f"{prediction['subject']} {prediction['verb']} {prediction['object']} "
                f"[{prediction['track_id']}] {prediction['score']:.2f}"
            )
            top_y, left_x = min(tops)
            _put_label(canvas, label, (left_x, top_y - 4), color, scale, thickness)

        header = (
            f"{meta['video_id']}  frame {index}/{meta['num_decoded_frames'] - 1} "
            f"(source {source_fid})  {len(active)} active relation(s)"
        )
        _put_label(
            canvas, header, (8, source_h - 8), (255, 255, 255), scale * 0.9, thickness
        )

        frame_h, frame_w = canvas.shape[:2]
        if (frame_w, frame_h) != (source_w, source_h):
            # The writer's frame size is fixed at open time, and decord's reported size can
            # disagree with the decoded pixels (odd containers, rotation metadata). Resize
            # rather than write a skewed stream or have every frame silently dropped.
            if written == 0:
                warn(
                    f"decoded frames are {frame_w}x{frame_h} but the container reports "
                    f"{source_w}x{source_h}; resizing to the reported size"
                )
            canvas = cv2.resize(canvas, (source_w, source_h))

        if writer is None:
            fps = max(1.0, meta["fps"] / max(1, args.every_n_frames))
            for codec in ("mp4v", "avc1", "H264"):
                candidate = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*codec),
                    fps,
                    (source_w, source_h),
                )
                if candidate.isOpened():
                    writer, writer_codec = candidate, codec
                    break
                candidate.release()
            if writer is None:
                die(
                    f"could not open a VideoWriter for {out_path}; no working mp4 codec "
                    f"(tried mp4v, avc1, H264). Use --no_video to skip rendering."
                )
            log(f"writing {out_path} at {fps:.2f} fps ({writer_codec})")
        writer.write(canvas)
        written += 1

    if writer is not None:
        writer.release()
    log(f"wrote {out_path} ({written} frames)")


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


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.every_n_frames < 1:
        die(f"--every_n_frames must be >= 1 (got {args.every_n_frames})")
    if not Path(args.video).exists():
        die(f"video not found: {args.video}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    ckpt1, saved1 = load_checkpoint(args.stage1_ckpt)
    ckpt2, saved2 = load_checkpoint(args.stage2_ckpt)

    # Only explicit flags become overrides. argparse defaults must NOT: the parser's
    # num_queries=100 / position_embedding="sine" are the *training* script's defaults and have
    # nothing to do with these checkpoints, so letting them win would build a 100-query model
    # and fail to load 200-query weights.
    s1_args = resolve_stage_args(
        args,
        args.stage1_config,
        saved1,
        {"dataset": args.dataset, "resolution": args.stage1_resolution},
    )
    s2_args = resolve_stage_args(
        args,
        args.stage2_config,
        saved2,
        {"dataset": args.dataset, "resolution": args.stage2_resolution},
    )
    check_stage_args(1, s1_args, saved1, args.stage1_config)
    check_stage_args(2, s2_args, saved2, args.stage2_config)

    if s1_args.stage != 1:
        die(
            f"stage 1 config {args.stage1_config} resolves to stage={s1_args.stage}, expected 1"
        )
    if s2_args.stage != 2:
        die(
            f"stage 2 config {args.stage2_config} resolves to stage={s2_args.stage}, expected 2"
        )
    if not s1_args.focal_loss or not s2_args.focal_loss:
        warn(
            "focal_loss is false for at least one stage; class decoding here assumes the "
            "sigmoid-focal heads (80 object classes, no no-object slot)"
        )

    dataset = s1_args.dataset
    obj_list = args.obj_list or f"data/{dataset}/obj.txt"
    action_list = args.action_list or f"data/{dataset}/action.txt"
    if s2_args.dataset != dataset:
        warn(
            f"stage 1 uses dataset {dataset!r} but stage 2 uses {s2_args.dataset!r}; "
            f"class lists come from {dataset!r}"
        )
        obj_list = args.obj_list or f"data/{dataset}/obj.txt"
        action_list = args.action_list or f"data/{dataset}/action.txt"
    obj_names, action_names = load_class_names(obj_list, action_list)

    device = resolve_device(args.device)
    log(f"device: {device}")

    video_id = args.video_id or Path(args.video).stem
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("output/inference") / dataset
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
    # Both stages run once per decoded frame, and rendering decodes the video a third time. A
    # long clip at a small stride is thousands of forward passes, so state the workload before
    # starting rather than after -- this is the number to lower --every_n_frames / --max_frames
    # against on a rented GPU.
    log(
        f"workload: {meta['num_decoded_frames']} frames x 2 stages + 1 render pass = "
        f"~{3 * meta['num_decoded_frames']} video passes "
        f"(cover {meta['num_source_frames'] / max(meta['fps'], 1e-6):.1f}s of source video)"
    )

    model_s1 = load_model(1, s1_args, ckpt1, device)
    model_s2 = load_model(2, s2_args, ckpt2, device)
    if isinstance(model_s1, VRDFormerTracking):
        die(
            "stage 1 built as VRDFormerTracking; inference needs the base VRDFormer.forward"
        )

    # The val transform is deterministic and does not depend on the number of frames, so build
    # it once per stage and apply it frame by frame.
    transform_s1 = make_video_transforms(
        "val",
        s1_args.cautious,
        s1_args.by_ratio,
        s1_args.resolution,
        overflow_boxes=s1_args.overflow_boxes,
    )
    transform_s2 = make_video_transforms(
        "val",
        s2_args.cautious,
        s2_args.by_ratio,
        s2_args.resolution,
        overflow_boxes=s2_args.overflow_boxes,
    )

    per_frame_dets, resized_hw, source_hw = run_stage1(
        model_s1, args.video, transform_s1, device, args
    )
    report_stage1_scores(per_frame_dets)
    if resized_hw is None:
        die("no frame was decoded")

    tracklets = form_tracklets(per_frame_dets, args)
    if not tracklets:
        warn("no tracklet survived; results.json will hold an empty predictions list")

    stage2_out = (
        run_stage2(
            model_s2,
            args.video,
            transform_s2,
            tracklets,
            device,
            args,
            s2_args.num_verb_classes,
        )
        if tracklets
        else None
    )
    predictions = (
        build_predictions(
            stage2_out, tracklets, obj_names, action_names, args, meta, model_s2
        )
        if stage2_out
        else []
    )

    if not predictions:
        warn(
            "no triplet cleared --min_score; the annotated video will carry detections only. "
            "Lower --min_score (and possibly --conf_sub/--conf_obj) to see more."
        )

    payload = {
        "video": args.video,
        "video_id": video_id,
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
            "resolution": s1_args.resolution,
        },
        "stage2": {
            "checkpoint": args.stage2_ckpt,
            "config": args.stage2_config,
            "resolution": s2_args.resolution,
        },
        "thresholds": {
            "conf_sub": args.conf_sub,
            "conf_obj": args.conf_obj,
            "conf_verb": args.conf_verb,
            "dedup_iou": args.dedup_iou,
            "iou_thresh": args.iou_thresh,
            "max_gap": args.max_gap,
            "min_score": args.min_score,
        },
        "num_detections": int(sum(len(d) for d in per_frame_dets)),
        "num_tracklets": len(tracklets),
        "num_predictions": len(predictions),
        "predictions": predictions,
        "detections": summarise_detections(
            per_frame_dets, obj_names, action_names, source_hw, args
        ),
    }
    save_results(video_out_dir / args.results_name, payload)
    if args.eval_prediction_name:
        write_eval_prediction(
            video_out_dir / args.eval_prediction_name, predictions, video_id
        )

    if not args.no_video:
        render_video(
            args.video, video_out_dir / args.video_name, predictions, meta, args
        )

    log("done")
    log(f"  {video_out_dir / args.results_name}")
    if not args.no_video:
        log(f"  {video_out_dir / args.video_name}")
    for prediction in predictions[:5]:
        log(
            f"  {prediction['score']:.3f}  {' '.join(prediction['triplet'])} "
            f"[{prediction['track_id']}] frames {prediction['frame_range']}"
        )


if __name__ == "__main__":
    main()
