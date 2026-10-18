"""Annotated-video rendering: one renderer per mode, sharing the label and codec machinery.

Both renderers are always called. The annotated video is a required deliverable of both modes, so
neither has a "skip rendering" path, and both write a playable file even when there is nothing to
draw -- an empty annotated video is itself the diagnosis.
"""

from vrdf_inference.console import die, log, warn
from vrdf_inference.video import iter_decoded_frames

# A distinct colour per tracklet; BGR, as OpenCV expects.
PALETTE = [
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

# Stage 1 has no tracklets, so a box's meaning comes from which side of the pair it is. BGR.
SUB_COLOR = (255, 190, 60)
OBJ_COLOR = (60, 190, 255)


def put_label(image, text, origin, color, scale, thickness):
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


def label_style(source_h):
    """Text scale and thickness for a frame this tall.

    Legibility is about the drawn pixels, not the frame's nominal size: at 320x240 a fixed 0.4
    floor produced ~5px text, so the scale is anchored to the frame height instead.
    """
    return min(1.2, max(0.45, source_h / 400.0)), max(1, round(source_h / 320.0))


def draw_header(canvas, text, source_h, scale, thickness):
    """The one-line status strip along the bottom of every frame."""
    put_label(canvas, text, (8, source_h - 8), (255, 255, 255), scale * 0.9, thickness)


def open_writer(out_path, fps, size):
    """Open an mp4 writer, trying codecs until one works.

    OpenCV's available codecs vary by build and platform, so the first three that commonly encode
    mp4 are tried in order. Failing all three is fatal: the annotated video is a required
    deliverable, so there is nothing sensible to continue with.
    """
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            log(f"writing {out_path} at {fps:.2f} fps ({codec})")
            return writer
        writer.release()
    die(
        f"could not open a VideoWriter for {out_path}; no working mp4 codec "
        f"(tried mp4v, avc1, H264)"
    )


def _fit_to_container(canvas, source_w, source_h, written):
    """Resize a frame the container disagrees about, warning once.

    The writer's frame size is fixed at open time, and decord's reported size can disagree with the
    decoded pixels (odd containers, rotation metadata). Resizing beats writing a skewed stream or
    having every frame silently dropped.
    """
    import cv2

    frame_h, frame_w = canvas.shape[:2]
    if (frame_w, frame_h) == (source_w, source_h):
        return canvas
    if written == 0:
        warn(
            f"decoded frames are {frame_w}x{frame_h} but the container reports "
            f"{source_w}x{source_h}; resizing to the reported size"
        )
    return cv2.resize(canvas, (source_w, source_h))


def render_detections(video_path, out_path, per_frame, meta, args):
    """Annotate a detection-only run: the subject and object boxes with their classes and scores.

    Boxes are drawn for every kept detection. Subject and object use two fixed colours so a pair
    reads as a pair, and the label ``adult 0.19 / table 0.17`` sits at the pair's top-left corner.
    On a small frame a label per detection is unreadable, so only the ``--max_labels_per_frame``
    most confident pairs are labelled; every pair still gets its boxes.

    Frames are re-decoded rather than cached, so the annotated video costs one more sequential read
    of the input instead of a full-resolution frame cache.
    """
    import cv2

    names = args.obj_names
    source_w, source_h = meta["source_width"], meta["source_height"]
    scale, thickness = label_style(source_h)
    max_labels = max(1, args.max_labels_per_frame)

    writer = None
    written = 0
    for index, source_fid, frame_rgb in iter_decoded_frames(
        video_path, args.every_n_frames, args.max_frames, args.decode_chunk
    ):
        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections = per_frame[index] if index < len(per_frame) else []
        # Strongest pairs get the text label; every pair still gets its boxes.
        labelled = {
            id(det)
            for det in sorted(detections, key=lambda d: -d["pair_score"])[:max_labels]
        }

        for det in detections:
            tops = []
            for key, color in (("box_px_sub", SUB_COLOR), ("box_px_obj", OBJ_COLOR)):
                x1, y1, x2, y2 = [round(float(v)) for v in det[key]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                tops.append((y1, x1))
            if id(det) not in labelled:
                continue
            label = (
                f"{names[det['sub_cls']]} {det['sub_score']:.2f} / "
                f"{names[det['obj_cls']]} {det['obj_score']:.2f}"
            )
            top_y, left_x = min(tops)
            put_label(canvas, label, (left_x, top_y - 4), SUB_COLOR, scale, thickness)

        header = (
            f"{meta['video_id']}  frame {index}/{meta['num_decoded_frames'] - 1} "
            f"(source {source_fid})  {len(detections)} detection(s)"
        )
        draw_header(canvas, header, source_h, scale, thickness)
        canvas = _fit_to_container(canvas, source_w, source_h, written)

        if writer is None:
            fps = max(1.0, meta["fps"] / max(1, args.every_n_frames))
            writer = open_writer(out_path, fps, (source_w, source_h))
        writer.write(canvas)
        written += 1

    if writer is not None:
        writer.release()
    log(f"wrote {out_path} ({written} frames)")


def render_triplets(video_path, out_path, predictions, meta, args):
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
        p["track_id"]: PALETTE[i % len(PALETTE)] for i, p in enumerate(predictions)
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
    scale, thickness = label_style(source_h)
    max_labels = max(1, args.max_labels_per_frame)

    writer = None
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
            put_label(canvas, label, (left_x, top_y - 4), color, scale, thickness)

        header = (
            f"{meta['video_id']}  frame {index}/{meta['num_decoded_frames'] - 1} "
            f"(source {source_fid})  {len(active)} active relation(s)"
        )
        draw_header(canvas, header, source_h, scale, thickness)
        canvas = _fit_to_container(canvas, source_w, source_h, written)

        if writer is None:
            fps = max(1.0, meta["fps"] / max(1, args.every_n_frames))
            writer = open_writer(out_path, fps, (source_w, source_h))
        writer.write(canvas)
        written += 1

    if writer is not None:
        writer.release()
    log(f"wrote {out_path} ({written} frames)")
