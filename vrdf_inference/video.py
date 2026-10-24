"""Frame IO for inference: decord decode, video metadata, and the dataset's val transform.

Nothing here knows about models. The stride / ``--max_frames`` selection lives in
:func:`selected_frame_ids`, which both the decode loop and the metadata pass use, so the frame axis
reported in ``results.json`` is by construction the one the pipeline actually processed.
"""

from pathlib import Path

from vrdf_inference.console import die

DECODER_HELP = (
    "decord is built against FFmpeg and ships without some newer codecs. AV1 (and some 10-bit "
    "HEVC) commonly fail here; OpenCV cannot read AV1 either, so there is no in-script fallback. "
    "Transcode with ffmpeg first, then re-run -- for example: "
    "ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4"
)


def selected_frame_ids(total, stride, max_frames=None):
    """The source frame ids a run will visit, in order.

    Both the decode loop and the metadata pass call this, so the frame count in ``results.json``
    is by construction the number of frames the pipeline actually processed.
    """
    ids = list(range(0, total, max(1, int(stride))))
    if max_frames is not None:
        ids = ids[:max_frames]
    return ids


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
    ids = selected_frame_ids(total, stride, max_frames)
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
    ids = selected_frame_ids(total, stride, max_frames)
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
