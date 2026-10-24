# Single-Video Inference Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1587-line `inference.py` into a `vrdf_inference/` package of single-purpose
modules and add a required `--mode stage1|stage2` flag, so detection-only and full-pipeline runs are
two obvious commands writing to two obvious output trees, with the annotated `video.mp4` a required
deliverable of both.

**Architecture:** `inference.py` at the repo root becomes a thin entry point (path setup + delegate).
All logic moves into `vrdf_inference/`: `console` (logging), `video` (decord IO), `checkpoints`
(load + build one stage), `stage1` (detection), `tracklets` (association, stage-2 only), `stage2`
(accumulation + classification), `render` (two renderers), `results` (JSON payloads), and `cli`
(arguments, config resolution, mode dispatch). Stage-2 logic is moved verbatim except for the
`--no_video` removal; stage-1 is the same detection path with a new detection-only renderer.

**Tech Stack:** Python 3.10+, PyTorch 2.5+ (CPU or CUDA), decord, OpenCV, SciPy (`linear_sum_assignment`), Ruff.

**Reference spec:** `docs/superpowers/specs/2026-10-18-inference-modularization-design.md`

---

## Testing approach for this repo (read first)

This repository has **no pytest suite and no CI**, and `pytest` is not installed in `.venv`. Do not
add it: the spec's non-goals say no new dependency, and `CLAUDE.md` documents that verification here
bottoms out at Ruff plus small self-checking scripts (the `models/ops/test*.py` idiom).

So each pure-logic task below ends with a check in **`scripts/inference/selfcheck.py`** — a plain
script of `assert`s with a printed summary, run with `.venv/bin/python`. It imports only pure
functions, so it needs no checkpoint and no GPU. Model-loading and end-to-end behaviour are verified
by the real smoke runs in Task 12 (Stage A of the spec) and later on RunPod (Stage B).

**Two conventions used throughout:**

1. **Moved code is referenced by line range, not re-pasted.** Where a task says "move
   `inference.py:1117-1159` verbatim", the exact source is in git at `HEAD` and re-pasting 400 lines
   into this plan would only create a second copy to drift. Every *changed* or *new* line is written
   out in full. This is deliberate, not a placeholder.
2. **Commit after every task.** The tool keeps working at each commit, so a mistaken split is
   bisectable.

**Every command assumes the repo root as CWD.**

---

## Reuse: what comes from the repo instead of being hand-rolled

The exploration below was done by reading `util/`, `models/`, `datasets/` and `scripts/` directly.
Two rules came out of it: import what exists, and **do not** import what looks reusable but is not.

| Need | Source | Verdict |
| --- | --- | --- |
| IoU for NMS and association cost | `util.box_ops.box_iou` (`util/box_ops.py:55`) | **Use it.** `[N,4]`/`[M,4]` xyxy → `(N,M)` matrix + union, coordinate-agnostic. Replaces the hand-rolled scalar IoU and vectorizes the association cost matrix. NaN only when *both* boxes are degenerate — guard with `nan_to_num` |
| Box format conversions | `util.box_ops.box_cxcywh_to_xyxy` / `box_xyxy_to_cxcywh` (`:15`, `:22`) | Already used; keep |
| Chunked decode, metadata, per-frame val transform | `inference.py:435-535` | **Move verbatim.** Verified by running it: applying the val transform to one frame reproduces the whole-clip result bit-for-bit (`max abs diff 0.0`) and is stable across repeats, so per-frame application is equivalent — the design depends on this and it holds |
| Val transform construction per stage | `inference.py:1473-1486` | **Move.** `resolution` selects `max_size` only (800/1000/1333) in the val branch; `cautious`/`by_ratio`/`overflow_boxes` are train-only and inert here |
| Stage-2 pseudo-target key contract | `datasets/dataset.py:239-243` (`prepare_data_stage2`) | **Match it exactly.** `unscaled_{sub,obj}_boxes` are xyxy in **resized** pixels (`= box_cxcywh_to_xyxy(normalized) * [w,h,w,h]` using the post-resize `size`), never source pixels. `roi_align` uses a hardcoded `spatial_scale=1/32` (`models/transformer.py:60-61, 78-83`) |
| `memory_update` required keys | `models/vrdformer_stage2.py:44-78` | `sub_boxes` is **mandatory even though its values are never read** — only `shape[0]` is used for the pair count (`:44`). `so_track_ids` must be a tensor (`.cpu()` at `:51`), not a list of lists |
| `relation_classifier` return order | `models/vrdformer_stage2.py:116` | **sub, verb, obj** — verb is second, not third. Non-eval branch, which is the only usable one: `is_eval=True` requires ground-truth triplets |
| `VRDFormer_S2.forward` return | `models/vrdformer_stage2.py:194` | Returns **only** `memory` (a list of per-clip dicts) — not a tuple, not logits. The existing code is right; do not "fix" it |
| Stage-1 forward return | `models/vrdformer.py:285` | 5 values: `out, targets, features_all, memory, hs`. **`features_all` is the 3rd**, the full backbone list before the `[-3:]` slice, and is what `prev_features` wants. `memory` (4th) is unrelated |
| Logit → class decoding | `models/matcher.py:81-88` | **No reusable helper exists.** The rule lives only in the matcher: sigmoid if `focal_loss` else softmax (verb always sigmoid). Every config sets `focal_loss: true`, so the existing unconditional sigmoid is correct; keep the existing warning rather than adding a branch |
| Hungarian matching | `models/matcher.py:12` | **Do not reuse.** It wraps the *same* scipy solver (`:172`) but solves DETR query↔target cost matching on normalized boxes, returns unmatched-by-threshold pairs, and needs label tensors. Association stays on a direct `linear_sum_assignment` over `min(iou_sub, iou_obj)` |
| Tracking wrapper | `models/vrdformer_track.py:158-191` | Confirmed: **no `else` branch**, so `targets=None` returns `None`. It also only associates during *training*. Building base `VRDFormer` with `args.tracking=False` is correct and is what the existing code does |
| `NestedTensor` construction | `util.misc.NestedTensor.from_tensor_list` (`:234`) | Already used correctly. Do **not** use `select_frame` (`:214`) here — it indexes the clip axis and would silently return a 3-D tensor on a single image |
| Progress logging | `util.misc.MetricLogger` (`:19`) | **Decline.** `log_every` (`:57`) requires a sized iterable and reports `s/it` + ETA, but it is iterator-shaped and this pipeline yields frames generator-style. Reusing it would mean materialising the frame list, for formatting we already have. Keep the existing `print`-style rate logs |
| Pair NMS in `decode_frame` | — | **Keep hand-rolled.** `torchvision.ops.nms`/`batched_nms` compare one box list on plain IoU; the suppression here is `min(iou_sub, iou_obj)` within a `(sub_cls, obj_cls)` group, which they cannot express |
| Checkpoint load + `module.` strip | — | **Keep hand-rolled.** `util/checkpoints.py:param_initializer` (`:173`) is training-bound (needs `args.resume`/`pretrain`, mutates the optimizer and `args.start_epoch`) and strips `detr.`, never `module.`. Keep `weights_only=False`: checkpoints pickle an `argparse.Namespace` |
| Eval scoring | `util.evaluate.evaluate_visual_relation` (`:73`), `util.compute.viou` (`:78`) | **Not called by the pipeline** — it grades a whole split. The sidecar is written so the *user* can call it; see the coordinate caveat in Task 8 |
| Interpreter/device resolution in a launcher | `scripts/inference/inference.sh:33-56` | **Keep.** It is the only launcher in `scripts/` that resolves a venv and probes for a GPU; the training wrappers just call bare `python` with `torch.distributed.launch` (and `stage2/train_deform*.sh` reference a non-existent `epython`) |

**Not adding a test decorator here, but note for Task 4/6:** `VRDFormer.forward` has no internal
`no_grad` and asserts `batch_size < 10`; it must be called one frame at a time inside
`torch.no_grad()`, which the existing code does.

Two findings that are **defects, not reuse opportunities**, recorded in Task 8 and surfaced to the
user rather than silently fixed: the eval sidecar's coordinate space, and `viou`'s frame-id
semantics.

---

### Task 1: Package scaffold and console module

**Files:**
- Create: `vrdf_inference/__init__.py`
- Create: `vrdf_inference/console.py`
- Create: `scripts/inference/selfcheck.py`

- [ ] **Step 1: Create the package init**

Create `vrdf_inference/__init__.py`:

```python
"""Single-video inference for VRDFormer, split by responsibility.

``inference.py`` at the repository root is the entry point; every module here is importable on its
own. See ``docs/superpowers/specs/2026-10-18-inference-modularization-design.md``.
"""

__all__ = [
    "checkpoints",
    "cli",
    "console",
    "render",
    "results",
    "stage1",
    "stage2",
    "tracklets",
    "video",
]
```

- [ ] **Step 2: Create the console module**

Create `vrdf_inference/console.py` (moved verbatim from `inference.py:88-101`, with the prefix kept
identical so existing log output does not change):

```python
"""Logging for the inference pipeline.

One prefix on every line makes a run's own output greppable apart from PyTorch and decord noise.
"""

import sys

LOG = "[inference]"


def log(msg):
    print(f"{LOG} {msg}", flush=True)


def warn(msg):
    print(f"{LOG} WARNING: {msg}", flush=True)


def die(msg):
    print(f"{LOG} ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)
```

- [ ] **Step 3: Create the self-check script with one assertion**

Create `scripts/inference/selfcheck.py`:

```python
"""Pure-logic checks for the inference package.

No checkpoint, no GPU, no decord decode: every check here exercises a pure function on synthetic
input, so it runs in a second on any machine. This is the repo's stand-in for a unit-test suite
(there is no pytest here -- see CLAUDE.md).

Run: .venv/bin/python scripts/inference/selfcheck.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def console_prefix_is_stable():
    from vrdf_inference import console

    assert console.LOG == "[inference]", console.LOG


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
```

- [ ] **Step 4: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: `ok   console_prefix_is_stable` then `1/1 checks passed`, exit 0.

- [ ] **Step 5: Lint and commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py
ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/__init__.py vrdf_inference/console.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): add package scaffold and console module"
```

---

### Task 2: `video.py` — decord IO

**Files:**
- Create: `vrdf_inference/video.py`
- Test: `scripts/inference/selfcheck.py` (append)

- [ ] **Step 1: Move the IO functions**

Create `vrdf_inference/video.py`. Move these from `inference.py` **verbatim**, changing only the
`die`/`warn`/`log` calls to import from `console`:

| Function / constant | Source lines in `inference.py` at `HEAD` |
| --- | --- |
| `DECODER_HELP` | 435-440 |
| `open_video` | 443-454 |
| `iter_decoded_frames` | 457-488 |
| `video_meta` | 491-519 |
| `transform_frame` | 522-535 |

Add this header and imports at the top:

```python
"""Frame IO for inference: decord decode, video metadata, and the dataset's val transform.

Nothing here knows about models. The stride/`--max_frames` selection lives in
:func:`selected_frame_ids`, which both the decode loop and the metadata pass use so the frame axis
can never disagree between them.
"""

from pathlib import Path

import numpy as np  # noqa: F401  # kept for parity with the moved transform_frame body

from vrdf_inference.console import die, log, warn  # noqa: F401
```

- [ ] **Step 2: Extract the shared frame-id selection**

`iter_decoded_frames` and `video_meta` currently each build `ids` with identical logic. Add to
`video.py`, then call it from both (replacing their inline `ids = ...` blocks):

```python
def selected_frame_ids(total, stride, max_frames=None):
    """The source frame ids a run will visit, in order.

    Both the decode loop and the metadata pass call this, so the frame count in ``results.json``
    is by construction the number of frames the pipeline actually processed.
    """
    ids = list(range(0, total, max(1, int(stride))))
    if max_frames is not None:
        ids = ids[:max_frames]
    return ids
```

In `iter_decoded_frames`, the body becomes `ids = selected_frame_ids(total, stride, max_frames)`
followed by the existing empty-`ids` `die`. In `video_meta`, the same, so its
`"frame_ids": [int(i) for i in ids]` is unchanged.

- [ ] **Step 3: Append the self-check**

Append to `scripts/inference/selfcheck.py`:

```python
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
    from vrdf_inference.video import transform_frame
    from datasets.dataset import make_video_transforms

    transform = make_video_transforms("val", True, False, "small", overflow_boxes=False)
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    tensor, height, width = transform_frame(transform, frame)
    assert tensor.ndim == 3, tensor.shape
    assert tensor.shape[0] == 3, tensor.shape
    assert (height, width) == tuple(tensor.shape[1:]), (height, width, tensor.shape)
```

Note the `transform_frame` check needs `numpy` in the self-check's own scope — add `import numpy as np`
to the script's imports at the top.

- [ ] **Step 4: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all four checks `ok`, `4/4 checks passed`, exit 0.

- [ ] **Step 5: Commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py && ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/video.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): move frame IO into vrdf_inference.video"
```

---

### Task 3: `checkpoints.py` — load a checkpoint, build one stage

**Files:**
- Create: `vrdf_inference/checkpoints.py`

- [ ] **Step 1: Move the loading functions**

Create `vrdf_inference/checkpoints.py`. Move **verbatim** from `inference.py` at `HEAD`:

| Function | Source lines |
| --- | --- |
| `load_class_names` | 365-376 |
| `load_checkpoint` | 379-391 |
| `resolve_device` | 394-402 |
| `load_model` | 405-427 |

Imports at the top:

```python
"""Checkpoint loading, class lists, device selection, and building one stage's model."""

import argparse
from pathlib import Path

import torch

from vrdf_inference.console import die, log, warn  # noqa: F401
from models import build_model
```

`load_model`'s existing `from models import build_model` at call scope becomes the module-level
import above; delete the function-local import so there is one source of truth.

- [ ] **Step 2: Verify the moved module imports cleanly**

Run: `.venv/bin/python -c "from vrdf_inference import checkpoints; print(checkpoints.load_class_names)"`
Expected: `<function load_class_names at 0x...>` — no ImportError.

- [ ] **Step 3: Commit**

```bash
ruff check vrdf_inference/checkpoints.py && ruff format vrdf_inference/checkpoints.py
git add vrdf_inference/checkpoints.py
git commit -m "refactor(inference): move checkpoint and model loading into checkpoints module"
```

---

### Task 4: `stage1.py` — detection, with the two entry points

**Files:**
- Create: `vrdf_inference/stage1.py`
- Test: `scripts/inference/selfcheck.py` (append)

- [ ] **Step 1: Move the decoding helpers and reuse `util.box_ops.box_iou`**

Create `vrdf_inference/stage1.py`. Move **verbatim** from `inference.py` at `HEAD`:

| Function | Source lines |
| --- | --- |
| `decode_stage1_frame` | 553-646 |
| `report_stage1_scores` | 713-722 |

Rename `decode_stage1_frame` to `decode_frame`. Keep the docstrings.

**Delete `_iou_xyxy` (source lines 543-550) and import `box_iou` from `util` instead.** The repo
already has a pairwise IoU: `util/box_ops.py:55` takes `[N,4]`/`[M,4]` xyxy tensors and returns an
`(N,M)` matrix plus the union. It is coordinate-agnostic, so normalized and pixel boxes both work.
The two call sites in `decode_frame` use it as:

```python
def pair_iou(box_a, box_b):
    """IoU between two single xyxy boxes, as a float. Thin wrapper over util's pairwise box_iou."""
    return float(box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0))[0][0, 0])
```

**Gotcha to handle in that wrapper, verified by running it:** `box_iou` computes `inter / union`
with no zero-union guard, so it returns **NaN when both boxes are degenerate** (zero area, union 0).
The deleted `_iou_xyxy` returned `0.0` in that case. One degenerate box against a normal one is
fine (returns `0.0`), so only the both-degenerate case matters — but this pipeline derives boxes
from sigmoid outputs and an exactly-zero-area box is possible, so keep the old behaviour explicitly:

```python
import torch

def pair_iou(box_a, box_b):
    """IoU between two single xyxy boxes, as a float.

    Wraps util's pairwise ``box_iou``. It divides by the union without a guard, so two degenerate
    (zero-area) boxes give NaN where the old scalar helper gave 0.0; ``nan_to_num`` restores that.
    """
    iou = box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0))[0][0, 0]
    return float(torch.nan_to_num(iou))
```

**Do not replace the per-frame pair NMS.** It suppresses on `min(iou_sub, iou_obj)` within a
`(sub_cls, obj_cls)` group (source lines 631-648). `torchvision.ops.nms` and `batched_nms` compare
one box list on plain IoU, and `batched_nms` only groups by an index — neither can express the
two-box minimum, so using them would change which detections survive. Keep the hand-rolled greedy
loop.

Imports:

```python
"""Stage-1 detection: decode one frame's output, or drive the whole video through the detector."""

import time

import numpy as np
import torch

import util.misc as utils
from models.vrdformer import VRDFormer
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from vrdf_inference.console import log, warn
from vrdf_inference.video import iter_decoded_frames, transform_frame
```

- [ ] **Step 2: Add the per-frame entry point**

Add to `stage1.py` — this is the function both modes share:

```python
@torch.no_grad()
def detect_frame(model, frame_rgb, prev_features, transform, device, args, source_hw):
    """Run the detector on one decoded frame.

    Returns ``(detections, ceilings, features)``. ``features`` is this frame's backbone output, to
    be handed back as ``prev_features`` on the next call: Stage 1 was trained with
    ``multi_frame_attention``, so consecutive frames must be fed in order.
    """
    frame_tensor, height, width = transform_frame(transform, frame_rgb)
    samples = utils.NestedTensor.from_tensor_list([frame_tensor]).to(device)
    out, _, features, _, _ = VRDFormer.forward(
        model, samples, targets=None, prev_features=prev_features
    )
    detections, ceiling = decode_frame(out, (height, width), source_hw, args)
    return detections, ceiling, features
```

- [ ] **Step 3: Add the video-level driver**

Add to `stage1.py`, replacing the loop body of the old `run_stage1` (source lines 649-710) with
calls to `detect_frame`:

```python
@torch.no_grad()
def run_detection(model, video_path, transform, device, args, on_frame=None):
    """Detect on every selected frame, carrying the previous frame's backbone features.

    Returns ``(per_frame, resized_hw, source_hw)``. ``per_frame`` is a list of detection lists,
    one entry per decoded frame, in decode order.

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
        detections, ceiling, prev_features = detect_frame(
            model, frame_rgb, prev_features, transform, device, args, source_hw
        )
        resized_hw = (frame_rgb.shape[0], frame_rgb.shape[1])
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
```

Delete the old `run_stage1` (lines 649-710) — `run_detection` replaces it.

**Note:** `run_detection` derives `resized_hw` from the decoded frame's own `shape[:2]` before the
transform, which is the *source* size, not the resized size. That is wrong for the payload's
`resized_size`. Fix it by having `detect_frame` also return the transformed size:

```python
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
```

and in `run_detection`:

```python
        source_hw = frame_rgb.shape[:2]
        detections, ceiling, prev_features, resized_hw = detect_frame(
            model, frame_rgb, prev_features, transform, device, args, source_hw
        )
```

Remove the now-duplicate `resized_hw = (frame_rgb.shape[0], frame_rgb.shape[1])` line.

- [ ] **Step 4: Append the self-check for the pure decoding path**

`decode_frame` needs a model output dict, which is easy to fake with tensors. Append to
`scripts/inference/selfcheck.py`:

```python
@check
def decode_frame_thresholds_and_scales():
    import argparse

    import torch

    from vrdf_inference.stage1 import decode_frame

    num_queries, num_obj, num_verb = 4, 80, 50
    out = {
        # Query 0 clearly above threshold, query 1 clearly below, 2/3 borderline.
        "pred_sub_logits": [torch.zeros(num_queries, num_obj)],
        "pred_obj_logits": [torch.zeros(num_queries, num_obj)],
        "pred_verb_logits": [torch.zeros(num_queries, num_verb)],
        # cxcywh, normalized.
        "pred_sub_boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * num_queries)],
        "pred_obj_boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * num_queries)],
    }
    out["pred_sub_logits"][0][0, 5] = 4.0  # sigmoid ~0.98
    out["pred_obj_logits"][0][0, 7] = 4.0

    args = argparse.Namespace(
        conf_sub=0.5, conf_obj=0.5, conf_verb=0.5, dedup_iou=0.9, max_dets_per_frame=50
    )
    detections, ceilings = decode_frame(out, (200, 300), (100, 150), args)

    assert len(detections) == 1, f"expected only the confident query, got {len(detections)}"
    det = detections[0]
    assert det["sub_cls"] == 5, det["sub_cls"]
    assert det["obj_cls"] == 7, det["obj_cls"]
    # A 0.2-wide normalized box on a 300px-wide resized frame is 60px, then 150/300 on the source.
    assert abs(det["box_resized_sub"][2] - det["box_resized_sub"][0] - 60.0) < 1e-3, det["box_resized_sub"]
    assert abs(det["box_px_sub"][2] - det["box_px_sub"][0] - 30.0) < 1e-3, det["box_px_sub"]
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
```

- [ ] **Step 5: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all six checks `ok`, `6/6 checks passed`, exit 0.

- [ ] **Step 6: Commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py && ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/stage1.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): split stage-1 detection into detect_frame and run_detection"
```

---

### Task 5: `tracklets.py` — association (stage 2 only)

**Files:**
- Create: `vrdf_inference/tracklets.py`
- Test: `scripts/inference/selfcheck.py` (append)

- [ ] **Step 1: Move the association**

Create `vrdf_inference/tracklets.py`. Move **verbatim** from `inference.py` at `HEAD`:

| Function | Source lines |
| --- | --- |
| `form_tracklets` | 730-849 |
| `_weighted_vote` | 852-857 → rename to `weighted_vote` |

Imports:

```python
"""Association of per-frame detections into tracklets.

Used only by the full pipeline: this is the module `--mode stage1` never imports, because a
detection-only run has no relations to track.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_iou
from vrdf_inference.console import log, warn
```

Update the two call sites inside `form_tracklets` that use `_iou_xyxy` to `pair_iou` (imported from
`vrdf_inference.stage1`), and the one that uses `_weighted_vote` to `weighted_vote`.

**Vectorize the cost matrix.** The double `for` loop over `(pending, dets)` (source lines 762-767)
builds exactly two pairwise IoU matrices. Replace it with util's `box_iou`, which removes O(n·m)
Python-level calls per frame:

```python
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
            cost = torch.minimum(iou_sub, iou_obj)
            # Same zero-union guard as pair_iou: two degenerate boxes give NaN, and NaN in the
            # cost matrix would make linear_sum_assignment silently return nonsense.
            cost = torch.nan_to_num(cost).numpy().astype(np.float64)
            allowed = cost >= args.iou_thresh
```

This is the same `min(iou_sub, iou_obj)` the deleted loop computed, and matching still runs through
`linear_sum_assignment(-cost)` as before, so the associated pairs are identical.

- [ ] **Step 2: Append the self-check**

Append to `scripts/inference/selfcheck.py`:

```python
@check
def weighted_vote_prefers_confidence():
    from vrdf_inference.tracklets import weighted_vote

    # 1 appears twice but with tiny weight; 2 appears once with a large weight.
    assert weighted_vote([1, 1, 2], [0.01, 0.01, 5.0]) == 2
    assert weighted_vote([1, 1, 2], [5.0, 5.0, 0.01]) == 1
    # Ties break deterministically on the value, not on insertion order.
    assert weighted_vote([3, 1], [1.0, 1.0]) == 1


def _synthetic_detection(frame_shift, cls=5):
    """A detection with a subject box at x+frame_shift and a fixed object box."""
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


@check
def tracklets_join_boxes_that_overlap():
    import argparse

    from vrdf_inference.tracklets import form_tracklets

    args = argparse.Namespace(
        iou_thresh=0.3, max_gap=2, min_tracklet_frames=1, cls_gate="motion", cls_vote="keep"
    )
    # Frame 0 and frame 1 overlap heavily -> one tracklet spanning both frames.
    per_frame = [
        [_synthetic_detection(0.0)],
        [_synthetic_detection(0.02)],
    ]
    tracklets = form_tracklets(per_frame, args)
    assert len(tracklets) == 1, f"expected one joined tracklet, got {len(tracklets)}"
    assert tracklets[0]["frame_indices"] == [0, 1], tracklets[0]["frame_indices"]


@check
def tracklets_split_on_no_overlap():
    import argparse

    from vrdf_inference.tracklets import form_tracklets

    args = argparse.Namespace(
        iou_thresh=0.3, max_gap=2, min_tracklet_frames=1, cls_gate="motion", cls_vote="keep"
    )
    # Frame 1's subject box moved far away -> no overlap -> two tracklets.
    per_frame = [
        [_synthetic_detection(0.0)],
        [_synthetic_detection(0.6)],
    ]
    tracklets = form_tracklets(per_frame, args)
    assert len(tracklets) == 2, f"expected two tracklets, got {len(tracklets)}"


@check
def tracklets_bridge_a_missed_frame_within_max_gap():
    import argparse

    from vrdf_inference.tracklets import form_tracklets

    args = argparse.Namespace(
        iou_thresh=0.3, max_gap=2, min_tracklet_frames=1, cls_gate="motion", cls_vote="keep"
    )
    # A gap at frame 1, then an overlapping box at frame 2, is inside max_gap=2.
    per_frame = [
        [_synthetic_detection(0.0)],
        [],
        [_synthetic_detection(0.02)],
    ]
    tracklets = form_tracklets(per_frame, args)
    assert len(tracklets) == 1, f"the gap was not bridged: {len(tracklets)} tracklets"
    assert tracklets[0]["frame_indices"] == [0, 2], tracklets[0]["frame_indices"]
```

- [ ] **Step 4: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all `ok`, exit 0. If `tracklets_bridge_a_missed_frame_within_max_gap` fails, check the
`<=` in the pending-prune line (`frame_index - p["last_frame"] <= args.max_gap`) — removing the
equality makes `max_gap=2` behave like `max_gap=1`.

- [ ] **Step 4: Commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py && ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/tracklets.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): move tracklet association into tracklets module"
```

---

### Task 6: `stage2.py` — accumulation and classification

**Files:**
- Create: `vrdf_inference/stage2.py`

- [ ] **Step 1: Move the stage-2 functions**

Create `vrdf_inference/stage2.py`. Move **verbatim** from `inference.py` at `HEAD`:

| Function | Source lines |
| --- | --- |
| `build_pseudo_target` | 865-913 |
| `run_stage2` | 916-975 → rename to `classify_tracklets` |
| `build_predictions` | 978-1104 |
| `det_box` | 1107-1109 |

Imports:

```python
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
```

Inside `classify_tracklets` (the renamed `run_stage2`), the call is `transform_frame(transform,
frame_rgb)`; it keeps its `@torch.no_grad()` decorator.

- [ ] **Step 2: Verify the moved module imports cleanly**

Run: `.venv/bin/python -c "from vrdf_inference import stage2; print(stage2.build_predictions)"`
Expected: `<function build_predictions at 0x...>` — no ImportError.

- [ ] **Step 3: Commit**

```bash
ruff check vrdf_inference/stage2.py && ruff format vrdf_inference/stage2.py
git add vrdf_inference/stage2.py
git commit -m "refactor(inference): move stage-2 classification into stage2 module"
```

---

### Task 6b: Fix the cross-stage box-scale mismatch

**Files:**
- Modify: `vrdf_inference/stage2.py`
- Test: `scripts/inference/selfcheck.py` (append)

**Why this task exists.** `build_pseudo_target` builds `unscaled_{sub,obj}_boxes` from
`box_resized_sub`/`box_resized_obj`, which were computed under the **stage-1** transform. But stage 2
resizes to its own size and `roi_align` samples at `1/32` of *stage-2's* feature map. When the two
stages use different `resolution` values the boxes are wrong by the ratio of the two sizes, and
because `max_size` clamps the *short* side too, that ratio is not 1.

Measured on this repo's checkpoints (stage 1 = `large`, stage 2 = `small`), by running the val
transform through `make_video_transforms`:

| Source | stage 1 (`large`) | stage 2 (`small`) |
| --- | --- | --- |
| 320×240 | 800×1066 | 600×800 |
| 1280×720 | 750×1333 | 450×800 |

So stage-1 resized pixels are up to **33% too large per axis** for stage-2's `roi_align`. This is a
pre-existing bug, not something the split introduces — but the module split is the right place to
fix it, and it grows more visible once stage 2 runs at its checkpoint's own resolution.

- [ ] **Step 1: Return the transform's output size from the stage-2 path**

`classify_tracklets` currently discards the size:

```python
        frame_tensor, _height, _width = transform_frame(transform, frame_rgb)
```

Have `build_pseudo_target` take the current stage-2 frame size and scale the normalized boxes by it.
Replace the four box lines in `build_pseudo_target` with:

```python
def build_pseudo_target(active, frame_index, device, num_verb_classes, resized_hw):
    """Assemble the one pseudo-GT dict ``VRDFormer_S2.forward`` reads for a single frame.

    Keys and conventions follow ``datasets/dataset.py:prepare_data_stage2`` and
    ``models/transformer.py``:

    * ``unscaled_{sub,obj}_boxes`` are xyxy in **resized** pixels -- the input ``roi_align`` samples
      at ``spatial_scale=1/32`` (``models/transformer.py:60-61, 78-83``).
    * ``{sub,obj}_boxes`` are normalized cxcywh, which is what ``memory_update`` reads.

    ``resized_hw`` is *this stage's* transform output size, not stage 1's. Deriving the pixel boxes
    from the normalized ones here (rather than passing stage 1's pixel boxes through) is what keeps
    the two stages independent when ``--stage1_resolution`` and ``--stage2_resolution`` differ:
    ``max_size`` changes with the resolution, and it clamps the short side too, so stage 1's 800x1066
    and stage 2's 600x800 are not interchangeable.
    """
    height, width = resized_hw
    scale = torch.tensor([width, height, width, height], dtype=torch.float32, device=device)

    sub_norm = torch.as_tensor(
        np.stack([t["frames"][frame_index]["box_n_sub"] for t in active]),
        dtype=torch.float32,
    )
    obj_norm = torch.as_tensor(
        np.stack([t["frames"][frame_index]["box_n_obj"] for t in active]),
        dtype=torch.float32,
    )

    target = {
        "unscaled_sub_boxes": (box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(sub_norm)) * scale).to(device),
        "unscaled_obj_boxes": (box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(obj_norm)) * scale).to(device),
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
```

Note the double conversion: `sub_norm` is xyxy in [0,1], so `box_xyxy_to_cxcywh` then
`box_cxcywh_to_xyxy` is an identity on values — it is written that way to mirror
`datasets/dataset.py:239-243` exactly. Simplify to `sub_norm * scale` if you prefer; the result is
identical and the self-check below covers it either way.

Then in `classify_tracklets`:

```python
        frame_tensor, height, width = transform_frame(transform, frame_rgb)
        samples = utils.NestedTensor.from_tensor_list([frame_tensor]).to(device)
        target = build_pseudo_target(
            active, index, device, num_verb_classes, (height, width)
        )
```

Add `box_cxcywh_to_xyxy` to the imports in `stage2.py`.

- [ ] **Step 2: Record both stage sizes in the payload, so a mismatch is visible**

In `results.triplet_payload`, add the stage-2 input size next to the stage-1 `resized_size` that
`base_payload` already writes. In `cli._run_stage2`, capture it from the run and pass it through:

```python
    payload["stage2"]["input_size"] = [width, height]
```

where `(height, width)` is the size `classify_tracklets` used. This makes "stage 1 ran at 800×1066,
stage 2 at 600×800" visible in `results.json` instead of implicit.

- [ ] **Step 3: Append the self-check**

Append to `scripts/inference/selfcheck.py`:

```python
@check
def pseudo_target_uses_the_given_size():
    import torch

    from vrdf_inference.stage2 import build_pseudo_target

    def tracklet(sub_tid, sub_box_n):
        frame = {
            "box_n_sub": sub_box_n,
            "box_n_obj": [0.4, 0.4, 0.8, 0.8],
            "box_resized_sub": [0.0, 0.0, 1.0, 1.0],
            "box_resized_obj": [0.0, 0.0, 1.0, 1.0],
            "pair_score": 0.2,
        }
        return {
            "sub_tid": sub_tid,
            "obj_tid": sub_tid,
            "sub_cls": 5,
            "obj_cls": 7,
            "frames": {0: frame},
        }

    active = [tracklet(0, [0.25, 0.25, 0.75, 0.75])]
    # Half-width box on each axis: 0.5 * 600 = 300 wide, 0.5 * 800 = 400 tall.
    target = build_pseudo_target(active, 0, torch.device("cpu"), 50, (800, 600))
    box = target["unscaled_sub_boxes"][0]
    assert abs(float(box[2] - box[0]) - 300.0) < 1e-3, box
    assert abs(float(box[3] - box[1]) - 400.0) < 1e-3, box
    # The same normalized box at a different stage-2 size must scale differently: this is the bug
    # the task fixes, so a size-insensitive implementation fails here.
    other = build_pseudo_target(active, 0, torch.device("cpu"), 50, (600, 800))
    assert abs(float(other["unscaled_sub_boxes"][0][2] - other["unscaled_sub_boxes"][0][0]) - 400.0) < 1e-3
    # memory_update reads shape[0] of sub_boxes, so it must exist and match the row count.
    assert target["sub_boxes"].shape[0] == len(active), target["sub_boxes"].shape
    assert target["num_inst"] == len(active), target["num_inst"]
```

- [ ] **Step 4: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: check count grows by one, all `ok`, exit 0.

- [ ] **Step 5: Commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py && ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/stage2.py vrdf_inference/results.py vrdf_inference/cli.py scripts/inference/selfcheck.py
git commit -m "fix(inference): scale stage-2 pseudo-target boxes by stage 2's own input size"
```

---

### Task 7: `render.py` — two renderers, shared machinery

**Files:**
- Create: `vrdf_inference/render.py`
- Test: `scripts/inference/selfcheck.py` (append)

- [ ] **Step 1: Extract the shared machinery**

Create `vrdf_inference/render.py`. Move from `inference.py` at `HEAD`, renamed to drop the leading
underscore where they are now shared across two renderers:

| Item | Source lines | New name |
| --- | --- | --- |
| `_PALETTE` | 1163-1184 | `PALETTE` |
| `_put_label` | 1187-1213 | `put_label` |
| codec-fallback writer block | 1307-1325 (inside `render_video`) | `open_writer` |

Add `open_writer` as a function, lifting the loop out of `render_video`:

```python
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
```

Also lift the two size/anchor helpers both renderers need, so the label legibility rules stay in
one place:

```python
def label_style(source_h):
    """Text scale and thickness for a frame this tall.

    Legibility is about drawn pixels, not the frame's nominal size: at 320x240 a fixed 0.4 floor
    produced ~5px text, so the scale is anchored to the frame height instead.
    """
    return min(1.2, max(0.45, source_h / 400.0)), max(1, round(source_h / 320.0))


def draw_header(canvas, text, source_h, scale, thickness):
    """The one-line status strip along the bottom of every frame."""
    put_label(canvas, text, (8, source_h - 8), (255, 255, 255), scale * 0.9, thickness)
```

- [ ] **Step 2: Write the stage-1 renderer**

Add to `render.py`:

```python
def render_detections(video_path, out_path, per_frame, meta, args):
    """Annotate a detection-only run: subject and object boxes, class names and scores.

    Boxes are drawn for every kept detection. Subject and object boxes use two fixed colours so a
    pair reads as a pair; the label ``adult 0.19 / table 0.17`` sits at the pair's top-left corner
    and is only drawn for the ``--max_labels_per_frame`` most confident pairs, because on a small
    frame a label per detection is unreadable.

    Frames are re-decoded rather than cached, so the annotated video costs one more sequential
    read of the input instead of a full-resolution frame cache. Always called: the video is a
    required deliverable, including when ``per_frame`` holds no detections at all.
    """
    import cv2

    by_frame = {index: dets for index, dets in enumerate(per_frame)}
    source_w, source_h = meta["source_width"], meta["source_height"]
    scale, thickness = label_style(source_h)
    max_labels = max(1, args.max_labels_per_frame)

    writer = None
    written = 0
    for index, source_fid, frame_rgb in iter_decoded_frames(
        video_path, args.every_n_frames, args.max_frames, args.decode_chunk
    ):
        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections = by_frame.get(index, [])
        # Strongest pairs get the text label; every pair still gets its boxes.
        labelled = {
            id(det)
            for det in sorted(detections, key=lambda d: -d["pair_score"])[:max_labels]
        }

        for det in detections:
            tops = []
            for key, color in (
                ("box_px_sub", SUB_COLOR),
                ("box_px_obj", OBJ_COLOR),
            ):
                x1, y1, x2, y2 = [round(float(v)) for v in det[key]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                tops.append((y1, x1))
            if id(det) not in labelled:
                continue
            top_y, left_x = min(tops)
            label = (
                f"{meta_obj(det, 'sub', args)} {det['sub_score']:.2f} / "
                f"{meta_obj(det, 'obj', args)} {det['obj_score']:.2f}"
            )
            put_label(canvas, label, (left_x, top_y - 4), SUB_COLOR, scale, thickness)

        header = (
            f"{meta['video_id']}  frame {index}/{meta['num_decoded_frames'] - 1} "
            f"(source {source_fid})  {len(detections)} detection(s)"
        )
        draw_header(canvas, header, source_h, scale, thickness)

        frame_h, frame_w = canvas.shape[:2]
        if (frame_w, frame_h) != (source_w, source_h):
            # The writer's frame size is fixed at open time, and decord's reported size can
            # disagree with the decoded pixels (odd containers, rotation metadata). Resize rather
            # than write a skewed stream or have every frame silently dropped.
            if written == 0:
                warn(
                    f"decoded frames are {frame_w}x{frame_h} but the container reports "
                    f"{source_w}x{source_h}; resizing to the reported size"
                )
            canvas = cv2.resize(canvas, (source_w, source_h))

        if writer is None:
            fps = max(1.0, meta["fps"] / max(1, args.every_n_frames))
            writer = open_writer(out_path, fps, (source_w, source_h))
        writer.write(canvas)
        written += 1

    if writer is not None:
        writer.release()
    log(f"wrote {out_path} ({written} frames)")
```

Add the two colours and the small name helper near `PALETTE`:

```python
# Stage 1 has no tracklets, so a box's meaning comes from which side of the pair it is. BGR.
SUB_COLOR = (255, 190, 60)
OBJ_COLOR = (60, 190, 255)


def meta_obj(det, side, args):
    """Class name for one side of a detection, from the names list bound at render time."""
    names = args.obj_names
    return names[det[f"{side}_cls"]]
```

`args.obj_names` is set by `cli.main` before rendering (see Task 9) so the renderer does not need
the class list threaded through its signature.

- [ ] **Step 3: Write the stage-2 renderer**

Move `render_video` (source lines 1216-1331) into `render.py` **verbatim**, renamed to
`render_triplets`, with three mechanical changes:
1. Its `_PALETTE` reference becomes `PALETTE`, `_put_label` becomes `put_label`.
2. Its inline codec-fallback loop becomes `writer = open_writer(out_path, fps, (source_w, source_h))`.
3. Its `scale`/`thickness` computation becomes `scale, thickness = label_style(source_h)`, and the
   header `_put_label(...)` call becomes `draw_header(canvas, header, source_h, scale, thickness)`.

Delete the `if not args.no_video:` guard concept entirely — the function is always called.

- [ ] **Step 4: Append the self-check**

Append to `scripts/inference/selfcheck.py`:

```python
@check
def label_style_scales_with_frame_height():
    from vrdf_inference.render import label_style

    small_scale, small_thick = label_style(240)
    large_scale, large_thick = label_style(2160)
    assert small_scale == 0.6, small_scale  # 240/400 = 0.6, inside the 0.45..1.2 band
    assert large_scale == 1.2, large_scale  # clamped at the top
    assert small_thick == 1 and large_thick >= 1, (small_thick, large_thick)


@check
def both_renderers_write_a_playable_video():
    """End-to-end render check on a synthetic clip: both renderers honour 'always write'."""
    import argparse
    import tempfile
    from pathlib import Path

    import cv2

    from vrdf_inference.render import render_detections, render_triplets

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src = tmp_path / "in.mp4"
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
            action_names=["v"] * 50,
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
        out_det = tmp_path / "det.mp4"
        render_detections(str(src), out_det, [[det]] * 5, meta, args)
        assert out_det.exists() and out_det.stat().st_size > 0, "stage-1 video not written"

        # The zero-detection case must still produce a video.
        out_empty = tmp_path / "empty.mp4"
        render_detections(str(src), out_empty, [[]] * 5, meta, args)
        assert out_empty.exists() and out_empty.stat().st_size > 0, "empty-case video not written"

        for written in (out_det, out_empty):
            reader = cv2.VideoCapture(str(written))
            assert reader.isOpened(), f"{written} is not readable"
            count = 0
            while True:
                ok, _ = reader.read()
                if not ok:
                    break
                count += 1
            reader.release()
            assert count == 5, f"{written} has {count} frames, expected 5"
```

- [ ] **Step 5: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all twelve checks `ok`, `12/12 checks passed`, exit 0. The render check needs OpenCV only
— no torch, no checkpoint.

- [ ] **Step 6: Commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py && ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/render.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): split rendering into detection and triplet renderers"
```

---

### Task 8: `results.py` — JSON payloads

**Files:**
- Create: `vrdf_inference/results.py`
- Test: `scripts/inference/selfcheck.py` (append)

- [ ] **Step 1: Move the writers**

Create `vrdf_inference/results.py`. Move **verbatim** from `inference.py` at `HEAD`:

| Function | Source lines |
| --- | --- |
| `save_results` | 1117-1121 |
| `write_eval_prediction` | 1124-1159 |
| `summarise_detections` | 1334-1367 |

Imports:

```python
"""Result payloads written next to the annotated video.

Two payload shapes, one per mode: a detection-only run has no tracklets and no triplets, so it does
not carry those keys at all rather than carrying them empty.
"""

from vrdf_inference.console import log
```

- [ ] **Step 1b: Record the eval sidecar's coordinate space honestly**

`write_eval_prediction` emits `sub_traj`/`obj_traj` in **source pixels**, but the ground truth that
`util.evaluate.evaluate_visual_relation` compares against is **normalized [0,1]**. Both facts were
verified in the source:

- `datasets/dataset.py:103` sets `normalize_coords = True`, and `:290-296` divides every annotated
  bbox by `anno['width']`/`anno['height']` in place.
- `get_relation_insts` (`datasets/dataset.py:373-408`) reads those mutated boxes, so its
  `sub_traj`/`obj_traj` are normalized.
- `util.compute.viou` (`:78`) compares the two trajectories directly, with no rescaling.

So a sidecar compared against dataset GT would score ~0 on every trajectory for a reason that has
nothing to do with the model. **Do not silently rescale**: a rescale is only correct when the input
mp4 is the original full-rate source video, which is not knowable from inside the script. Instead,
record the situation in the payload and in the docstring:

Add to the `write_eval_prediction` docstring (moved code, so this is the one addition):

```
    Coordinate caveat: these trajectories are in source-pixel xyxy, whereas
    ``datasets/dataset.get_relation_insts`` returns ``[0,1]``-normalized boxes
    (``datasets/dataset.py:290-296``). ``util.compute.viou`` compares them without rescaling, so a
    sidecar scored against dataset ground truth needs one side rescaled by the source width/height
    first. That rescale is only valid when the input video is the original full-rate source.
```

And add a machine-readable note in the sidecar record, so a consumer does not have to read this
plan:

```python
            "traj_coord_space": "source_pixel_xyxy",
```

- [ ] **Step 2: Add the two payload builders**

Add to `results.py`, lifting the inline dict from `cli.main` (source lines 1526-1563):

```python
def base_payload(video_path, video_id, mode, meta, resized_hw, args, stage_args):
    """The metadata every run records, device-independent and mode-independent.

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
            meta["per_frame"], args.obj_names, args.action_names, args.source_hw, args
        ),
    }


def detection_payload(video_path, video_id, meta, resized_hw, args, stage_args):
    """Payload for ``--mode stage1``: metadata plus per-frame detections, nothing else."""
    return base_payload(video_path, video_id, "stage1", meta, resized_hw, args, stage_args)


def triplet_payload(
    video_path, video_id, meta, resized_hw, args, stage_args, stage2_args, tracklets, predictions
):
    """Payload for ``--mode stage2``: the detection payload plus tracklets and triplets."""
    payload = base_payload(video_path, video_id, "stage2", meta, resized_hw, args, stage_args)
    payload["stage2"] = {
        "checkpoint": args.stage2_ckpt,
        "config": args.stage2_config,
        "resolution": stage2_args.resolution,
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
```

- [ ] **Step 3: Append the self-check**

Append to `scripts/inference/selfcheck.py`:

```python
@check
def stage1_payload_omits_triplet_keys():
    import argparse

    from vrdf_inference import results

    args = argparse.Namespace(
        every_n_frames=5,
        conf_sub=0.15,
        conf_obj=0.15,
        conf_verb=0.15,
        dedup_iou=0.9,
        stage1_ckpt="ck1.pth",
        stage1_config="c1.json",
        stage2_ckpt="ck2.pth",
        stage2_config="c2.json",
        obj_names=["a"] * 80,
        action_names=["v"] * 50,
    )
    stage_args = argparse.Namespace(dataset="vidorsmall", resolution="large")
    meta = {
        "num_source_frames": 100,
        "num_decoded_frames": 20,
        "frame_ids": [0, 5, 10],
        "fps": 30.0,
        "source_width": 320,
        "source_height": 240,
        "per_frame": [[]],
    }
    payload = results.detection_payload(
        "v.mp4", "vid", meta, (200, 300), args, stage_args
    )
    assert payload["mode"] == "stage1", payload["mode"]
    assert "predictions" not in payload, sorted(payload)
    assert "num_tracklets" not in payload, sorted(payload)
    assert payload["num_detections"] == 0, payload["num_detections"]
    assert payload["source_size"] == [320, 240], payload["source_size"]
    assert "stage2" not in payload, sorted(payload)
```

- [ ] **Step 4: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all thirteen checks `ok`, exit 0.

- [ ] **Step 5: Commit**

```bash
ruff check vrdf_inference scripts/inference/selfcheck.py && ruff format vrdf_inference scripts/inference/selfcheck.py
git add vrdf_inference/results.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): move result payloads into results module"
```

---

### Task 9: `cli.py` and the `inference.py` entry point

**Files:**
- Create: `vrdf_inference/cli.py`
- Modify: `inference.py` (replace the whole file with a shim)

- [ ] **Step 1: Move the argument machinery and change the resolution precedence**

Create `vrdf_inference/cli.py`. Move **verbatim** from `inference.py` at `HEAD`:

| Item | Source lines |
| --- | --- |
| `ARCH_KEYS` | 111-129 |
| `build_arg_parser` | 132-304 |
| `resolve_stage_args` | 307-323 |
| `check_stage_args` | 326-357 |

**One change to `resolve_stage_args`.** Today the merge order is *argparse defaults < checkpoint args
< config JSON < CLI overrides*, so the config JSON wins. For `resolution` that is wrong on this
repo's checkpoints: `data/ckpts/vidorsmall_stage2` was trained at `"small"` while
`configs/vidorsmall_stage2.json` says `"large"`. Applying `large` loads fine but runs `roi_align` at
a scale the weights never saw. Replace the function with:

```python
# The checkpoint is the only record of the scale the weights actually saw, so for this one key it
# outranks the config JSON. Without that, every stage-2 run needs a remembered
# --stage2_resolution small or it silently predicts at the wrong scale: the checkpoint in
# data/ckpts/vidorsmall_stage2 was trained at "small", but configs/vidorsmall_stage2.json says
# "large". A CLI override still wins, so overriding stays possible.
CKPT_WINS_KEYS = ("resolution",)


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
    have_ckpt_key = ckpt_args is not None
    for key, value in config.items():
        # A CKPT_WINS_KEYS key the checkpoint actually stores keeps the checkpoint's value; every
        # other key, and any ckpt-wins key the checkpoint does not store, takes the config's.
        if key in CKPT_WINS_KEYS and have_ckpt_key and key in ckpt_args:
            continue
        merged[key] = value
    merged.update({k: v for k, v in overrides.items() if v is not None})
    merged.pop("dataset_config", None)
    return argparse.Namespace(**merged)
```

**Leave `check_stage_args` as it is.** Its resolution branch now fires only when a CLI override
contradicts the checkpoint, which is exactly when it is worth saying.

Imports:

```python
"""Command line, config resolution, and mode dispatch for single-video inference."""

import argparse
import json
import random
import sys
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
```

Import `tracklets`, `stage2`, and `model_s2`'s builder **lazily inside the stage-2 branch** (see
Step 3) so `--mode stage1` never imports them.

- [ ] **Step 2: Change the parser: required `--mode`, no `--no_video`**

In the moved `build_arg_parser`, make these edits:

1. Add, immediately after the `--video` argument:

```python
    parser.add_argument(
        "--mode",
        required=True,
        choices=("stage1", "stage2"),
        help="stage1: detect and annotate only. stage2: detect, associate, classify, and annotate "
        "with relation triplets. Required: there is no default, so a cheap detection-only run and "
        "an expensive full run can never be confused for one another.",
    )
```

2. **Delete** the `--no_video` argument (source lines 270-272).

3. Replace the `--output_dir` help text (source lines 165-170) with:

```python
    parser.add_argument(
        "--output_dir",
        default=None,
        help="the per-mode output directory; results go to <output_dir>/<video_id>/ "
        "(default: output/inference/<mode>)",
    )
```

4. Add `--mode`-independent rendering control that already exists: keep
   `--max_labels_per_frame` as is.

- [ ] **Step 3: Write `main` with mode dispatch**

Add to `cli.py`, adapting `inference.py`'s `main` (source lines 1375-1583). Stage 1 stops after
detection; stage 2 continues through association and classification:

```python
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
    # Bound here so the renderers and the payload builder can read them without threading the
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
    passes = 2 if args.mode == "stage1" else 3
    log(
        f"workload: {meta['num_decoded_frames']} frames x {passes - 1} model pass(es) + 1 render "
        f"pass = ~{passes} video passes"
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

    _run_stage2(args, meta, per_frame, resized_hw, source_hw, video_out_dir, device, s1_args)
```

Then add the stage-2 half as its own function, so the stage-2-only imports are clearly scoped:

```python
def _run_stage2(args, meta, per_frame, resized_hw, source_hw, video_out_dir, device, s1_args):
    """Association, classification, and triplet output. Only reachable from --mode stage2."""
    from vrdf_inference import tracklets as tracklets_mod
    from vrdf_inference import stage2 as stage2_mod

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
        warn("focal_loss is false for stage 2; class decoding assumes the sigmoid-focal heads")
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

    stage2_out = (
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
        else None
    )
    predictions = (
        stage2_mod.build_predictions(
            stage2_out, tracklet_list, args.obj_names, args.action_names, args, meta, model_s2
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
        tracklet_list,
        predictions,
    )
    results.save_results(video_out_dir / args.results_name, payload)
    if args.eval_prediction_name:
        results.write_eval_prediction(
            video_out_dir / args.eval_prediction_name, predictions, video_id
        )

    render_triplets(args.video, video_out_dir / args.video_name, predictions, meta, args)

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
```

- [ ] **Step 4: Replace `inference.py` with the entry point**

Write `inference.py` at the repo root:

```python
"""Single-video inference for VRDFormer. Entry point; the pipeline lives in ``vrdf_inference``.

    python inference.py --video clip.mp4 --mode stage1   # detection + annotated video
    python inference.py --video clip.mp4 --mode stage2   # + tracklets, triplets, annotations

Both modes always write an annotated ``video.mp4`` and a ``results.json`` under
``output/inference/<mode>/<video_id>/``. See ``docs/RUNPOD_INFERENCE.md`` for the guides per mode
and ``docs/superpowers/specs/2026-10-18-inference-modularization-design.md`` for the design.
"""

import sys
from pathlib import Path

# The package imports the repository's own top-level modules (models, util, datasets, main), which
# only resolve with the repo root on sys.path. Adding it here means inference.py works regardless
# of the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vrdf_inference.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify the CLI surface**

Run: `.venv/bin/python inference.py --help`
Expected: usage line, and `--mode {stage1,stage2}` listed as required.

Run: `.venv/bin/python inference.py --video /dev/null`
Expected: exit code 2 with `error: the following arguments are required: --mode`.

Run: `.venv/bin/python inference.py --help | grep -c no_video`
Expected: `0` — the flag is gone.

- [ ] **Step 6: Append the self-check for the resolution precedence**

Append to `scripts/inference/selfcheck.py`:

```python
@check
def checkpoint_wins_on_resolution():
    import argparse
    import json
    import tempfile
    from pathlib import Path

    from vrdf_inference.cli import resolve_stage_args

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Path(tmpdir) / "c.json"
        config.write_text(json.dumps({"resolution": "large", "num_queries": 100}))

        base = argparse.Namespace(resolution="large", num_queries=100, seed=42)
        ckpt = {"resolution": "small", "num_queries": 200}

        # The checkpoint outranks the config for resolution...
        resolved = resolve_stage_args(base, str(config), ckpt, {})
        assert resolved.resolution == "small", resolved.resolution
        # ...and still wins for keys the config does not mention.
        assert resolved.num_queries == 200, resolved.num_queries
        # Other keys the config sets keep config-wins.
        ckpt_num_queries = {"resolution": "small", "num_queries": 100}
        resolved = resolve_stage_args(base, str(config), ckpt_num_queries, {})
        assert resolved.resolution == "small", resolved.resolution
        assert resolved.num_queries == 100, resolved.num_queries

        # A checkpoint that does not store resolution leaves the config's value alone.
        resolved = resolve_stage_args(base, str(config), {"num_queries": 200}, {})
        assert resolved.resolution == "large", resolved.resolution

        # An explicit CLI override beats the checkpoint.
        resolved = resolve_stage_args(
            base, str(config), ckpt, {"resolution": "large", "dataset": None}
        )
        assert resolved.resolution == "large", resolved.resolution
```

- [ ] **Step 7: Run the self-check**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all fourteen checks `ok`, exit 0. This one imports `cli`, so it
also proves the CLI module imports cleanly on CPU.

- [ ] **Step 8: Verify the stage-1 import boundary**

Run:
```bash
.venv/bin/python -c "
import subprocess, sys
code = 'import sys; sys.path.insert(0, \".\"); from vrdf_inference.cli import main; print(\"stage2\" in sys.modules, \"tracklets\" in sys.modules)'
print(subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout)
"
```
Expected: `False False` — importing the CLI must not pull in the stage-2 modules.

- [ ] **Step 9: Commit**

```bash
ruff check vrdf_inference inference.py scripts/inference/selfcheck.py && ruff format vrdf_inference inference.py scripts/inference/selfcheck.py
git add vrdf_inference/cli.py inference.py scripts/inference/selfcheck.py
git commit -m "refactor(inference): add mode dispatch and reduce inference.py to an entry point"
```

---

### Task 10: Update the shell launcher

**Files:**
- Modify: `scripts/inference/inference.sh`

- [ ] **Step 1: Rewrite the launcher for `<mode> <video>`**

Replace `scripts/inference/inference.sh` entirely with:

```bash
#!/usr/bin/env bash
# VRDFormer single-video inference.
#
#   sh scripts/inference/inference.sh stage1 clip.mp4
#   sh scripts/inference/inference.sh stage2 clip.mp4 --every_n_frames 5 --max_frames 120
#
# stage1: detect and annotate only. stage2: detect, associate, classify into relation triplets,
# and annotate. Both write an annotated mp4 plus results.json under
# output/inference/<mode>/<video_id>/.
#
# Every flag after the two positional arguments is passed straight through to inference.py, so the
# full option list in `python inference.py --help` is available. Later argparse flags win, so a
# pass-through flag overrides anything this script sets first.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: sh scripts/inference/inference.sh <stage1|stage2> <video.mp4> [inference.py flags...]" >&2
    exit 2
fi

MODE=$1
VIDEO=$2
shift 2

case "$MODE" in
    stage1|stage2) ;;
    *)
        echo "[inference.sh] mode must be stage1 or stage2 (got '$MODE')" >&2
        exit 2
        ;;
esac

# Pick the interpreter. An activated venv (VIRTUAL_ENV) always wins, since that is the documented
# way to run inference; otherwise fall back to the project's .venv so the script works without
# `source .venv/bin/activate` first. A bare `python` is the last resort and is checked below,
# because it is usually the system interpreter without numpy/torch.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PY=python
elif [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python
fi

if ! "$PY" -c "import numpy, torch" >/dev/null 2>&1; then
    echo "[inference.sh] $PY cannot import numpy/torch." >&2
    echo "[inference.sh] Create the environment and activate it first:" >&2
    echo "[inference.sh]   bash scripts/runpod/install_stage1_deps.sh && source .venv/bin/activate" >&2
    exit 1
fi

# Pick the device rather than defaulting to CUDA: an explicit --device cuda on a CPU-only box
# stops with an error, which is a confusing first run on a workstation. An explicit --device
# from the caller still wins.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE=cuda
else
    DEVICE=cpu
    echo "[inference.sh] no usable GPU found; falling back to --device cpu (slow)" >&2
fi

"$PY" inference.py \
    --video "$VIDEO" \
    --mode "$MODE" \
    --device "$DEVICE" \
    "$@"
```

Note what is **not** here any more: the hardcoded `--stage2_resolution small`. It dragged a
stage-2-specific setting into every call, which was harmless when there was only one pipeline but
is wrong now that `--mode stage1` exists. Pass it explicitly when needed:
`sh scripts/inference/inference.sh stage2 clip.mp4 --stage2_resolution small`.

- [ ] **Step 2: Verify the script**

Run: `bash -n scripts/inference/inference.sh`
Expected: no output, exit 0.

Run: `sh scripts/inference/inference.sh`
Expected: the usage line on stderr, exit 2.

Run: `sh scripts/inference/inference.sh stage3 clip.mp4`
Expected: `mode must be stage1 or stage2 (got 'stage3')`, exit 2.

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/inference.sh
git commit -m "refactor(inference): launcher takes a mode and stops forcing stage2 resolution"
```

---

### Task 11: Documentation

**Files:**
- Modify: `docs/RUNPOD_INFERENCE.md`
- Modify: `docs/superpowers/specs/2026-10-18-inference-modularization-design.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the inference guide**

In `docs/RUNPOD_INFERENCE.md`:

1. Update the status note at the top to say the implementation is now split into `inference.py`
   (entry point) plus `vrdf_inference/`, and that there are two modes.
2. Replace the "Running it" section's command examples with the two mode commands:

```bash
source .venv/bin/activate
sh scripts/inference/inference.sh stage1 data/vidorsmall/videos/10042935613.mp4
sh scripts/inference/inference.sh stage2 data/vidorsmall/videos/10042935613.mp4
```

3. Replace the "Output" section's tree with:

```
output/inference/
  stage1/<video_id>/
    video.mp4            detections annotated (subject/object boxes + class labels)
    results.json         metadata + per-frame detections
  stage2/<video_id>/
    video.mp4            triplets annotated (one colour per tracklet)
    results.json         metadata + tracklets + predictions + per-frame detections
    eval_prediction.json util/evaluate-shaped sidecar
```

4. State plainly that **the annotated video is written by both modes, always**, including when
   nothing was detected, and that `--no_video` no longer exists.
5. Add a "CPU vs GPU" subsection recording the two-stage verification from the spec: CPU smoke
   tests at `--max_frames 8`, the full GPU run on RunPod, what must match (`num_source_frames`,
   `stride`, `source_size`) and what may not (`num_frames`, box counts at the threshold boundary).

- [ ] **Step 2: Reconcile the spec with what was built**

In `docs/superpowers/specs/2026-10-18-inference-modularization-design.md`:
1. Under Output, replace the `--output_root` sentence with the single-flag reality:
   `--output_dir` names the per-mode directory (default `output/inference/<mode>`), and
   `--output_root` does not exist.
2. Confirm the module list matches the built tree, and note the two renames
   (`decode_stage1_frame` → `decode_frame`, `run_stage2` → `classify_tracklets`).

- [ ] **Step 3: Note the new entry points in CLAUDE.md**

In `CLAUDE.md`, add a short section after "Commands" recording that single-video inference is
`inference.py --video <mp4> --mode stage1|stage2` (or the launcher), that the pipeline lives in
`vrdf_inference/`, and that the annotated video is a required deliverable of both modes. Keep it to
a few lines — this is a pointer, not a tutorial.

- [ ] **Step 4: Verify the docs match the code**

Run: `.venv/bin/python inference.py --help | head -20`
Cross-check every flag named in the updated `docs/RUNPOD_INFERENCE.md` against this output. Any
flag in the docs that is not in `--help` (or vice versa for the documented ones) is a docs bug.

- [ ] **Step 5: Commit**

```bash
git add docs/RUNPOD_INFERENCE.md docs/superpowers/specs/2026-10-18-inference-modularization-design.md CLAUDE.md
git commit -m "docs: update inference guide for the two-mode package layout"
```

---

### Task 12: Stage A verification — CPU smoke tests

**Files:** none created; this task produces evidence.

- [ ] **Step 1: Lint the whole tree**

Run: `ruff check . && ruff format --check .`
Expected: `All checks passed!` and `N files already formatted`.

- [ ] **Step 2: Run the self-check suite**

Run: `.venv/bin/python scripts/inference/selfcheck.py`
Expected: all `ok`, exit 0.

- [ ] **Step 3: Stage-1 smoke run**

Run:
```bash
.venv/bin/python inference.py \
    --video data/vidorsmall/videos/10042935613.mp4 \
    --mode stage1 --max_frames 8 --device cpu
```
Expected: log ends with `stage 1: best score before thresholding -- subject ...`, then the two
output paths. Takes a few minutes on CPU (~2.6 s/frame).

- [ ] **Step 4: Assert the stage-1 artefacts**

Run:
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import decord
d = Path("output/inference/stage1/10042935613")
payload = json.loads((d / "results.json").read_text())
print("num_detections:", payload["num_detections"], "frames:", len(payload["detections"]["frames"]))
assert payload["num_detections"] > 0, "no detections; lower --conf_sub/--conf_obj"
assert len(payload["detections"]["frames"]) == 8, len(payload["detections"]["frames"])
assert "predictions" not in payload, "stage-1 payload must not carry triplets"
vr = decord.VideoReader(str(d / "video.mp4"))
print("video frames:", len(vr))
assert len(vr) == payload["num_frames"], (len(vr), payload["num_frames"])
print("stage 1 artefacts OK")
PY
```
Expected: `stage 1 artefacts OK`. If `num_detections` is 0, the run printed the ceiling to lower
the thresholds to — re-run with those values; a 0 there is a threshold problem, not a code failure,
but the *video* assertion must pass either way.

- [ ] **Step 5: Stage 1 is genuinely stage-1 only**

Run:
```bash
.venv/bin/python inference.py \
    --video data/vidorsmall/videos/10042935613.mp4 \
    --mode stage1 --max_frames 2 --device cpu \
    --stage2_ckpt /nonexistent.pth --output_dir /tmp/inf_stage1_only
```
Expected: completes successfully and writes `/tmp/inf_stage1_only/10042935613/video.mp4`. A
`checkpoint not found: /nonexistent.pth` error means stage-1 mode is still touching the stage-2
checkpoint.

- [ ] **Step 6: Zero-detection still writes a video**

Run:
```bash
.venv/bin/python inference.py \
    --video data/vidorsmall/videos/10042935613.mp4 \
    --mode stage1 --max_frames 3 --device cpu \
    --conf_sub 0.99 --conf_obj 0.99 --output_dir /tmp/inf_empty
```
Then assert:
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import decord
d = Path("/tmp/inf_empty/10042935613")
payload = json.loads((d / "results.json").read_text())
assert payload["num_detections"] == 0, payload["num_detections"]
vr = decord.VideoReader(str(d / "video.mp4"))
print("frames:", len(vr))
assert len(vr) == payload["num_frames"], (len(vr), payload["num_frames"])
print("zero-detection video OK")
PY
```
Expected: `zero-detection video OK` — this is the check that the "always write the video" rule holds
on the empty path.

- [ ] **Step 7: Stage-2 smoke run**

Run:
```bash
.venv/bin/python inference.py \
    --video data/vidorsmall/videos/10042935613.mp4 \
    --mode stage2 --max_frames 8 --device cpu
```
Expected: stage-1 log, then `tracklets: ... kept of ... built`, then `triplets: N kept at
--min_score 0.05`. **No `--stage2_resolution` flag**: the checkpoint's saved `small` now outranks
the config's `large` (Task 9 Step 1), so resolution resolution happens automatically and no mismatch
warning should appear. If a warning about resolution does appear, the precedence change did not take
effect — that is the bug to fix before going further.

- [ ] **Step 8: Assert the stage-2 artefacts**

Run:
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import decord
d = Path("output/inference/stage2/10042935613")
for name in ("video.mp4", "results.json", "eval_prediction.json"):
    assert (d / name).exists(), f"missing {name}"
payload = json.loads((d / "results.json").read_text())
print("tracklets:", payload["num_tracklets"], "predictions:", payload["num_predictions"])
assert payload["mode"] == "stage2", payload["mode"]
assert "predictions" in payload and "num_tracklets" in payload, sorted(payload)
vr = decord.VideoReader(str(d / "video.mp4"))
print("video frames:", len(vr))
assert len(vr) == payload["num_frames"], (len(vr), payload["num_frames"])
print("stage 2 artefacts OK")
PY
```
Expected: `stage 2 artefacts OK`.

- [ ] **Step 9: Parity spot-check against the pre-rewrite script**

The old single-file script is in git at the commit that preceded this work. Run it with the same
flags and compare counts:

```bash
mkdir -p /tmp/parity && git show HEAD~12:inference.py > /tmp/parity/old_inference.py
.venv/bin/python /tmp/parity/old_inference.py \
    --video data/vidorsmall/videos/10042935613.mp4 --max_frames 8 --device cpu \
    --stage2_resolution small --output_dir /tmp/parity/out
```

**The old script needs `--stage2_resolution small` and the new one does not.** The old resolution
order let the config's `large` win, so it only ran at `small` because the flag forced it. The new
order takes the checkpoint's `small` by default. Passing the flag to the old script is what makes
the two runs comparable — without it the old script runs at a different scale and the comparison
would be measuring the precedence change, not the code split. The new run in Step 7 deliberately
passed no flag.

Then compare:
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
old = json.loads(Path("/tmp/parity/out/10042935613/results.json").read_text())
new = json.loads(Path("output/inference/stage2/10042935613/results.json").read_text())
print("old resolution", old["stage2"]["resolution"], "-> new", new["stage2"]["resolution"])
print("num_detections", old["num_detections"], "->", new["num_detections"])
print("num_tracklets ", old["num_tracklets"], "->", new["num_tracklets"])
print("num_predictions", old["num_predictions"], "->", new["num_predictions"])
assert old["stage2"]["resolution"] == new["stage2"]["resolution"] == "small", (
    old["stage2"]["resolution"], new["stage2"]["resolution"]
)
# Stage 1 is untouched by the split, so these must match exactly. A difference here IS a bug.
assert old["num_detections"] == new["num_detections"], "detection count changed in the split"
assert old["num_tracklets"] == new["num_tracklets"], "tracklet count changed in the split"
print("stage-1 parity OK")
print("stage-2 box-scale fix: predictions", old["num_predictions"], "->", new["num_predictions"])
PY
```

**What must match and what must not.** This is the one place where "identical numbers" is the wrong
acceptance criterion, so be precise:

- `num_detections` and `num_tracklets` **must be identical**. Both are stage-1 geometry, which the
  split does not touch. A difference is a bug in the split — find the cause, do not adjust numbers.
- `num_predictions` **may legitimately differ**, because Task 6b fixes the cross-stage box scale. The
  old script fed stage-1's 800×1066 resized boxes to a stage-2 `roi_align` running at 600×800 (33%
  too large per axis), sampling the wrong image regions; the new code scales to stage 2's own size.
  Different `roi_align` inputs give different relation logits, so a different number of triplets
  clears `--min_score`. That is the fix working, not a regression.
- Sanity-check the direction rather than the exact value: the new `num_predictions` should be in the
  same order of magnitude, and the top triplets should be drawn from the same few verbs. A collapse
  to 0, or a jump to *every* tracklet, means the scale fix is wrong (e.g. boxes off by 800 instead of
  the correct ratio) — investigate before accepting.

To confirm the fix is what moved the number, re-run the new code forcing the old geometry:
`--stage2_resolution large` makes both stages resize identically, and `num_predictions` should then
land back on the old script's value. That equivalence is the strongest available evidence that the
scale change is the cause and nothing else is.

**Reading the parity ref:** `HEAD~12` assumes the plan's commit count (Tasks 1-11 plus Task 6b, with
Task 12 committing only if it needed fixes). Confirm before running:

```bash
git log --oneline -14 | tail -4      # the last line shown should be the pre-rewrite commit
git show HEAD~12:inference.py | wc -l   # expect 1587 for the pre-rewrite file
```

If `git show HEAD~12:inference.py` fails or the line count is not 1587, adjust the ref until it
does. Do not skip this check: running the *new* file as "the old one" would make the comparison
compare an implementation with itself and pass vacuously.

- [ ] **Step 10: Record the Stage A evidence**

Post the actual numbers from Steps 3-9 (not a claim that they passed) to whoever asked for the work:
`num_detections`, `num_tracklets`, `num_predictions`, video frame counts, and the parity line. Stage
B (RunPod) is the user's to run later; these CPU figures are its comparison baseline.

- [ ] **Step 11: Commit any fixes**

If earlier steps needed code changes, commit them now:

```bash
git add -A
git commit -m "fix(inference): address issues found in the CPU smoke run"
```

If nothing changed, skip this step — do not create an empty commit.

---

## Self-review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| `--mode stage1` = detection only, no stage-2 checkpoint needed | 9 (Step 3 + Step 6), 12 (Step 5) |
| `--mode stage2` = full pipeline | 9, 12 (Steps 7-8) |
| Both modes always write the annotated `video.mp4` | 7 (Step 2 note + Step 4 empty-case check), 12 (Steps 6, 8) |
| `--no_video` removed | 9 (Step 2.2), 10 (launcher note) |
| `output/inference/stage1|stage2/<video_id>/` | 9 (Step 3) |
| Package of single-purpose modules | 1-9 |
| Module stated purpose / dependencies | Each module's docstring, Tasks 1-8 |
| Stage-2 behavioural parity | 12 (Step 9) |
| Zero-detection still renders | 7 (Step 4), 12 (Step 6) |
| Launcher takes `<mode> <video>` | 10 |
| CPU smoke tests (Stage A) | 12 |
| GPU run documented for RunPod (Stage B) | 11 (Step 1.5) |
| No new dependency, no pytest suite invented | "Testing approach" section |

**Placeholder scan:** every step names exact file paths, exact commands and expected output. Moved
code is referenced by line range with the reason stated up front, and every changed or new line is
written out in full. No "TBD", no "handle edge cases", no "similar to Task N".

**Type consistency:** `decode_frame` (was `decode_stage1_frame`) is used consistently in Tasks 4, 5
(`iou_xyxy` import) and 9. `classify_tracklets` (was `run_stage2`) is consistent in Tasks 6 and 9.
`weighted_vote` (was `_weighted_vote`) in Tasks 5. `label_style`/`draw_header`/`put_label`/
`open_writer`/`PALETTE`/`SUB_COLOR`/`OBJ_COLOR` are defined in Task 7 and consumed in Task 7 only.
`results.detection_payload` / `triplet_payload` / `save_results` / `write_eval_prediction` are
defined in Task 8 and called only in Task 9. `args.obj_names` / `args.action_names` are set in
Task 9 Step 3 and read in Tasks 7 and 8. `CKPT_WINS_KEYS` / `resolve_stage_args` are defined and
checked in Task 9, and relied on by Task 12 Step 7 (no flag) and Step 9 (flag for the old script).

**Deviations from a pure move (called out so they are not mistaken for regressions):**

| Deviation | Where | Why |
| --- | --- | --- |
| `resolution` precedence: checkpoint > config | Task 9 Step 1 | `data/ckpts/vidorsmall_stage2` trained at `small`; `configs/vidorsmall_stage2.json` says `large` |
| `--no_video` removed | Task 9 Step 2 | The annotated video is a required deliverable of both modes |
| `--mode` required | Task 9 Step 2 | A cheap run and an expensive run must not be confusable |
| launcher drops hardcoded `--stage2_resolution small` | Task 10 | Stage-2-specific; the precedence change makes it unnecessary |
| `decode_stage1_frame` → `decode_frame`, `_iou_xyxy` → `iou_xyxy` | Task 4 | Used by two modules now, so no longer private |
| `_weighted_vote` → `weighted_vote` | Task 5 | Same reason |
| `run_stage2` → `classify_tracklets` | Task 6 | The old name collides with the removed top-level driver |
