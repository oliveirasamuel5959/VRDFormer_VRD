# Single-video inference: modular rewrite

**Date:** 2026-10-18
**Status:** design approved, not yet implemented

## Problem

`inference.py` is a single 1587-line file that always runs the full
`video → Stage 1 → association → Stage 2 → triplets` pipeline. Two consequences:

1. **There is no detection-only path.** Inspecting what the Stage-1 detector sees means
   running the expensive Stage-2 pass as well, or reading `results.json` and ignoring the
   triplet half.
2. **One file holds four unrelated jobs** — decord IO, Stage-1 decoding, Hungarian tracklet
   association, Stage-2 accumulation, rendering, and argument/config resolution. Nothing can be
   read, tested, or changed in isolation.

This design splits the file into a package of single-purpose modules and adds a `--mode` flag
so detection-only and full-pipeline runs are two obvious commands writing to two obvious places.

## Goals

- `--mode stage1` runs detection only: decode → detect → annotate → `results.json`. It must not
  import Stage 2, run the association, or require a Stage-2 checkpoint to exist.
- `--mode stage2` runs the full pipeline and writes triplet-annotated output.
- **Both modes always write the annotated `video.mp4`.** It is a primary deliverable, not an
  optional side effect: a run that produced only `results.json` has failed. This holds at zero
  detections too — the video is still written, with the header and no boxes, because an empty
  annotated video is itself the diagnosis.
- Output lands at `output/inference/stage1/<video_id>/` and `output/inference/stage2/<video_id>/`.
- Each module is readable on its own and states what it does, how to call it, and what it depends on.
- **Behavioural parity with today's script on the stage-1 path**, and parity on the stage-2 path
  *except* the deliberate box-scale fix below. The thresholds, the `prev_features` carry, the
  calibration warnings and the mp4 codec fallback were measured and debugged; the rewrite must not
  quietly change them.

## Known defects this work surfaces (and how it handles them)

Exploration of `util/`, `models/`, `datasets/` and `scripts/` turned up three pre-existing issues
that touch the inference path. They are recorded here because the split makes them visible and two
of them change output.

1. **Cross-stage box scale (fixed here).** `build_pseudo_target` built `unscaled_{sub,obj}_boxes`
   from stage-1's resized-pixel boxes, but stage 2 resizes to its own size and `roi_align` samples at
   `1/32` of *stage-2's* feature map. Because `max_size` clamps the short side too, stage 1 at `large`
   and stage 2 at `small` differ by up to 33% per axis (measured: 800×1066 vs 600×800 for a 320×240
   source). Stage 2 was therefore sampling the wrong image regions. The rewrite scales the pixel boxes
   by **stage 2's own** transform output size. This legitimately changes `num_predictions`; see
   Verification step 9 for how to confirm the change is the fix and not a regression.

2. **Eval sidecar coordinate space (documented, not silently changed).** The sidecar writes
   trajectories in source pixels; `datasets/dataset.get_relation_insts` returns `[0,1]`-normalized
   boxes (`dataset.py:290-296`, `normalize_coords = True` at `:103`), and `util.compute.viou` compares
   them without rescaling. A sidecar scored against dataset GT is therefore not comparable as-is. The
   rewrite records the coordinate space in the payload and the docstring rather than guessing a
   rescale, which is only valid when the input is the original full-rate source.

3. **`viou` frame-id semantics (documented).** `viou` indexes `traj[k]` as frame `duration[0] + k`,
   and `get_relation_insts` uses `(begin_fid, end_fid)` from the annotations, which are **original
   video** frame ids. The returned `frame_ids` are decord indices of the input file, which only
   coincide when the input is that full-rate source. The `source_duration`/`sub_traj` construction is
   correct given that assumption; it is stated rather than assumed.

Two further pre-existing items are deliberately **out of scope** (unreachable from inference, and
touching them would violate "surgical changes"): `RandomHorizontalFlip`'s `__call__` is dedented to
module level so the class is not callable, and `hflip` flips a misspelled `oub_boxes` key. Both are
train-split-only and never reached by the val transform this pipeline uses.

## Non-goals

- No new model, checkpoint, or training change.
- No new dependency.
- **CPU and GPU runs are not required to produce bit-identical output.** They are compared, not
  diffed: metadata and threshold-independent counts are expected to match, box counts may not,
  and a large divergence is a bug to investigate rather than a pass. See Verification.
- The inherited rough edges stay as they are: weak vidorsmall checkpoints, short predicted
  tracklets, `sub_traj` on the source-frame axis.

## Modules

```
inference.py                  entry point: sys.path setup + vrdf_inference.cli.main()
scripts/inference/inference.sh   launcher: <mode> <video> [flags]
vrdf_inference/
  cli.py       build_arg_parser(), resolve_stage_args(), check_stage_args(), main() dispatch
  video.py     decord IO: open_video, iter_decoded_frames, video_meta, transform_frame, meta
  stage1.py    load_model(1), decode_frame, run_stage1_frame, run_detection
  tracklets.py form_tracklets, weighted_vote        <- stage-2 only
  stage2.py    load_model(2), build_pseudo_target, accumulate, classify
  render.py    render_detections (stage 1), render_triplets (stage 2), put_label, palette
  results.py   detection_payload, triplet_payload, write_eval_prediction, save_results
  console.py   log, warn, die
```

`tracklets.py` is separate from `stage2.py` because association is not classification: it is the
one module the stage-1 mode never touches, and keeping it apart makes that visible.

### Two entry points into Stage 1

Today `run_stage1` does two things: drives the decode loop _and_ decodes each frame's output. Only
the second is needed by both modes.

```python
# stage1.py
def decode_frame(model, frame_rgb, prev_features, transform, args, source_hw):
    """One frame → (detections, ceilings, features)"""

def run_detection(model, video_path, transform, device, args, on_frame=None):
    """Drive decode+detect over the whole video.

    Yields each frame's detections so the caller decides what to keep. `on_frame` lets the
    stage-2 pass accumulate state without a second forward pass.
    """
```

Stage-2 and stage-1 modes both call `run_detection`; stage 2 additionally feeds the per-frame
detections into `form_tracklets`.

## CLI

`--mode` is **required** and has no default, so a detection-only run can never silently become the
expensive full pipeline (and vice versa).

```bash
python inference.py --video clip.mp4 --mode stage1
python inference.py --video clip.mp4 --mode stage2 --conf_sub 0.10
sh scripts/inference/inference.sh stage1 clip.mp4
```

`--stage2_ckpt` and `--stage2_config` are `--mode stage2` only. In stage-1 mode they keep their
default _values_ (so no help text changes) but are never read and never validated, so a stage-1
run works on a machine with no Stage-2 checkpoint on disk.

The rest of the flag surface is unchanged, including the `main.get_args_parser()` parent, the
`conflict_handler="resolve"` re-registration of `--dataset_config`, and the
`argparse defaults < checkpoint args < config JSON < explicit CLI` resolution order, with **one
exception for `resolution`** (below).

### The checkpoint wins on `resolution`

`data/ckpts/vidorsmall_stage2` was trained at `resolution: "small"`, but
`configs/vidorsmall_stage2.json` now says `"large"` (commit `b833e60` changed it). So for the
`resolution` key alone the order becomes:

```
argparse default  <  config JSON  <  checkpoint  <  --stage{1,2}_resolution
```

The checkpoint is the only evidence of the scale the weights actually saw; the config is a
training-time file that has since been edited. Without this, every stage-2 run needs a remembered
`--stage2_resolution small` or it silently applies `roi_align` at the wrong scale and quality drops
with no error. A CLI override still wins, so overriding stays possible.

The existing `check_stage_args` warning stays for *other* keys, and the resolution warning becomes
unreachable by default — it now fires only when a CLI override contradicts the checkpoint, which is
exactly when it is worth saying.

**`--no_video` is removed.** It existed to skip rendering, which is exactly the capability this
design makes non-negotiable. Dropping it also removes the `if not args.no_video:` branch that
`main` and the summary handler currently wrap around rendering. The default behaviour is
unchanged — the flag was off unless passed — so stage-2 parity is unaffected. If a genuinely
long clip ever needs a pass with no video, that is a `--max_frames` prefix run, not a flag.

## Output

```
output/inference/
  stage1/<video_id>/                     # --mode stage1
    video.mp4            detections annotated
    results.json         metadata + per-frame detections
  stage2/<video_id>/                     # --mode stage2
    video.mp4            triplets annotated
    results.json         metadata + tracklets + predictions + per-frame detections
    eval_prediction.json util/evaluate-shaped sidecar
```

`--output_root` moves the `output/inference/` root; `--output_dir` still overrides everything for a
single run. The dataset name is recorded in `results.json`, not in the path.

## The two annotated videos

Both renderers share `put_label`, the colour palette, the header, the per-frame re-decode and the
`mp4v → avc1 → H264` codec fallback. They differ in what a box means.

- **Stage 1** draws every kept detection: the subject box in the subject colour, the object box in
  the object colour, one label `adult 0.19 / table 0.17` at the pair's top-left corner.
- **Stage 2** is unchanged from today: one colour per tracklet, one label
  `subject verb object [track_id] score`, with `--max_labels_per_frame` limiting how many are
  spelled out.

## Verification

Verification happens in two places, because the pipeline runs on both: **CPU smoke tests run here
and now**, and the **GPU run happens later on RunPod**. Both use the same commands; only `--device`
and the frame budget differ.

### Stage A — local CPU smoke tests (this machine, part of implementation)

This machine has torch 2.8 CPU, decord, cv2 and `data/ckpts/` populated, and the launcher already
falls back to `--device cpu` when `nvidia-smi` fails. Stage 1 costs ~2.6 s/frame on CPU, so every
check below uses `--max_frames 8` on `data/vidorsmall/videos/10042935613.mp4` and finishes in
under a minute.

1. **Lint:** `ruff check . && ruff format --check .`
2. **Stage 1 runs and writes the promised artefacts:**
   `output/inference/stage1/10042935613/{video.mp4,results.json}` exist; `results.json` has
   `num_detections > 0` and a `detections.frames` list of length 8.
3. **Both videos are real videos, not empty files.** For each mode, open the written `video.mp4`
   with `decord.VideoReader` and assert `len(vr) == num_frames` from that run's `results.json`.
   A 0-byte or wrong-length file means rendering silently failed while the run reported success —
   file existence alone does not catch that.
4. **Stage 1 is genuinely stage-1 only:** `--mode stage1` succeeds with
   `--stage2_ckpt /nonexistent.pth`.
5. **Stage 2 runs and still produces triplets:**
   `output/inference/stage2/10042935613/{video.mp4,results.json,eval_prediction.json}` exist, and
   `num_predictions` is non-zero and in the same order of magnitude as the pre-rewrite script measured
   on this clip. It is not expected to be *equal* — see step 6.
6. **Parity spot-check:** the pre-rewrite script (kept at `git show HEAD~12:inference.py`) run with
   the same flags produces the same `num_detections` and `num_tracklets`. Those two are stage-1
   geometry and must match exactly; a difference is a bug in the split. `num_predictions` is expected
   to differ, because defect 1's box-scale fix changes what `roi_align` samples — see the note under
   deviation 1 above. Confirm the cause by re-running the new code with `--stage2_resolution large`,
   which restores the old (identical-sizes) geometry and should reproduce the old count.
7. **Zero-detection run still writes a video.** Force it with `--conf_sub 0.99 --conf_obj 0.99`:
   the run must still produce a playable `video.mp4` of the right length. This is the path where
   "always write the video" is most likely to be quietly skipped.
8. **`ruff` clean on the package**, and the shell launcher passes `bash -n`.

### Stage B — GPU run on RunPod (the user runs this later)

The CPU smoke tests prove the wiring, not the throughput: 8 frames at ~2.6 s/frame says nothing
about a full clip. The GPU run is the real one, and it is the user's to launch.

```bash
sh scripts/inference/inference.sh stage1 data/vidorsmall/videos/10042935613.mp4 --device cuda
sh scripts/inference/inference.sh stage2 data/vidorsmall/videos/10042935613.mp4 --device cuda
```

No `--max_frames`, so the whole clip runs at the default `--every_n_frames 5`. What to check:

- `results.json` reports the same `num_source_frames`, `stride` and `source_size` as the CPU run —
  metadata is device-independent, so any difference is a bug. `num_frames` is **not** comparable:
  the CPU pass is capped at 8 frames and this one is not.
- Both modes wrote a `video.mp4` of the recorded length (the same assertion as step 3).
- The `best score before thresholding` line. Read it expecting the GPU number to be **at least** the
  CPU one: it is a max over frames, and the GPU run sees the whole clip while the CPU run sees the
  first 8 frames, so more frames can only raise it. Stage 1 also decays over time on this
  checkpoint (~0.18 at frame 0 to ~0.09 by frame 4 at stride 5), so in practice the two maxima come
  from the same early frames and should land close. A GPU figure *below* the CPU one is the real
  signal — same weights, same early frames, so it points at a decoding difference.
- Box *counts* legitimately differ a little: CPU and GPU kernels are not bit-identical, and a query
  sitting exactly on the `--conf_sub`/`--conf_obj` boundary can fall either side.
- Wall-clock throughput, for the record: Stage 1 fps and Stage 2 fps are both logged.

To make the comparison cheap, the CPU artefacts are kept out of the way of the GPU ones: run the
CPU pass with `--output_root output/inference_cpu` (or `--output_dir` for a single run) so the two
sets sit side by side and neither overwrites the other. `output/` is already gitignored, so
neither ends up in the tree.

**A large CPU/GPU divergence is a finding, not a pass.** If GPU `num_detections` is wildly off the
CPU figure at the same thresholds, that is worth root-causing with `superpowers:systematic-debugging`
before trusting either number — the same checkpoint and the same code path should not disagree by
much.

## Risks

| Risk                                                         | Mitigation                                                            |
| ------------------------------------------------------------ | --------------------------------------------------------------------- |
| Splitting silently changes stage-2 numbers                   | Verification step 6 diffs counts against `git show HEAD:inference.py` |
| A run reports success but the annotated video is missing or unplayable | Video is a required deliverable in both modes; verification steps 3 and 7 open it and assert its length |
| `inference.py` at root shadowed by the package               | Package is `vrdf_inference/`, not `inference/`                        |
| `import` of `models`/`util` breaks when run from another CWD | `inference.py` resolves its own directory onto `sys.path` first       |
| Stage-1 mode accidentally needs a stage-2 checkpoint         | Verification step 4 runs with a bogus `--stage2_ckpt`                 |
| A CPU-only bug (or a CPU-only pass) hides a GPU failure      | CPU smoke tests are Stage A, the full-clip GPU run is Stage B, and Stage B re-checks the same metadata assertions |
| Checkpoint-wins precedence hides a genuinely wrong config    | The precedence applies to `resolution` only; every other key keeps config-wins, and `check_stage_args` still reports disagreements |
