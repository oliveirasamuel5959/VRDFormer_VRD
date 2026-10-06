# Plan: Build an 8× Smaller VidOR Dataset (`vidorsmall`)

Goal: a training-scale subset of VidOR that trains Stage 1 + Stage 2 in a fraction of the time and
compute, on single- or multi-GPU, and is small enough to host on Kaggle for remote (Colab/RunPod)
training. Roughly **8× smaller than the full dataset** in video bytes, annotation bytes, clip count,
and epoch time.

This plan generates the dataset the same way `spec/train_pipeline/01_data_preparation.md` describes,
but from a *subset* of raw videos/annotations, and writes it under a new name so the full `vidor/`
tree is left untouched.

---

## 1. Decisions (confirmed with the user)

| Decision | Choice |
|----------|--------|
| What "8× smaller" means | Reduce **both splits to ~1/8** of their videos (train 7000 → ~875, val 835 → ~104) |
| How videos are chosen | **Coverage-stratified** — guarantee all 50 verbs + 80 objects, then maximize distinct relation triplets (biased toward rare ones) |
| Where the subset lives | **New dir `data/vidorsmall/`**; `data/vidor/` originals fully preserved |
| Kaggle delivery | **New dataset slug** `samuelpatricio/vrdformer-vidor-small` (existing 17.5 GB dataset untouched) |
| Video re-encode | **None** — copy `.mp4` files byte-for-byte (smaller file size *and* predictable, no quality loss) |

---

## 2. Measured current state (before starting)

From the live tree:

| Artifact | Size / count | Notes |
|----------|--------------|-------|
| `data/vidor/videos/*.mp4` | **7,835 files, 27.0 GB** (avg 3.5 MB) | Every min is 0.5–50 MB; no re-encode needed to shrink |
| `data/vidor/annotations/train/*.json` | **7,000 files, 2.4 GB** | |
| `data/vidor/annotations/val/*.json` | **835 files, 277 MB** | |
| `data/vidor/metadata/vidor_annotations.pkl` | **3.07 GB** | The big derived pickle — **lives here, not in `data/metadata/`** |
| `data/vidor/metadata/batches/` + `data/metadata/batches/` | 13 × ~100 MB each (**~1.3 GB per location**) | Duplicate derived shards from `data/batch_get_anno.py` |
| `data/metadata/vidor_train_frames_stage{1,2}.json` | 303,120 / 282,389 clips over 2,942 / 2,940 videos | Stage-1/2 clip indices, already regenerated here |
| `data/metadata/vidor_val_frames.json` | 835 videos, 841,929 frames | Positive-frame bitmaps |
| `data/vidor/{action,obj,rel,spatial}.txt` | 50 / 80 / 50 / 8 non-blank lines | 50 verb classes, 80 object classes |
| `data/vidor/train_files.json` | 5,563 IDs | **Unreferenced anywhere in the code** — raw-release leftover |
| DETR weights | `data/vidor/detr-r101-2c7b67e5.pth` and `data/weights/detr-r101-2c7b67e5.pth` | Identical (md5 `7ec0027c…`), 243 MB each |

**Videos ↔ annotations are a perfect 1:1 match** (7835 = 7000 + 835), so a video subset is exactly an
annotation subset. There are 1,437 train annotations *not* in `train_files.json`; `train_files.json`
must **not** be used as the source of truth — iterate the annotation directory, as `prepare.py` does.

### Two facts that constrain the design

1. **`data/metadata/` is the hardcoded load path.** `datasets/vidor.py:101` and `datasets/vidvrd.py:80`
   build `anno_file = "data/metadata/%s_annotations.pkl" % dbname` (CWD-relative), and the frame
   indices come from `data/metadata/%s_%s_frames_stage%d.json` (train) / `data/metadata/%s_val_frames.json`
   (val). The 3 GB pkl sitting in `data/vidor/metadata/` is *not* what the loader reads — the Colab
   notebook moves it into `data/metadata/` at runtime. Regenerated small metadata must land in
   `data/metadata/`.
2. **`engine.py:233` reads `data/<dataset>/action.txt`.** So for `dataset = "vidorsmall"` the class
   list must exist at **`data/vidorsmall/action.txt`**, or stage-2 eval crashes immediately.

Nice property: `models/__init__.py:23-24` derives head sizes as `80 if 'vidor' in args.dataset else 35`
and `50 if 'vidor' in args.dataset else 132`. **`"vidorsmall"` contains `"vidor"`, so class counts stay
80/50** — no model change. `datasets/__init__.py:57` routes anything ≠ `"vidvrd"` to the VidOR class, so
`dataset: "vidorsmall"` needs no new dataset class.

---

## 3. Approach

```
data/vidor/                      (untouched, 27 GB)
   │
   │  tools/make_vidor_small.py     [Step 2]
   │   • scan 7,000 train JSONs → per-video classes + triplets
   │   • greedy coverage-stratified pick 875 train / 104 val
   ▼
data/vidorsmall/
   ├── videos/*.mp4                     (copied, ~3.4 GB)
   ├── annotations/train/*.json  (875)
   ├── annotations/val/*.json    (104)
   └── action.txt obj.txt rel.txt spatial.txt   (copied)
   │
   │  data/prepare.py --dbname vidorsmall   [Step 4]  (tiny patch to allow the name)
   ▼
data/metadata/
   ├── vidorsmall_annotations.pkl               (~380 MB)
   ├── vidorsmall_train_frames_stage1.json
   ├── vidorsmall_train_frames_stage2.json
   └── vidorsmall_val_frames.json
   │
   ├──▶ local train / smoke  [Step 6, 7]
   └──▶ Kaggle bundle + new dataset  [Step 8, 9]
```

---

## 4. Steps

### Step 1 — Preflight & disk budget

```bash
cd /home/samuel/projects/VRDFormer_VRD
df -h .                                     # need ~5 GB free for the small set + ~4 GB bundle
ls data/vidor/videos | wc -l                # 7835
ls data/vidor/annotations/train | wc -l     # 7000
```

Estimated small footprint: videos ~3.4 GB, JSONs ~0.34 GB, pkl ~0.38 GB, frame JSONs ~2 MB ⇒ **~4.1 GB**,
plus the Kaggle bundle (hard-linked in Step 8, ~0 extra; a copy would add ~4 GB). No deletion of
`data/vidor/` originals is required for this plan (see the optional cleanup in §6).

### Step 2 — Write the subset generator: `data/make_vidor_small.py`

New script. Reads raw annotations, selects videos, copies them. Deterministic (`random.seed(42)`, ties
broken by sorted video ID). Uses only the annotation dirs (never `train_files.json`).

Interface:

```bash
cd data
python make_vidor_small.py \
    --src vidor --dst vidorsmall \
    --train_ratio 0.125 --val_ratio 0.125 \
    --seed 42 [--dry-run]
```

Algorithm:

1. **Scan** each train/val JSON once; store a compact per-video summary:
   `{verbs:set, objects:set, triplets:set[(subj, pred, obj)], n_rels, frame_count}`.
   (~7000 files / 2.4 GB — a few minutes; the raw JSONs are not kept in memory.)
2. **Targets:** `N_train = round(7000 × 0.125) = 875`, `N_val = round(835 × 0.125) = 104`.
   Coverage targets: all 50 verbs, all 80 objects.
3. **Coverage-stratified greedy pick** (per split):
   - Element weight = `1 / global_frequency(element)` so rare verbs/objects/triplets dominate.
   - Phase A: greedily pick the video maximizing *uncovered* verb+object gain until all classes are
     covered or the budget is spent.
   - Phase B: continue greedily on triplet gain (weighted by rarity) until `N` videos are chosen.
   - Ties → smallest video ID (deterministic).
4. **Copy** the selected `videos/<id>.mp4` (byte-for-byte, `shutil.copy2`) and
   `annotations/{train,val}/<id>.json` into `data/vidorsmall/`. Copy `action.txt`, `obj.txt`, `rel.txt`,
   `spatial.txt` alongside.
5. **Report** (and write `data/vidorsmall/selection_report.json`): per-split video counts, byte totals,
   verb/object/triplet coverage vs. global, and the list of globally-rare classes that ended up with
   <20 instances in the subset (so the user can judge whether to raise the ratio).
6. `--dry-run` prints the selection + coverage report and copies nothing.

Guard: assert the selected set has full verb/object coverage; if not, warn loudly and print which
classes are missing (the val 1/8 cut in particular may not cover all classes — see §5).

### Step 3 — Patch `data/prepare.py` to accept `vidorsmall`

The only blocker is the locked `--dbname` choices; the file/path logic already works for any name
(`dbname.replace("part","")` = `"vidorsmall"`, and `"part" not in dbname`, so no 100-video truncation).

```python
# data/prepare.py:284
parser.add_argument("--dbname", default="vidvrd", type=str,
                    choices=["vidvrd", "vidor", "vidorpart", "vidorsmall"])
```

Backward compatible; nothing else changes.

### Step 4 — Generate the small dataset

Run **from inside `data/`** (same CWD rule as `spec/train_pipeline/01_data_preparation.md`):

```bash
cd data
python make_vidor_small.py --src vidor --dst vidorsmall --train_ratio 0.125 --val_ratio 0.125
```

Verify the copy:

```bash
ls data/vidorsmall/videos | wc -l                # ~979
ls data/vidorsmall/annotations/train | wc -l     # ~875
ls data/vidorsmall/annotations/val | wc -l       # ~104
du -sh data/vidorsmall                           # ~3.7 GB
```

### Step 5 — Regenerate annotations + frame indices

Exactly the sequence from `01_data_preparation.md` §1.3–1.4, with `--dbname vidorsmall`:

```bash
cd data

python prepare.py --func get_anno --dbname vidorsmall --root_dir .
# -> data/metadata/vidorsmall_annotations.pkl   (~380 MB)

python prepare.py --func get_fid --dbname vidorsmall --split train --stage 1 \
    --timestep 8 --minmax_dur 32 --root_dir .
python prepare.py --func get_fid --dbname vidorsmall --split train --stage 2 \
    --timestep 8 --minmax_dur 32 --root_dir .
python prepare.py --func get_fid --dbname vidorsmall --split val \
    --timestep 8 --minmax_dur 32 --root_dir .
# -> data/metadata/vidorsmall_{train_frames_stage1,train_frames_stage2,val_frames}.json
```

The small pkl (~380 MB) fits comfortably in RAM, so `data/sharded_annotations.py` /
`batch_get_anno.py` are **not needed** for `vidorsmall`.

### Step 6 — Validate

1. **Metadata loads and roughly matches:**
   ```bash
   python -c "
   import pickle as pkl, json
   a = pkl.load(open('data/metadata/vidorsmall_annotations.pkl','rb'))
   print('videos', len(a), 'frame annos', sum(len(v['frame_annos']) for v in a.values()))
   for n in ['vidorsmall_train_frames_stage1','vidorsmall_train_frames_stage2','vidorsmall_val_frames']:
       d = json.load(open(f'data/metadata/{n}.json'))
       print(n, len(d['train_begin_fids']) if 'train_begin_fids' in d else len(d))
   "
   ```
2. **Every referenced clip's video exists** (the frames JSON references `vid-fid`; the pkl references
   the same `vid`): assert `{f.split('-')[-2] for f in train_begin_fids} ⊆ set(videos)`.
3. **Class coverage:** all 50 verbs and 80 objects appear in `vidorsmall/annotations/train`.
4. **Decode smoke test:** open one copied `.mp4` with `decord.VideoReader` and read a frame — confirms
   the copy is a valid video, not a truncated file.
5. **End-to-end smoke (Stage 1, 1 epoch):** with a small config (Step 7), confirm loss decreases and an
   epoch finishes. Budget: full stage 1 was 303k clips; ~37k clips here ⇒ roughly **8× faster per epoch**.
   Note the known harmless `NameError: eval_one_epoch` at stage-1 epoch end (rough edge in CLAUDE.md).

### Step 7 — Configs

New files (clone the `vidor_kaggle_*` pair, repoint the name and paths):

- `configs/vidorsmall_stage1.json` — `dataset: "vidorsmall"`, `vidor_path: "data/vidorsmall"`,
  `pretrain: "data/weights/detr-r101-2c7b67e5.pth"`, `output_dir: "data/ckpts/vidorsmall_stage1"`,
  stage-1 fields as `vidor_kaggle_stage1.json`.
- `configs/vidorsmall_stage2.json` — `dataset: "vidorsmall"`, same path,
  `pretrain: "data/ckpts/vidorsmall_stage1/checkpoint0000.pth"`, stage-2 fields as
  `vidor_kaggle_stage2.json` (keep `train_clip_sample_ratio: 0.25`, `amp`, `benchmark`).

Both need the keys that exist only in JSON (`cautious`, `by_ratio`) — inherit from the source configs.
Remember `--dataset_config` **overrides** CLI args (`main.py:143-148`), so edit the JSON, not the flags.
Configs that hardcode absolute paths (the original `vidor_stage*.json`) are irrelevant here — start from
the relative-path `vidor_kaggle_*` pair.

### Step 8 — Kaggle bundle + upload

Build an upload-ready folder whose *contents* extract cleanly into `data/`:

```
data/vidor_small_bundle/
├── dataset-metadata.json
├── vidorsmall/                 -> data/vidorsmall/  (videos, annotations, *.txt)
├── metadata/                   -> data/metadata/    (the 4 vidorsmall_* files)
└── weights/detr-r101-2c7b67e5.pth   -> data/weights/
```

Create it without duplicating video bytes on disk:

```bash
cd /home/samuel/projects/VRDFormer_VRD
mkdir -p data/vidor_small_bundle/{metadata,weights}
cp -al data/vidorsmall data/vidor_small_bundle/vidorsmall     # hardlink (same FS); fallback: cp -r
cp data/metadata/vidorsmall_* data/vidor_small_bundle/metadata/
cp data/weights/detr-r101-2c7b67e5.pth data/vidor_small_bundle/weights/
cat > data/vidor_small_bundle/dataset-metadata.json <<'EOF'
{"title": "VRDFormer-VidOR-Small", "id": "samuelpatricio/vrdformer-vidor-small", "licenses": [{"name": "CC0-1.0"}]}
EOF
du -sh data/vidor_small_bundle          # ~4.1 GB (hardlinks cost ~0 extra on disk)
```

Upload (new dataset, existing one untouched):

```bash
kaggle datasets create -p data/vidor_small_bundle --dir-mode zip
# -> samuelpatricio/vrdformer-vidor-small   (~4 GB, comfortably under limits)
```

**Do not** re-zip or version the old 17.5 GB `vrdformer-vidor` dataset.

### Step 9 — Colab/RunPod notebook variant

Add `notebooks/colab_kaggle_train_small.ipynb` (copy of `colab_kaggle_train.ipynb`, retargeted):

- Cell 2: `kaggle datasets download samuelpatricio/vrdformer-vidor-small` → `unzip -d data`.
  With the bundle layout above, extraction lands `vidorsmall/`, `metadata/`, `weights/` directly under
  `data/` — **no `mv` juggling** (unlike the current notebook's cells 7–8).
- Verify cell: check `data/vidorsmall/videos`, `data/vidorsmall/annotations/{train,val}`,
  `data/metadata/vidorsmall_train_frames_stage1.json`, `data/metadata/vidorsmall_val_frames.json`,
  `data/weights/detr-r101-2c7b67e5.pth`, `data/vidorsmall/action.txt`, and both configs.
- Train cells: `--dataset_config configs/vidorsmall_stage1.json` / `..._stage2.json`.
- Eval cell: `--eval --dataset_config configs/vidorsmall_stage2.json --resume .../checkpoint.pth`.

---

## 5. What changed in the numbers (measured after generation)

The 1/8 cut was applied **per video**. Because the coverage-stratified picker favors
relation-dense videos, those videos are longer and carry more clips than average, so the
clip/epoch reduction is smaller than 8×:

| Quantity | Full vidor | `vidorsmall` (measured) | Factor |
|----------|-----------|-------------------------|--------|
| Train videos | 7,000 | 875 | 8.0× |
| Val videos | 835 | 104 | 8.0× |
| Videos on disk | 28.98 GB | 4.17 GB | 6.9× |
| Avg video size | 3.70 MB | 4.26 MB | (picker favors longer videos) |
| Annotations dir | 2.80 GB | 0.47 GB | 6.0× |
| Annotations pkl | 3.07 GB | 0.61 GB | 5.0× |
| Stage-1 train clips | 303,120 (2,942 videos) | 116,241 (875 videos) | **2.6× fewer** |
| Stage-2 train clips | 282,389 | 106,492 | 2.7× fewer |
| Clips per video | 103 | 133 | +29% |
| Kaggle bundle zip | 17.5 GB | 4.45 GB | 3.9× |

**So an epoch is ~2.6× faster, not 8×.** The stored dataset and download/upload are ~6–8×
smaller; training throughput is ~2.6× better. For a bigger clip reduction, lower the ratios
(e.g. `--train_ratio 0.04`) rather than the picker, or subsample clips with
`train_clip_sample_ratio` in the config (Stage 2 already runs at 0.25).

Coverage came out better than predicted: **all 50 verbs and 80 objects are present in both
splits**, and 5,777 of 6,258 (92%) of all relation triplets survive in the train subset.

**Coverage caveat:** while class *presence* is guaranteed, sample *counts* for rare classes are
thin, so:

- Rare verbs (e.g. `shout_at`, `cut`, `get_on`, `knock`) have few instances — their per-class mAP
  is noisy and near-zero.
- `zeroshot_triplets = val_triplets − train_triplets` is computed against the *small* train split
  (`datasets/__init__.py`), so more val triplets count as "zero-shot" than on full data — zero-shot
  numbers are **inflated and not comparable** to the paper's full-data results.
- This is fine for fast iteration and smoke tests. If evaluation stability matters, bump `--val_ratio`
  (e.g. 0.25–1.0) — val is only 277 MB of JSON + 104–835 small videos.

Add a guard so an **empty** zeroshot set doesn't crash the eval path:

```python
# datasets/__init__.py, dataloader_initializer
if not args.debug:
    try:
        dataset_val.zeroshot_triplets = dataset_val.get_triplets().difference(dataset_train.get_triplets())
    except Exception as e:
        print(f'[warn] zeroshot triplet computation failed, using empty set: {e}')
        dataset_val.zeroshot_triplets = set()
```

---

## 6. Optional: reclaim disk (only if needed, and only derived artifacts)

The plan above deletes nothing, so `data/vidor/` stays intact. If space runs short, these are
**derived duplicates** that can be removed *after* the small dataset is validated and uploaded —
and only with the user's explicit confirmation, since they are what the *existing* Kaggle dataset
bundles:

| Candidate | Size | Why safe to drop |
|-----------|------|------------------|
| `data/vidor/metadata/vidor_annotations.pkl` | 3.07 GB | Regenerable via `prepare.py --func get_anno --dbname vidor`; not the loader's path |
| `data/vidor/metadata/batches/` and `data/metadata/batches/` | ~1.3 GB each | Intermediate shards from `batch_get_anno.py`; regenerate on demand |
| `data/vidor/detr-r101-2c7b67e5.pth` | 243 MB | Byte-identical to `data/weights/detr-r101-2c7b67e5.pth` |
| `data/vidor/kaggle_dataset_metadata/` | ~KB | Superseded by `data/vidor_small_bundle/dataset-metadata.json` |
| `data/metadata/vidorpart*.pkl`, `vidorpartsmoke_*` | ~34 MB | `vidorpart` experiment leftovers |

**Never** delete under `data/vidor/{videos,annotations}` unless the user explicitly asks to drop the full
dataset — those are the originals the small set is derived from.

---

## 7. Risks & rough edges to watch

1. **`prepare.py` must run with CWD = `data/`** and reads `data/<name>/action.txt` + writes
   `data/metadata/` — running it from the repo root silently writes to the wrong place. Always `cd data`
   first (same trap documented in `01_data_preparation.md`).
2. **`engine.py` stage-2 eval needs `data/vidorsmall/action.txt`** (§2). The generator copies it; verify it.
3. **Stage 1 has no eval path** — `eval_one_epoch` `NameError` at the end of each stage-1 epoch
   (`main.py:224`). Harmless for training; don't mistake it for a data bug.
4. **`--debug` skips raw-annotation loading and the zeroshot diff**, so it will not catch a broken
   `vidorsmall` annotation set. Use the Step 6 checks, not `--debug`, to validate the new dataset.
5. **`vidvrd_stage1_deform.json` / `vidor_stage1_deform.json`** are not involved here; stay on the
   non-deformable path (`num_feature_levels: 1`, no `models/ops` compile).
6. **Tie-break determinism:** the picker must sort candidates by video ID on score ties, or reruns
   produce different subsets and the Kaggle dataset becomes irreproducible.
7. **Hardlink caveat:** `cp -al` requires the bundle and `data/vidorsmall` on the same filesystem. If not,
   fall back to `cp -r` (costs ~+4 GB) — check `df` for both targets.
8. **Zero-shot comparison:** results from `vidorsmall` are for pipeline/debug validation, not for
   reporting against published full-data numbers (§5).

---

## 8. Verification checklist

- [ ] `data/vidorsmall/` has ~875 train + ~104 val JSONs and ~979 videos; `data/vidor/` unchanged.
- [ ] All 50 verbs and 80 objects present in `vidorsmall/annotations/train`; coverage report written.
- [ ] `data/metadata/vidorsmall_annotations.pkl` loads; video count matches the copy.
- [ ] The three `vidorsmall_*frames*.json` files exist; every referenced video exists on disk.
- [ ] A copied `.mp4` decodes in decord.
- [ ] Stage-1 config runs 1 epoch end-to-end and the loss decreases.
- [ ] Stage-2 config trains + `--eval` runs (needs `data/vidorsmall/action.txt`).
- [ ] `data/vidor_small_bundle/` ≈ 4 GB; `dataset-metadata.json` id is
      `samuelpatricio/vrdformer-vidor-small`.
- [ ] Kaggle dataset downloads and the notebook's verify cell passes on a fresh runtime.

---

## 9. Files to create / modify

| File | Action | Purpose |
|------|--------|---------|
| `data/make_vidor_small.py` | **new** | Coverage-stratified subset generator + copier |
| `data/prepare.py` | modify (1 line) | Allow `--dbname vidorsmall` |
| `datasets/__init__.py` | modify (guard) | Don't crash on an empty zeroshot triplet set |
| `configs/vidorsmall_stage1.json` | **new** | Stage-1 config, relative paths, `dataset: vidorsmall` |
| `configs/vidorsmall_stage2.json` | **new** | Stage-2 config |
| `notebooks/colab_kaggle_train_small.ipynb` | **new** | Colab notebook for the small Kaggle dataset |
| `data/vidor_small_bundle/dataset-metadata.json` | **new** | Kaggle upload metadata |
| `plans/vidor_small_dataset.md` | this file | The plan |

No changes to `models/`, `datasets/vidor.py`, the training loop, or any config used by the full dataset.
