# Phase 1: Data Preparation

## Current State (before starting)

```
data/vidor/
├── annotations/
│   ├── train/          ✅ 7,000 JSON annotation files (flat)
│   └── val/            ⚠️ EMPTY — need validation data
├── videos/             ❌ EMPTY — need to re-download
├── action.txt          ✅ 42 action predicates
├── obj.txt             ✅ 80 object classes
├── rel.txt             ✅ 50 relations (action + spatial)
├── spatial.txt         ✅ 8 spatial relations
└── train_files.json    ✅ List of 5,563 training video IDs
```

## Step 1.1: Re-download Videos

The 2,964 training videos (~10GB) need to be placed in `data/vidor/videos/`.

**Requirements:**
- Flat directory (no subdirectories)
- Files named by video ID: `<video_id>.mp4`
- Example: `2401075277.mp4`, `2440175990.mp4`, etc.

**Source:** VidOR dataset from the official release. Training videos are organized by category folders (0000/, 0001/, etc.) in the raw dataset.

**After download:**
```bash
# If downloaded as category subdirs, flatten:
find data/vidor/videos_raw -name '*.mp4' -exec mv {} data/vidor/videos/ \;
```

## Step 1.2: Download Validation Annotations

Place validation JSON files in `data/vidor/annotations/val/` (flat, no subdirectories).

**Source:** VidOR dataset validation split — annotation JSONs organized by category in the raw release.

```bash
# After download, flatten:
find data/vidor/annotations_raw/validation -name '*.json' -exec cp {} data/vidor/annotations/val/ \;
```

**Verification after Steps 1.1-1.2:**
```bash
ls data/vidor/videos/ | wc -l      # should be 2964+ videos
ls data/vidor/annotations/train/ | wc -l   # should be 7000 JSONs
ls data/vidor/annotations/val/ | wc -l     # should have validation JSONs
```

## Step 1.3: Generate Annotation Pickle

The script reads raw JSON annotations and converts them to per-frame pickle format.

**IMPORTANT:** Must run from inside `data/` because `prepare.py` reads `action.txt`/`obj.txt` relative to CWD and writes to `metadata/` relative to CWD.

```bash
cd data
python prepare.py --func get_anno --dbname vidor --root_dir .
```

**Output:** `data/metadata/vidor_annotations.pkl`

**What it does:**
- Reads all JSONs from `vidor/annotations/train/` and `vidor/annotations/val/`
- For each video, iterates over `relation_instances` to build per-frame annotation dicts
- Each frame gets: `sub_labels`, `obj_labels`, `verb_labels`, `so_track_ids`, `sub_boxes`, `obj_boxes`
- Pickles the result as `{video_id: {"frame_annos": {fid: {...}}, "rel_tag_uids": {...}}}`

**Verification:**
```python
import pickle
with open('metadata/vidor_annotations.pkl', 'rb') as f:
    annos = pickle.load(f)
print(f'{len(annos)} videos loaded')
sample_vid = list(annos.keys())[0]
print(f'Sample video: {sample_vid}')
print(f'Frames with annotations: {len(annos[sample_vid]["frame_annos"])}')
```

## Step 1.4: Generate Frame Indices

These JSONs define which frame sequences form valid training clips.

```bash
cd data

# Stage 1: frame pairs for detection + tracking (stride 8, max 32-frame window)
python prepare.py --func get_fid --dbname vidor --split train --stage 1 --timestep 8 --minmax_dur 32 --root_dir .

# Stage 2: 8-frame sequences for relation classification
python prepare.py --func get_fid --dbname vidor --split train --stage 2 --timestep 8 --minmax_dur 32 --root_dir .

# Validation: per-video positive frame bitmaps
python prepare.py --func get_fid --dbname vidor --split val --timestep 8 --minmax_dur 32 --root_dir .
```

**Outputs:**
- `data/metadata/vidor_train_frames_stage1.json` — `{"train_begin_fids": [...], "durations": [...]}`
- `data/metadata/vidor_train_frames_stage2.json` — same format
- `data/metadata/vidor_val_frames.json` — `{video_id: [0,1,0,1,...]}` per-frame positive masks

**What these do:**
- Stage 1 (`--stage 1`): Scans video frames at stride `timestep=8`. For each start position, looks ahead up to `minmax_dur=32` frames. Keeps clips where at least 2 consecutive frames have active relations.
- Stage 2 (`--stage 2`): Same scan but requires at least 4 consecutive frames AND class consistency (same track IDs and verb labels at begin/end of clip).
- Val: Creates a per-video bitmap where `pos_fids[fid] = 1` if any relation exists in that frame.

**Verification:**
```bash
python -c "
import json
for name in ['vidor_train_frames_stage1', 'vidor_train_frames_stage2', 'vidor_val_frames']:
    with open(f'data/metadata/{name}.json') as f:
        data = json.load(f)
    if 'train_begin_fids' in data:
        print(f'{name}: {len(data[\"train_begin_fids\"])} training clips')
    else:
        print(f'{name}: {len(data)} videos with frame bitmaps')
"
```

## Expected Output Summary

After Phase 1, you should have:
```
data/metadata/
├── vidor_annotations.pkl           # Main annotation pickle
├── vidor_train_frames_stage1.json  # Stage 1 train clip index
├── vidor_train_frames_stage2.json  # Stage 2 train clip index
└── vidor_val_frames.json           # Validation frame bitmap

data/vidor/
├── annotations/
│   ├── train/    (7,000 JSONs)
│   └── val/      (validation JSONs)
├── videos/       (3,000+ MP4s)
├── action.txt, obj.txt, rel.txt, spatial.txt
└── train_files.json
```
