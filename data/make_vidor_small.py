"""Build a small, coverage-stratified subset of VidOR for fast training runs.

Reads the raw annotation JSONs of a source dataset (default ``vidor``) and selects a
subset of videos so that:

  1. every verb (action.txt) and object (obj.txt) class is represented, and
  2. the number of distinct relation triplets (subject, predicate, object) is maximized,
     weighted by global rarity so that rare classes/triplets are not dropped.

The selected videos and their annotation JSONs are copied byte-for-byte into a new
dataset directory (default ``vidorsmall``); the source tree is never modified. A
``selection_report.json`` describing the selection is written next to the copy.

Usage (run from inside ``data/``):

    python make_vidor_small.py --src vidor --dst vidorsmall \
        --train_ratio 0.125 --val_ratio 0.125

    python make_vidor_small.py --src vidor --dst vidorsmall --dry-run --limit 300
"""
import argparse
import glob
import json
import os
import random
import shutil
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op progress wrapper
    def tqdm(iterable, **kwargs):
        return iterable


# --------------------------------------------------------------------------------------
# Annotation scanning
# --------------------------------------------------------------------------------------

def load_class_dicts(src_dir):
    """Map class names to integer ids using action.txt / obj.txt of the source dataset."""
    with open(os.path.join(src_dir, "action.txt"), "r") as f:
        action_list = [l.strip() for l in f.readlines() if l.strip()]
    with open(os.path.join(src_dir, "obj.txt"), "r") as f:
        obj_list = [l.strip() for l in f.readlines() if l.strip()]
    return ({n: i for i, n in enumerate(action_list)},
            {n: i for i, n in enumerate(obj_list)})


def summarize_anno(path, action_dict, obj_dict):
    """Read one raw annotation JSON and return a compact summary for selection.

    Returns None if the file cannot be parsed or has no usable relations.
    """
    with open(path, "r") as f:
        data = json.load(f)

    tid2cls = {}
    for obj in data["subject/objects"]:
        tid2cls[obj["tid"]] = obj["category"]

    verbs, objects, triplets = set(), set(), set()
    for rel in data["relation_instances"]:
        pred = rel["predicate"]
        if pred not in action_dict:
            continue
        sid, oid = rel["subject_tid"], rel["object_tid"]
        scls, ocls = tid2cls.get(sid), tid2cls.get(oid)
        if scls is None or ocls is None or scls not in obj_dict or ocls not in obj_dict:
            continue
        pid = action_dict[pred]
        verbs.add(pid)
        objects.add(obj_dict[scls])
        objects.add(obj_dict[ocls])
        triplets.add((obj_dict[scls], pid, obj_dict[ocls]))

    return {
        "video_id": data.get("video_id", os.path.splitext(os.path.basename(path))[0]),
        "frame_count": data.get("frame_count", 0),
        "n_rels": len(data["relation_instances"]),
        "verbs": verbs,
        "objects": objects,
        "triplets": triplets,
    }


def scan_split(anno_dir, action_dict, obj_dict, limit=None):
    """Scan every annotation JSON in ``anno_dir`` into compact per-video summaries."""
    files = sorted(glob.glob(os.path.join(anno_dir, "*.json")))
    if limit is not None:
        files = files[:limit]

    summaries = {}
    for path in tqdm(files, desc="scan %s" % os.path.basename(anno_dir)):
        vid = os.path.splitext(os.path.basename(path))[0]
        try:
            summaries[vid] = summarize_anno(path, action_dict, obj_dict)
        except Exception as e:  # noqa: BLE001 - report and skip malformed entries
            print("[warn] skipping %s: %s" % (path, e))

    empty = [v for v, s in summaries.items() if not s["triplets"]]
    if empty:
        print("[warn] %d videos have no usable relations (e.g. %s)" % (len(empty), empty[:3]))
    return summaries


def global_frequencies(summaries):
    """Count how many videos each verb / object / triplet appears in."""
    verb_freq, obj_freq, trip_freq = Counter(), Counter(), Counter()
    for s in summaries.values():
        verb_freq.update(s["verbs"])
        obj_freq.update(s["objects"])
        trip_freq.update(s["triplets"])
    return verb_freq, obj_freq, trip_freq


# --------------------------------------------------------------------------------------
# Selection: two-phase weighted greedy coverage (lazy / CELF)
# --------------------------------------------------------------------------------------

def _greedy_cover(vid_list, elements_of, weights, budget, already_selected, desc):
    """Pick up to ``budget`` videos maximizing weight of newly covered elements.

    ``elements_of(vid)`` returns the element set of a video; ``weights[element]`` is its
    rarity weight. Plain greedy: at each step take the video with the largest marginal
    gain in uncovered weight. At this scale (~10^3 picks over ~10^4 videos) recomputing
    every gain each step is cheap and keeps the logic obviously correct. Ties break on the
    smallest video id so the selection is deterministic.
    """
    covered = set()
    for vid in already_selected:
        covered.update(elements_of(vid))

    elems = {v: list(elements_of(v)) for v in vid_list if v not in already_selected}
    remaining = sorted(elems.keys())

    def gain(vid):
        return sum(weights[e] for e in elems[vid] if e not in covered)

    picked = []
    for _ in tqdm(range(min(budget, len(remaining))), desc=desc):
        best_vid, best_gain = None, 0.0
        for vid in remaining:
            g = gain(vid)
            if g > best_gain or (g == best_gain and g > 0 and best_vid is not None and vid < best_vid):
                best_gain, best_vid = g, vid
        if best_vid is None or best_gain <= 0:
            break
        picked.append(best_vid)
        remaining.remove(best_vid)
        covered.update(elems[best_vid])

    return picked


def select_videos(summaries, n_target, rng):
    """Two-phase selection: guarantee class coverage, then maximize triplet coverage."""
    verb_freq, obj_freq, trip_freq = global_frequencies(summaries)
    vid_list = sorted(summaries.keys())

    # Rarity weights: a class/triplet seen in fewer videos is more valuable to include.
    verb_w = {e: 1.0 / c for e, c in verb_freq.items()}
    obj_w = {e: 1.0 / c for e, c in obj_freq.items()}
    trip_w = {e: 1.0 / c for e, c in trip_freq.items()}

    def class_elements(vid):
        s = summaries[vid]
        return set(("v", e) for e in s["verbs"]) | set(("o", e) for e in s["objects"])

    class_weights = {}
    class_weights.update({("v", e): w for e, w in verb_w.items()})
    class_weights.update({("o", e): w for e, w in obj_w.items()})

    # Phase A: cover every verb and object class, preferring rare ones.
    phase_a = _greedy_cover(vid_list, class_elements, class_weights, n_target,
                            already_selected=set(), desc="phase A: class coverage")

    # Phase B: spend the remaining budget maximizing rare-triplet coverage.
    remaining_budget = max(0, n_target - len(phase_a))
    phase_b = _greedy_cover(vid_list, lambda v: summaries[v]["triplets"], trip_w,
                            remaining_budget, already_selected=phase_a,
                            desc="phase B: triplet coverage")

    selected = phase_a + phase_b

    # Fill any leftover budget with the relation-densest remaining videos.
    if len(selected) < n_target:
        chosen = set(selected)
        rest = sorted((v for v in vid_list if v not in chosen),
                      key=lambda v: (-summaries[v]["n_rels"], v))
        selected += rest[:n_target - len(selected)]
        rng.shuffle(selected)  # order is irrelevant downstream; keep it non-obvious

    selected = sorted(selected)
    report = {
        "n_target": n_target,
        "n_selected": len(selected),
        "phase_a_picks": len(phase_a),
        "phase_b_picks": len(phase_b),
        "verbs_covered": len({e for v in selected for e in summaries[v]["verbs"]}),
        "verbs_total": len(verb_freq),
        "objects_covered": len({e for v in selected for e in summaries[v]["objects"]}),
        "objects_total": len(obj_freq),
        "triplets_covered": len({t for v in selected for t in summaries[v]["triplets"]}),
        "triplets_total": len(trip_freq),
        "missing_verbs": sorted(set(verb_freq) - {e for v in selected for e in summaries[v]["verbs"]}),
        "missing_objects": sorted(set(obj_freq) - {e for v in selected for e in summaries[v]["objects"]}),
    }
    return selected, report


# --------------------------------------------------------------------------------------
# Materialize the subset
# --------------------------------------------------------------------------------------

def copy_subset(root_dir, src, dst, split, selected, summaries, copy=True):
    """Copy the selected videos, their annotations and the class lists into ``dst``."""
    src_dir = os.path.join(root_dir, src)
    dst_dir = os.path.join(root_dir, dst)

    video_src = os.path.join(src_dir, "videos")
    video_dst = os.path.join(dst_dir, "videos")
    anno_src = os.path.join(src_dir, "annotations", split)
    anno_dst = os.path.join(dst_dir, "annotations", split)

    total_bytes = 0
    if copy:
        os.makedirs(video_dst, exist_ok=True)
        os.makedirs(anno_dst, exist_ok=True)

    for vid in tqdm(selected, desc="copy %s" % split):
        src_mp4 = os.path.join(video_src, vid + ".mp4")
        src_json = os.path.join(anno_src, vid + ".json")
        if not os.path.isfile(src_mp4):
            print("[warn] missing video for %s, skipping" % vid)
            continue
        total_bytes += os.path.getsize(src_mp4)
        if copy:
            shutil.copy2(src_mp4, os.path.join(video_dst, vid + ".mp4"))
            shutil.copy2(src_json, os.path.join(anno_dst, vid + ".json"))

    if copy:
        for name in ["action.txt", "obj.txt", "rel.txt", "spatial.txt"]:
            src_txt = os.path.join(src_dir, name)
            if os.path.isfile(src_txt):
                shutil.copy2(src_txt, os.path.join(dst_dir, name))

    return total_bytes


def main():
    parser = argparse.ArgumentParser("Build a small coverage-stratified VidOR subset")
    parser.add_argument("--root_dir", default=".", type=str,
                        help="directory containing <src>/ and <dst>/ (default: CWD)")
    parser.add_argument("--src", default="vidor", type=str, help="source dataset dir name")
    parser.add_argument("--dst", default="vidorsmall", type=str, help="destination dataset dir name")
    parser.add_argument("--train_ratio", default=0.125, type=float)
    parser.add_argument("--val_ratio", default=0.125, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=None, type=int,
                        help="scan at most N annotations per split (for quick dry runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the selection without copying anything")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src_dir = os.path.join(args.root_dir, args.src)
    assert os.path.isdir(src_dir), "source dataset dir %s not found" % src_dir

    action_dict, obj_dict = load_class_dicts(src_dir)
    print("[info] %d verbs, %d objects in class lists" % (len(action_dict), len(obj_dict)))

    full_report = {"src": args.src, "dst": args.dst, "seed": args.seed,
                   "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
                   "dry_run": args.dry_run, "splits": {}}

    for split, ratio in [("train", args.train_ratio), ("val", args.val_ratio)]:
        anno_dir = os.path.join(src_dir, "annotations", split)
        summaries = scan_split(anno_dir, action_dict, obj_dict, limit=args.limit)
        if not summaries:
            print("[warn] no annotations found in %s" % anno_dir)
            continue

        n_target = max(1, int(round(len(summaries) * ratio)))
        print("[info] %s: %d videos scanned, target %d" % (split, len(summaries), n_target))

        selected, report = select_videos(summaries, n_target, rng)
        report["n_available"] = len(summaries)
        report["ratio"] = ratio

        n_bytes = copy_subset(args.root_dir, args.src, args.dst, split, selected,
                              summaries, copy=not args.dry_run)
        report["video_bytes"] = n_bytes
        full_report["splits"][split] = report

        print("[info] %s: selected %d videos (%.2f GB), verbs %d/%d, objects %d/%d, "
              "triplets %d/%d" % (
                  split, report["n_selected"], n_bytes / 1e9,
                  report["verbs_covered"], report["verbs_total"],
                  report["objects_covered"], report["objects_total"],
                  report["triplets_covered"], report["triplets_total"]))
        if report["missing_verbs"] or report["missing_objects"]:
            print("[warn] %s missing verbs=%s objects=%s" % (
                split, report["missing_verbs"], report["missing_objects"]))

    if not args.dry_run:
        out = os.path.join(args.root_dir, args.dst, "selection_report.json")
        with open(out, "w") as f:
            json.dump(full_report, f, indent=2)
        print("[info] wrote %s" % out)


if __name__ == "__main__":
    main()
