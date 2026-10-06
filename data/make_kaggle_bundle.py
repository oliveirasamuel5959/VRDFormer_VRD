"""Build an upload-ready Kaggle bundle for the small VidOR subset.

Assembles a directory whose *contents* extract directly into ``data/`` of a fresh
checkout, so the Colab/RunPod notebook needs no path juggling:

    <bundle>/
    ├── dataset-metadata.json        (Kaggle dataset metadata, not part of the archive)
    ├── vidorsmall/                  -> data/vidorsmall/
    ├── metadata/                    -> data/metadata/  (vidorsmall_* files)
    └── weights/detr-r101-2c7b67e5.pth -> data/weights/

Videos are hard-linked when the bundle and source live on the same filesystem, so the
bundle costs almost no extra disk; otherwise they are copied.

Usage (from the repo root):

    python data/make_kaggle_bundle.py                       # build the bundle
    python data/make_kaggle_bundle.py --zip                 # also write the .zip
    python data/make_kaggle_bundle.py --name vidorsmall --kaggle_id samuelpatricio/vrdformer-vidor-small
"""
import argparse
import json
import os
import shutil
import sys

METADATA_FILES = [
    "vidorsmall_annotations.pkl",
    "vidorsmall_train_frames_stage1.json",
    "vidorsmall_train_frames_stage2.json",
    "vidorsmall_val_frames.json",
]
WEIGHTS = "detr-r101-2c7b67e5.pth"


def link_or_copy(src, dst):
    """Hard-link src to dst if possible, else copy. Returns 'link' or 'copy'.

    Idempotent: an existing dst is replaced, and an already-linked pair (same inode)
    is left alone rather than raising SameFileError on re-runs.
    """
    if os.path.exists(dst):
        try:
            if os.path.samefile(src, dst):
                return "link"
        except OSError:
            pass
        os.remove(dst)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    parser = argparse.ArgumentParser("Build a Kaggle upload bundle for vidorsmall")
    parser.add_argument("--root", default=".", help="repo root (default: CWD)")
    parser.add_argument("--name", default="vidorsmall", help="dataset dir name to bundle")
    parser.add_argument("--bundle", default=None, help="output bundle dir (default: data/vidor_small_bundle)")
    parser.add_argument("--kaggle_id", default="samuelpatricio/vrdformer-vidor-small")
    parser.add_argument("--title", default="VRDFormer-VidOR-Small")
    parser.add_argument("--zip", action="store_true", help="also create <bundle>.zip")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    data_dir = os.path.join(root, "data")
    name = args.name
    bundle = args.bundle or os.path.join(data_dir, "vidor_small_bundle")

    src_dataset = os.path.join(data_dir, name)
    assert os.path.isdir(src_dataset), "dataset dir %s not found" % src_dataset
    os.makedirs(os.path.join(bundle, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(bundle, "weights"), exist_ok=True)

    # 1. dataset dir (videos + annotations + class lists), hard-linked
    dst_dataset = os.path.join(bundle, name)
    if os.path.exists(dst_dataset):
        shutil.rmtree(dst_dataset)
    os.makedirs(dst_dataset)
    n_link = n_copy = 0
    for sub in ["videos", "annotations"]:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(src_dataset, sub)):
            rel = os.path.relpath(dirpath, src_dataset)
            os.makedirs(os.path.join(dst_dataset, rel), exist_ok=True)
            for fn in filenames:
                how = link_or_copy(os.path.join(dirpath, fn), os.path.join(dst_dataset, rel, fn))
                n_link += how == "link"
                n_copy += how == "copy"
    for fn in os.listdir(src_dataset):
        src = os.path.join(src_dataset, fn)
        if os.path.isfile(src):
            link_or_copy(src, os.path.join(dst_dataset, fn))

    # 2. regenerated metadata
    for fn in METADATA_FILES:
        src = os.path.join(data_dir, "metadata", fn)
        assert os.path.isfile(src), "missing metadata file %s (run prepare.py first)" % src
        link_or_copy(src, os.path.join(bundle, "metadata", fn))

    # 3. DETR weights
    weights_src = os.path.join(data_dir, "weights", WEIGHTS)
    assert os.path.isfile(weights_src), "missing weights %s" % weights_src
    link_or_copy(weights_src, os.path.join(bundle, "weights", WEIGHTS))

    # 4. Kaggle dataset metadata
    with open(os.path.join(bundle, "dataset-metadata.json"), "w") as f:
        json.dump({"title": args.title, "id": args.kaggle_id,
                   "licenses": [{"name": "CC0-1.0"}]}, f, indent=4)

    print("[info] bundle at %s (%d linked, %d copied)" % (bundle, n_link, n_copy))
    print("[info] upload with:  kaggle datasets create -p %s --dir-mode zip" % bundle)

    if args.zip:
        zip_path = bundle.rstrip("/") + ".zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        _zip_bundle(bundle, zip_path)
        print("[info] wrote %s (%.2f GB)" % (zip_path, os.path.getsize(zip_path) / 1e9))


def _zip_bundle(bundle, zip_path):
    """Zip the *contents* of `bundle` for upload, excluding dataset-metadata.json.

    Paths are stored relative to the bundle root (e.g. ``metadata/...``,
    ``vidorsmall/...``), matching ``kaggle datasets create --dir-mode zip``, so the
    archive extracts straight into ``data/``. Kaggle reads dataset-metadata.json
    from the directory itself, so it is left out of the archive.
    """
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for dirpath, _dirnames, filenames in os.walk(bundle):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, bundle)
                if rel == "dataset-metadata.json":
                    continue
                zf.write(full, rel)


if __name__ == "__main__":
    sys.exit(main())
