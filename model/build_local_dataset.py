"""Reconstruct the exact training dataset LOCALLY from the annotator images + COCO.

The COCO JSON gives ground-truth labels + filenames but no pixels. The Bulk-ZIP
download gives the pixels. This script combines them to rebuild
``model/Preprocessed/{vehicle}/{class}/{stem}.png`` exactly as it existed on the
cluster, so train/val/test error can be measured on THIS laptop with the same
70/15/15 split the models were trained on.

Mapping (verified against coco_export.json + split_70_15_15.json):
  - image label = 'unsafe' if it has any 'unsafe' box else 'safe' (license ignored)
  - canonical path = {vehicle}/{positive if unsafe else negative}/{stem}.png
  - this resolves all 11 same-id-in-two-folders collisions with 0 clashes and
    matches all 385 split entries.

Usage:
    python build_local_dataset.py --images-root /path/to/unzipped_images \
        [--coco ../coco_export.json]

Then:
    python export_metrics.py        # now runs on the local dataset

It backs up the committed (cluster-path) split to split_70_15_15.cluster.json
and writes a local-path split so du.get_partition() works here. Don't commit the
rewritten split / the Preprocessed images — they're local scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import du

# Reuse the canonical standardize step (EXIF -> RGB -> resize short side 512 -> center crop).
_PRED = du.ROOT.parent / "predictor"
if str(_PRED) not in sys.path:
    sys.path.insert(0, str(_PRED))
from predictor_app.preprocess import standardize_image  # noqa: E402

REPO_ROOT = du.ROOT.parent


def _labels_from_coco(coco: dict):
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    boxes = defaultdict(set)
    for a in coco["annotations"]:
        boxes[a["image_id"]].add(cats[a["category_id"]])
    out = {}  # original file_name -> (label_str, canonical_rel)
    for im in coco["images"]:
        fn = im["file_name"]
        label = "unsafe" if "unsafe" in boxes.get(im["id"], set()) else "safe"
        vehicle = fn.split("/")[0]
        stem = fn.split("/")[-1]
        cls = "positive" if label == "unsafe" else "negative"
        out[fn] = (label, f"{vehicle}/{cls}/{stem}")
    return out


def _index_images(root: Path):
    """basename -> list of full paths, for fallback matching."""
    idx = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            idx[p.name].append(p)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True,
                    help="folder of unzipped annotator images (Bulk ZIP)")
    ap.add_argument("--coco", default=str(REPO_ROOT / "coco_export.json"))
    args = ap.parse_args()

    images_root = Path(args.images_root).expanduser().resolve()
    coco = json.loads(Path(args.coco).read_text())
    fn_to = _labels_from_coco(coco)
    basename_idx = _index_images(images_root)

    # partition lookup keyed by canonical rel path (suffix after Preprocessed/)
    cluster_split = json.loads((du.ROOT / "split_70_15_15.json").read_text())
    rel_to_part = {k.split("Preprocessed/")[-1]: v for k, v in cluster_split.items()}

    du.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    local_split = {}
    made, missing = 0, []
    for fn, (label, canonical) in fn_to.items():
        part = rel_to_part.get(canonical)
        if part is None:
            continue  # the 1 no-box image dropped from the 385 split
        # locate source pixels: prefer exact original path, else unique basename
        src = images_root / fn
        if not src.exists():
            cands = basename_idx.get(Path(fn).name, [])
            if len(cands) == 1:
                src = cands[0]
            else:
                missing.append(fn)
                continue
        dst = du.DATA_ROOT / canonical
        dst.parent.mkdir(parents=True, exist_ok=True)
        standardize_image(src).save(dst)
        local_split[str(dst.resolve())] = part
        made += 1

    if missing:
        print(f"[warn] {len(missing)} images not found under {images_root}:")
        for m in missing[:10]:
            print("   ", m)

    # back up cluster split once, then write local-path split for du
    sp = du.ROOT / "split_70_15_15.json"
    backup = du.ROOT / "split_70_15_15.cluster.json"
    if not backup.exists():
        backup.write_text(sp.read_text())
        print(f"backed up cluster split -> {backup.name}")
    sp.write_text(json.dumps(local_split, indent=2))

    from collections import Counter
    print(f"reconstructed {made} images into {du.DATA_ROOT}")
    print("partition:", dict(Counter(local_split.values())))
    print("now run:  python export_metrics.py")


if __name__ == "__main__":
    main()
