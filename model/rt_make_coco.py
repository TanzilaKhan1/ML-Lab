"""Regenerate repo/coco_export.json from the fresh per-image annotation JSONs
(432 images) so the tracked COCO matches the current dataset.

Categories: license=1, safe=2, unsafe=3 (same ids as the old export).
Each annotation file = one image. bbox/area derived per annotation type:
  - type 'bbox'             -> [x,y,width,height] (+ angle kept in attributes)
  - type 'polyline'/'polygon' -> axis-aligned bbox of its points (+ points kept)
Image width/height read from the original raw image.
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJ = Path("/home/hpc4090/asif_tanzila/ML_lab")
ANN = PROJ / "data" / "_annotations"
RAW = PROJ / "data" / "raw_dl"
OUT = PROJ / "repo" / "coco_export.json"
CATS = [{"id": 1, "name": "license", "supercategory": "none"},
        {"id": 2, "name": "safe", "supercategory": "none"},
        {"id": 3, "name": "unsafe", "supercategory": "none"}]
CAT_ID = {"license": 1, "safe": 2, "unsafe": 3}
EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]


def raw_dims(vehicle, folder, stem):
    base = RAW / vehicle / folder
    for e in EXTS:
        p = base / f"{stem}{e}"
        if p.exists():
            with Image.open(p) as im:
                return im.size  # (w,h)
    hits = list((RAW / vehicle).rglob(f"{stem}.*"))
    if hits:
        with Image.open(hits[0]) as im:
            return im.size
    return (None, None)


def ann_bbox(a):
    t = a.get("type")
    if t == "bbox":
        return [a.get("x", 0.0), a.get("y", 0.0), a.get("width", 0.0), a.get("height", 0.0)]
    pts = a.get("points") or []
    if pts:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return [0.0, 0.0, 0.0, 0.0]


def main():
    images, annotations = [], []
    img_id, ann_id = 0, 0
    skipped = 0
    files = sorted(p for p in ANN.rglob("*.json")
                   if not p.name.startswith("_") and "/_trash/" not in str(p))
    for f in files:
        d = json.loads(f.read_text())
        fn = d.get("filename")
        if not fn:
            parts = f.relative_to(ANN).parts  # vehicle/folder/stem.json
            fn = f"{parts[-3]}/{parts[-2]}/{f.stem}.png"
        vehicle, folder, name = fn.split("/")
        stem = Path(name).stem
        w, h = raw_dims(vehicle, folder, stem)
        if w is None:
            skipped += 1
            continue
        img_id += 1
        images.append({"id": img_id, "file_name": fn, "width": w, "height": h})
        for a in d.get("annotations", []) or []:
            lab = (a.get("label") or "").lower()
            if lab not in CAT_ID:
                continue
            bbox = ann_bbox(a)
            ann_id += 1
            entry = {"id": ann_id, "image_id": img_id, "category_id": CAT_ID[lab],
                     "iscrowd": 0, "bbox": [round(v, 4) for v in bbox],
                     "area": round(bbox[2] * bbox[3], 4),
                     "attributes": {"type": a.get("type")}}
            if a.get("type") == "bbox" and a.get("angle"):
                entry["attributes"]["angle"] = a["angle"]
            if a.get("points"):
                entry["attributes"]["points"] = a["points"]
            annotations.append(entry)

    coco = {"images": images, "annotations": annotations, "categories": CATS}
    OUT.write_text(json.dumps(coco, indent=1))
    from collections import Counter
    bycat = Counter(x["category_id"] for x in annotations)
    print(f"wrote {OUT}")
    print(f"  images={len(images)}  annotations={len(annotations)}  skipped(no raw)={skipped}")
    print(f"  by category: license={bycat[1]} safe={bycat[2]} unsafe={bycat[3]}")


if __name__ == "__main__":
    main()
