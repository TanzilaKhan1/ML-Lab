"""Build box-supervised crops from the annotation boxes.

For each labelled image:
  - UNSAFE: zoomed crop(s) around the union of `unsafe` boxes (the hanging
    passenger), padded for context. Teaches the model the hanger appearance at
    the scale that tiling presents at inference -> helps tiny/distant hangers.
  - SAFE: crop around the `safe`/door boxes (door region WITHOUT a hanger), else
    a centre crop. Teaches "door region, nobody hanging" so tiling doesn't
    false-positive on every door.

Coords are in annotation space (imageWidth/imageHeight); scaled to the raw image,
clamped to frame, rotation ignored. Outputs Crops/{vehicle}/{neg|pos}/{stem}__c{N}.png
and crops_index.json mapping crop -> {source_stem, vehicle, label}.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = Path(__file__).parent
PROJECT = MODEL_DIR.parent.parent              # ML_lab/
DATA = PROJECT / "data"
RAW = DATA / "raw_dl"
ANN = DATA / "_annotations"
OUT = MODEL_DIR / "Crops"
LABEL_MAP = json.loads((DATA / "label_map.json").read_text())

# label -> train class folder (negative=safe, positive=unsafe)
TRAIN_CLASS = {"safe": "negative", "unsafe": "positive"}


def union_box(boxes, sx, sy, W, H):
    """Union of axis-aligned boxes in raw px, clamped to frame. None if empty."""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for b in boxes:
        if b.get("width") is None or b.get("x") is None:
            continue
        x, y, w, h = b["x"] * sx, b["y"] * sy, b["width"] * sx, b["height"] * sy
        xs1.append(x); ys1.append(y); xs2.append(x + w); ys2.append(y + h)
    if not xs1:
        return None
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [x1, y1, x2, y2]


def pad_box(box, W, H, pad, min_frac):
    """Expand box by `pad` fraction each side; enforce a minimum size."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = max(bw * (1 + 2 * pad), W * min_frac)
    bh = max(bh * (1 + 2 * pad), H * min_frac)
    nx1, ny1 = max(0, cx - bw / 2), max(0, cy - bh / 2)
    nx2, ny2 = min(W, cx + bw / 2), min(H, cy + bh / 2)
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def main():
    index = {}
    n_crops = 0
    counts = {"safe": 0, "unsafe": 0}
    for rec in LABEL_MAP:
        stem, vehicle, label = rec["stem"], rec["vehicle"], rec["label"]
        folder_class = rec["folder_class"]
        raw = RAW / vehicle / folder_class / f"{stem}.png"
        ann = ANN / vehicle / folder_class / f"{stem}.json"
        if not raw.exists() or not ann.exists():
            continue
        d = json.loads(ann.read_text())
        annW = d.get("imageWidth") or 0
        annH = d.get("imageHeight") or 0
        with Image.open(raw) as im:
            im = im.convert("RGB")
            W, H = im.size
            sx = W / annW if annW else 1.0
            sy = H / annH if annH else 1.0
            boxes = d.get("annotations") or []
            if label == "unsafe":
                tgt = [b for b in boxes if b.get("label", "").lower() == "unsafe"]
                pads = [0.6, 1.2]            # tight + medium context
                min_frac = 0.22
            else:
                tgt = [b for b in boxes if b.get("label", "").lower() in ("safe", "license")]
                pads = [0.8]
                min_frac = 0.35
            ub = union_box(tgt, sx, sy, W, H)
            crops = []
            if ub is not None:
                for p in pads:
                    crops.append(pad_box(ub, W, H, p, min_frac))
            else:
                # fallback: centre crop (covers door region generically)
                cw, ch = int(W * 0.7), int(H * 0.85)
                x1 = (W - cw) // 2; y1 = (H - ch) // 2
                crops.append((x1, y1, x1 + cw, y1 + ch))
            out_dir = OUT / vehicle / TRAIN_CLASS[label]
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, box in enumerate(crops):
                crop = im.crop(box)
                if crop.size[0] < 8 or crop.size[1] < 8:
                    continue
                outp = out_dir / f"{stem}__c{i}.png"
                crop.save(outp, format="PNG")
                index[str(outp)] = {"source_stem": stem, "vehicle": vehicle, "label": label}
                n_crops += 1
                counts[label] += 1
    (MODEL_DIR / "crops_index.json").write_text(json.dumps(index, indent=2))
    print(f"made {n_crops} crops -> {OUT}")
    print(f"  safe crops: {counts['safe']} | unsafe crops: {counts['unsafe']}")


if __name__ == "__main__":
    main()
