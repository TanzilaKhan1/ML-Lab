"""Rebuild the safe/unsafe dataset from the freshly-synced R2 raw images +
annotation-derived labels, then create a leak-free 70/15/15 split.

Steps
-----
1. Read ``data/label_map.json`` (annotation-derived ground truth; produced by
   ``build_labels.py``). Label rule: UNSAFE if any 'unsafe' box else SAFE.
2. Locate each raw image under ``data/raw_dl/{vehicle}/{folder_class}/`` by stem
   regardless of extension (png/jpg/...), EXIF-orient + standardize to 512x512,
   and write the canonical original to
   ``repo/model/Preprocessed/{vehicle}/{negative|positive}/{stem}.png``
   (negative=safe, positive=unsafe). Stem collisions are disambiguated.
3. Build a 4-way STRATIFIED (vehicle x class) 70/15/15 split so val and test
   each contain *every* category (bus-safe, bus-unsafe, legua-safe,
   legua-unsafe). The split is keyed by the exact absolute path strings that
   ``du.scan_dataset()`` produces, so every downstream trainer
   (``du.get_partition()``) honours it. No augmented files exist yet => no leak.
4. Materialise human-browsable physical folders:
   ``data/dataset_split/{train,val,test}/{safe,unsafe}/`` (ORIGINALS only;
   augmentation is added later by ``rt_augment_train.py`` to TRAIN only).

Run from repo/model:  python rt_build_dataset.py
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFile, ImageOps
from sklearn.model_selection import train_test_split

import du

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

PROJECT = du.ROOT.parent.parent                 # /home/hpc4090/asif_tanzila/ML_lab
LABEL_MAP = PROJECT / "data" / "label_map.json"
RAW = PROJECT / "data" / "raw_dl"
PREP = du.DATA_ROOT                             # repo/model/Preprocessed
PHYS = PROJECT / "data" / "dataset_split"       # human-browsable train/val/test
TARGET = 512
SEARCH_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".heic", ".heif"]
CLASS_OF = {"negative": "safe", "positive": "unsafe"}


def standardize(img: Image.Image, size: int = TARGET) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
    nw, nh = img.size
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def find_raw(vehicle: str, folder_class: str, stem: str) -> Path | None:
    base = RAW / vehicle / folder_class
    for ext in SEARCH_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any file with this stem anywhere under the vehicle folder
    hits = [p for p in (RAW / vehicle).rglob(f"{stem}.*") if p.suffix.lower() in SEARCH_EXTS]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def fresh_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def main():
    records = json.loads(LABEL_MAP.read_text())
    print(f"label_map records: {len(records)}")

    # ---- 1+2. rebuild Preprocessed (canonical originals) ----
    fresh_dir(PREP)
    seen: dict[tuple, str] = {}      # (vehicle, train_class, stem) -> source path (collision guard)
    made, missing, collisions = 0, [], 0
    counts = Counter()
    for r in records:
        vehicle, fclass, tclass, stem = r["vehicle"], r["folder_class"], r["train_class"], r["stem"]
        src = find_raw(vehicle, fclass, stem)
        if src is None:
            missing.append(f"{vehicle}/{fclass}/{stem}")
            continue
        key = (vehicle, tclass, stem)
        out_stem = stem
        if key in seen:                # same canonical target from a different source
            collisions += 1
            out_stem = f"{stem}__{fclass[:3]}"
        seen[key] = str(src)
        out_dir = PREP / vehicle / tclass
        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            standardize(im).save(out_dir / f"{out_stem}.png", format="PNG", optimize=True)
        made += 1
        counts[(vehicle, CLASS_OF[tclass])] += 1

    print(f"preprocessed originals: {made} | missing raw: {len(missing)} | stem-collisions disambiguated: {collisions}")
    for m in missing[:10]:
        print("   MISSING", m)
    print("per vehicle x class:", {f"{v}/{c}": n for (v, c), n in sorted(counts.items())})

    # ---- 3. 4-way stratified 70/15/15 split keyed by du paths ----
    paths, labels, _ = du.scan_dataset()                 # absolute path strings, labels 0/1
    paths = [str(p) for p in paths]
    # 4-way stratify key = vehicle + class  (vehicle parsed from canonical path)
    def vehicle_of(p):
        return Path(p).parent.parent.name
    strat = [f"{vehicle_of(p)}_{lab}" for p, lab in zip(paths, labels)]
    idx = list(range(len(paths)))

    rest, test = train_test_split(idx, test_size=du.TEST_FRAC, stratify=strat,
                                  random_state=du.SEED)
    rest_strat = [strat[i] for i in rest]
    val_rel = du.VAL_FRAC / (1.0 - du.TEST_FRAC)
    train, val = train_test_split(rest, test_size=val_rel, stratify=rest_strat,
                                  random_state=du.SEED)
    assign = {}
    for i in train: assign[paths[i]] = "train"
    for i in val:   assign[paths[i]] = "val"
    for i in test:  assign[paths[i]] = "test"
    du.SPLIT_FILE.write_text(json.dumps(assign, indent=2))
    print(f"\nwrote split -> {du.SPLIT_FILE.name}  ({len(assign)} originals)")

    # report split x (vehicle x class)
    by = defaultdict(Counter)
    for i in idx:
        by[assign[paths[i]]][f"{vehicle_of(paths[i])}/{du.CLASS_DISPLAY[labels[i]]}"] += 1
    for part in ("train", "val", "test"):
        tot = sum(by[part].values())
        safe = sum(v for k, v in by[part].items() if k.endswith("safe") and not k.endswith("unsafe"))
        unsafe = sum(v for k, v in by[part].items() if k.endswith("unsafe"))
        print(f"  {part:5s}: {tot:3d}  safe={safe:3d} unsafe={unsafe:3d}  | {dict(sorted(by[part].items()))}")

    # ---- 4. materialise physical train/val/test folders (originals) ----
    fresh_dir(PHYS)
    for part in ("train", "val", "test"):
        for cls in ("safe", "unsafe"):
            (PHYS / part / cls).mkdir(parents=True, exist_ok=True)
    phys_counts = Counter()
    for p, lab in zip(paths, labels):
        part = assign[p]
        cls = du.CLASS_DISPLAY[lab]                       # safe / unsafe
        veh = vehicle_of(p)
        dst = PHYS / part / cls / f"{veh}__{Path(p).name}"
        shutil.copy2(p, dst)
        phys_counts[(part, cls)] += 1
    print(f"\nmaterialised physical split under {PHYS}")
    for (part, cls), n in sorted(phys_counts.items()):
        print(f"  {part}/{cls}: {n}")

    (PHYS / "SPLIT_INFO.json").write_text(json.dumps({
        "total_originals": made,
        "missing_raw": missing,
        "stem_collisions_disambiguated": collisions,
        "split_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "stratified_by": "vehicle x class (4-way)",
        "counts": {f"{part}/{cls}": n for (part, cls), n in sorted(phys_counts.items())},
        "note": "ORIGINALS only. Augmentation added to TRAIN by rt_augment_train.py.",
    }, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
