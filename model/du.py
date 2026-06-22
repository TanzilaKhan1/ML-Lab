"""Shared data utilities for the annotation-labelled safe/unsafe classifier.

Label convention (matches predictor + scan_dataset):
    0 = negative = SAFE   (no passenger hanging)
    1 = positive = UNSAFE (passenger hanging on door)

The image-level labels come from the *annotations* (ground truth), already
baked into the Merged/Preprocessed folder structure by organize_dataset.py.

One canonical 70/15/15 STRATIFIED split is built once and cached to JSON so
every model (classical + CNN + ResNet) is trained and evaluated on the exact
same partitions — a fair comparison.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "Preprocessed"
SPLIT_FILE = ROOT / "split_70_15_15.json"

SEED = 42
CLASS_NAMES = ["negative", "positive"]          # index = label
CLASS_DISPLAY = {0: "safe", 1: "unsafe"}
EXTS = {".png", ".jpg", ".jpeg", ".webp"}

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def source_stem(stem: str) -> str:
    """Strip augmentation suffixes so augmented copies group with their source.

    Originals look like 'IMG_3294'; offline-augmented copies (if any) like
    'IMG_3294_aug3'. Group key = the original stem.
    """
    return re.sub(r"_aug\d+$", "", stem)


def scan_dataset(root: Path = DATA_ROOT):
    """Walk root/{vehicle}/{class}/* -> (paths, labels, groups)."""
    paths, labels, groups = [], [], []
    for vehicle_dir in sorted(Path(root).iterdir()):
        if not vehicle_dir.is_dir():
            continue
        for cls_dir in sorted(vehicle_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            if cls_name not in CLASS_NAMES:
                continue
            label = CLASS_NAMES.index(cls_name)
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in EXTS and p.is_file():
                    paths.append(p)
                    labels.append(label)
                    groups.append(f"{vehicle_dir.name}/{source_stem(p.stem)}")
    return paths, labels, groups


def build_split(seed: int = SEED):
    """Create a reproducible 70/15/15 stratified split keyed by image path.

    Stratify by class label. Returns dict path -> 'train'|'val'|'test' and
    writes it to SPLIT_FILE.
    """
    paths, labels, _ = scan_dataset()
    idx = list(range(len(paths)))
    # first carve out test (15%)
    rest_idx, test_idx = train_test_split(
        idx, test_size=TEST_FRAC, stratify=labels, random_state=seed)
    # from the remaining 85%, carve out val so val is 15% of the *whole*
    rest_labels = [labels[i] for i in rest_idx]
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_rel, stratify=rest_labels, random_state=seed)

    assign = {}
    for i in train_idx:
        assign[str(paths[i])] = "train"
    for i in val_idx:
        assign[str(paths[i])] = "val"
    for i in test_idx:
        assign[str(paths[i])] = "test"

    SPLIT_FILE.write_text(json.dumps(assign, indent=2))
    return assign


def load_split():
    if not SPLIT_FILE.exists():
        return build_split()
    return json.loads(SPLIT_FILE.read_text())


def get_partition(split=None):
    """Return dict part -> (paths, labels) for 'train'/'val'/'test'."""
    if split is None:
        split = load_split()
    paths, labels, _ = scan_dataset()
    out = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for p, y in zip(paths, labels):
        part = split.get(str(p))
        if part is None:
            continue
        out[part][0].append(p)
        out[part][1].append(y)
    return out


if __name__ == "__main__":
    assign = build_split()
    paths, labels, _ = scan_dataset()
    from collections import Counter
    part = get_partition(assign)
    print(f"total images: {len(paths)} | safe(neg): {labels.count(0)} | unsafe(pos): {labels.count(1)}")
    for name in ("train", "val", "test"):
        ps, ys = part[name]
        c = Counter(ys)
        pct = 100 * len(ys) / len(paths)
        print(f"  {name:5s}: {len(ys):3d} ({pct:4.1f}%)  safe={c[0]:3d}  unsafe={c[1]:3d}")
