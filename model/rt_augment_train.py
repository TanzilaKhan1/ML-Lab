"""Offline 'A-to-Z' augmentation of the TRAIN split ONLY (no data leakage).

For every TRAIN original it writes N label-preserving augmented copies as
``{stem}_aug{k}.png`` into BOTH:
  - ``repo/model/Preprocessed/{vehicle}/{class}/`` (so every du-based trainer
    picks them up: du.source_stem strips ``_aug\\d+`` -> the copy groups with its
    source -> always stays in TRAIN), and
  - ``data/dataset_split/train/{safe,unsafe}/`` (human-browsable).

Augmentation is class-balanced: more copies for the minority UNSAFE class so the
augmented train set is ~50/50. VAL and TEST are never touched (kept as real
originals) so evaluation stays honest.

Transforms (label-preserving, A-Z): horizontal flip; affine
(shift/scale/rotate/shear); perspective; random-resized crop; brightness/
contrast; gamma; hue-sat-value; RGB shift; CLAHE; color jitter; sharpen/emboss;
blur (motion/gauss/median/defocus); noise (gauss/ISO/multiplicative); JPEG
compression; downscale; coarse dropout (cutout); weather (shadow/fog/sunflare);
occasional grayscale.

Run from repo/model:  python rt_augment_train.py [--per-safe 3 --per-unsafe 7]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

import du

PROJECT = du.ROOT.parent.parent
PHYS = PROJECT / "data" / "dataset_split"
SIZE = 512
CLASS_OF = {"negative": "safe", "positive": "unsafe"}


def build_aug() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Affine(scale=(0.9, 1.12), translate_percent=(0.0, 0.08),
                     rotate=(-15, 15), shear=(-8, 8), p=1.0),
            A.Perspective(scale=(0.02, 0.08), p=1.0),
            A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.78, 1.0),
                                ratio=(0.8, 1.25), p=1.0),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.28, contrast_limit=0.28, p=1.0),
            A.RandomGamma(gamma_limit=(70, 140), p=1.0),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06, p=1.0),
        ], p=0.85),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=14, sat_shift_limit=28, val_shift_limit=22, p=1.0),
            A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18, p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.55),
        A.OneOf([
            A.Sharpen(p=1.0),
            A.Emboss(p=1.0),
        ], p=0.20),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Defocus(radius=(3, 6), p=1.0),
        ], p=0.30),
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.18), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),
        ], p=0.30),
        A.OneOf([
            A.ImageCompression(quality_range=(40, 85), p=1.0),
            A.Downscale(scale_range=(0.5, 0.85), p=1.0),
        ], p=0.25),
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomSunFlare(src_radius=120, p=1.0),
        ], p=0.15),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(0.05, 0.18),
                        hole_width_range=(0.05, 0.18), p=0.30),
        A.ToGray(p=0.05),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-safe", type=int, default=3, help="aug copies per SAFE train image")
    ap.add_argument("--per-unsafe", type=int, default=7, help="aug copies per UNSAFE train image")
    ap.add_argument("--seed", type=int, default=du.SEED)
    args = ap.parse_args()

    split = json.loads(du.SPLIT_FILE.read_text())
    aug = build_aug()

    # train originals (no _aug yet); group by class
    train_origs = [Path(p) for p, part in split.items()
                   if part == "train" and "_aug" not in Path(p).stem]
    per = {"negative": args.per_safe, "positive": args.per_unsafe}

    orig_counts, new_counts = Counter(), Counter()
    made = 0
    for p in train_origs:
        tclass = p.parent.name                 # negative / positive
        cls = CLASS_OF[tclass]                  # safe / unsafe
        vehicle = p.parent.parent.name
        orig_counts[cls] += 1
        n = per.get(tclass, 3)
        base = np.asarray(Image.open(p).convert("RGB"))
        for k in range(n):
            s = (hash((p.stem, k)) ^ args.seed) & 0x7FFFFFFF
            random.seed(s); np.random.seed(s)
            out = aug(image=base)["image"]
            if out.shape[:2] != (SIZE, SIZE):
                out = np.asarray(Image.fromarray(out).resize((SIZE, SIZE), Image.LANCZOS))
            im = Image.fromarray(out)
            aug_stem = f"{p.stem}_aug{k}"
            # 1) Preprocessed (consumed by trainers via du)
            prep_path = p.parent / f"{aug_stem}.png"
            im.save(prep_path, format="PNG", optimize=True)
            split[str(prep_path)] = "train"     # explicitly TRAIN -> no leakage
            # 2) physical browsable folder
            im.save(PHYS / "train" / cls / f"{vehicle}__{aug_stem}.png", format="PNG", optimize=True)
            new_counts[cls] += 1
            made += 1

    du.SPLIT_FILE.write_text(json.dumps(split, indent=2))

    print(f"generated {made} augmented TRAIN images")
    print("original train per class:", dict(orig_counts))
    print("augmented copies per class:", dict(new_counts))
    total = {c: orig_counts[c] + new_counts[c] for c in orig_counts}
    print("TOTAL train per class (orig+aug):", total)

    # sanity: confirm no aug leaked into val/test and du sees the enlarged train
    part = du.get_partition()
    from collections import Counter as C
    for name in ("train", "val", "test"):
        ys = part[name][1]
        c = C(ys)
        n_aug = sum(1 for pth in part[name][0] if "_aug" in Path(pth).stem)
        print(f"  du {name:5s}: {len(ys):4d}  safe={c[0]:4d} unsafe={c[1]:4d}  (aug files here: {n_aug})")


if __name__ == "__main__":
    main()
