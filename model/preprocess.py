"""Standardize all dataset images to PNG, fixed resolution.

Walks Merged/{bus,leguna}/{positive,negative}, converts each image to RGB,
resizes short side to TARGET then center-crops to TARGET x TARGET, saves as PNG
in Preprocessed/ mirroring the source structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).parent
SRC = ROOT / "Merged"
DST = ROOT / "Preprocessed"
TARGET = 512
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tiff"}


def standardize(img: Image.Image, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1

    files: list[Path] = []
    for vehicle in sorted(SRC.iterdir()):
        if not vehicle.is_dir():
            continue
        for cls in sorted(vehicle.iterdir()):
            if not cls.is_dir():
                continue
            for p in cls.iterdir():
                if p.is_file() and p.suffix.lower() in EXTS:
                    files.append(p)

    print(f"found {len(files)} images under {SRC}")
    ok, skipped, failed = 0, 0, 0
    for src_path in tqdm(files, desc="preprocess"):
        rel = src_path.relative_to(SRC)
        out_dir = DST / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (src_path.stem + ".png")
        if out_path.exists():
            skipped += 1
            continue
        try:
            with Image.open(src_path) as img:
                std = standardize(img, TARGET)
                std.save(out_path, format="PNG", optimize=True)
            ok += 1
        except (UnidentifiedImageError, OSError, ValueError) as e:
            failed += 1
            print(f"FAIL {src_path}: {e}", file=sys.stderr)

    print(f"\ndone | ok={ok} skipped={skipped} failed={failed}")
    for vehicle in sorted(DST.iterdir()):
        if not vehicle.is_dir():
            continue
        for cls in sorted(vehicle.iterdir()):
            if cls.is_dir():
                n = sum(1 for _ in cls.iterdir())
                print(f"  {vehicle.name}/{cls.name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
