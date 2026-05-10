"""
Augment images in positive/ and negative/ folders to reach 500-600 each.
Originals are kept as-is; augmented copies are added alongside them.
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

BASE = Path("/Users/asif/Documents/git/Untitled/ML-Lab/Final_Dataset/raw_images")
TARGET = 550  # aim for middle of 500-600
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Augmentation functions — each takes a PIL Image, returns a PIL Image
# ---------------------------------------------------------------------------

def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(deg):
    def fn(img):
        return img.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=None)
    fn.__name__ = f"rot{deg:+d}"
    return fn

def brightness(factor):
    def fn(img):
        return ImageEnhance.Brightness(img).enhance(factor)
    fn.__name__ = f"bright{factor}"
    return fn

def contrast(factor):
    def fn(img):
        return ImageEnhance.Contrast(img).enhance(factor)
    fn.__name__ = f"contrast{factor}"
    return fn

def saturation(factor):
    def fn(img):
        return ImageEnhance.Color(img).enhance(factor)
    fn.__name__ = f"sat{factor}"
    return fn

def sharpness(factor):
    def fn(img):
        return ImageEnhance.Sharpness(img).enhance(factor)
    fn.__name__ = f"sharp{factor}"
    return fn

def blur(radius):
    def fn(img):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    fn.__name__ = f"blur{radius}"
    return fn

def add_noise(img):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def crop_resize(cx_frac, cy_frac, scale=0.8):
    """Crop a scaled region centered at (cx_frac, cy_frac) then resize back."""
    def fn(img):
        w, h = img.size
        cw, ch = int(w * scale), int(h * scale)
        left = max(0, min(int(cx_frac * w - cw / 2), w - cw))
        top  = max(0, min(int(cy_frac * h - ch / 2), h - ch))
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    fn.__name__ = f"crop_{cx_frac}_{cy_frac}"
    return fn

def compose(*fns):
    def fn(img):
        for f in fns:
            img = f(img)
        return img
    fn.__name__ = "_".join(f.__name__ for f in fns)
    return fn


# ---------------------------------------------------------------------------
# Augmentation pipeline — ordered from safest to most aggressive
# ---------------------------------------------------------------------------
AUGMENTATIONS = [
    hflip,
    rotate(10),
    rotate(-10),
    rotate(20),
    rotate(-20),
    brightness(1.3),
    brightness(0.7),
    contrast(1.3),
    contrast(0.7),
    saturation(1.4),
    saturation(0.6),
    sharpness(2.0),
    blur(1.5),
    add_noise,
    crop_resize(0.5, 0.5),          # center crop
    crop_resize(0.35, 0.35),        # top-left crop
    crop_resize(0.65, 0.35),        # top-right crop
    crop_resize(0.35, 0.65),        # bottom-left crop
    crop_resize(0.65, 0.65),        # bottom-right crop
    compose(hflip, brightness(1.3)),
    compose(hflip, brightness(0.7)),
    compose(hflip, contrast(1.3)),
    compose(rotate(10), brightness(1.2)),
    compose(rotate(-10), brightness(0.8)),
    compose(hflip, saturation(1.4)),
    compose(blur(1.0), brightness(1.2)),
    compose(rotate(15), contrast(1.2)),
    compose(rotate(-15), saturation(0.7)),
    compose(hflip, sharpness(2.0)),
    compose(crop_resize(0.5, 0.5), brightness(1.2)),
    compose(crop_resize(0.5, 0.5), hflip),
    compose(add_noise, brightness(1.1)),
]


# ---------------------------------------------------------------------------
# Augment a folder
# ---------------------------------------------------------------------------
def augment_folder(folder: Path, target: int):
    sources = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    # exclude already-augmented files
    sources = [f for f in sources if "_aug" not in f.stem]
    n_orig = len(sources)

    # Remove any previously generated augmented files so we start clean
    for f in folder.glob("*_aug_*.png"):
        f.unlink()

    needed = target - n_orig
    if needed <= 0:
        print(f"  {folder.name}: {n_orig} originals already >= {target}, skipping")
        return

    print(f"  {folder.name}: {n_orig} originals → generating {needed} augmented images")

    generated = 0
    aug_cycle = list(AUGMENTATIONS)
    idx = 0  # which augmentation to apply next

    while generated < needed:
        src = sources[generated % n_orig]
        aug_fn = aug_cycle[idx % len(aug_cycle)]
        idx += 1

        try:
            img = Image.open(src).convert("RGB")
            aug_img = aug_fn(img)
            out_name = f"{src.stem}_aug_{generated+1:04d}.png"
            aug_img.save(folder / out_name, "PNG")
            generated += 1
            if generated % 50 == 0:
                print(f"    {generated}/{needed}")
        except Exception as e:
            print(f"    WARNING: {src.name} with {aug_fn.__name__}: {e}")

    total = len(list(folder.glob("*.png")))
    print(f"  {folder.name}: done — {total} total images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for cls in ("positive", "negative"):
    folder = BASE / cls
    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue
    print(f"\n[{cls}]")
    augment_folder(folder, TARGET)

print("\nAll done.")
