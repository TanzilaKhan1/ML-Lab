"""
Image Augmentation Script for ML Labs/Accepted
Applies geometric, color/lighting, and blur/noise augmentations
to simulate varied dashcam/surveillance conditions.

Usage:
    python augment_images.py                          # 5 augmentations per image (default)
    python augment_images.py --num-augmentations 10   # 10 augmentations per image

Requirements:
    pip install Pillow pillow-heif albumentations opencv-python numpy
"""

import os
import argparse
import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A

# Support HEIC files
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("WARNING: pillow-heif not installed. HEIC files will be skipped.")
    print("Install with: pip install pillow-heif")

INPUT_DIR = "/Users/asif/Downloads/ML Labs/Accepted"
OUTPUT_DIR = "/Users/asif/Downloads/ML Labs/Augmented"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
if HEIC_SUPPORT:
    SUPPORTED_EXTENSIONS.add(".heic")


def build_augmentation_pipeline():
    """Build the full augmentation pipeline with all three categories."""
    return A.Compose([
        # ── Geometric ──
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(
            height=0.8, width=0.8,       # crop to 80% of image
            p=0.4,
        ) if False else A.OneOf([         # use RandomResizedCrop for scale variation
            A.RandomResizedCrop(
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                size=(640, 640),          # will be resized back later
                p=1.0,
            ),
            A.NoOp(p=1.0),
        ], p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.Affine(
            scale=(0.85, 1.15),           # scale/zoom variation
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            p=0.3,
        ),

        # ── Color / Lighting ──
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=25,
            p=0.5,
        ),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1,
                p=1.0,
            ),
        ], p=0.3),

        # ── Blur & Noise ──
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.15), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        A.ImageCompression(quality_range=(40, 85), p=0.3),  # JPEG artifacts
    ])


def load_image(path):
    """Load image (including HEIC) and return as numpy RGB array."""
    img = Image.open(path)
    img = img.convert("RGB")
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Augment images for ML training")
    parser.add_argument(
        "--num-augmentations", "-n", type=int, default=5,
        help="Number of augmented copies per original image (default: 5)"
    )
    parser.add_argument(
        "--input-dir", type=str, default=INPUT_DIR,
        help=f"Input directory (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Also save original (converted) copies for reference
    originals_dir = os.path.join(args.output_dir, "originals")
    os.makedirs(originals_dir, exist_ok=True)

    transform = build_augmentation_pipeline()

    # Collect all supported images
    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"No supported images found in {args.input_dir}")
        print(f"Supported formats: {SUPPORTED_EXTENSIONS}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Generating {args.num_augmentations} augmentations each → {len(image_files) * args.num_augmentations} augmented images")
    print(f"Output: {args.output_dir}\n")

    total_saved = 0
    for idx, filename in enumerate(image_files, 1):
        filepath = os.path.join(args.input_dir, filename)
        stem = os.path.splitext(filename)[0]

        try:
            img = load_image(filepath)
        except Exception as e:
            print(f"  SKIP {filename}: {e}")
            continue

        original_h, original_w = img.shape[:2]

        # Save original as JPG
        cv2.imwrite(
            os.path.join(originals_dir, f"{stem}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        # Generate augmentations
        for aug_idx in range(args.num_augmentations):
            augmented = transform(image=img)["image"]

            # Resize back to original dimensions if crop/resize changed it
            if augmented.shape[:2] != (original_h, original_w):
                augmented = cv2.resize(augmented, (original_w, original_h))

            out_name = f"{stem}_aug{aug_idx:02d}.jpg"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(
                out_path,
                cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            total_saved += 1

        print(f"  [{idx}/{len(image_files)}] {filename} → {args.num_augmentations} augmentations")

    print(f"\nDone! {total_saved} augmented images saved to {args.output_dir}")
    print(f"Original JPG copies saved to {originals_dir}")


if __name__ == "__main__":
    main()
