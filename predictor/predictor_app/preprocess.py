"""Image preprocessing for the door-hanging safety classifier.

Matches the training pipeline exactly:
    1. EXIF transpose + RGB conversion
    2. Standardize: resize short side to 512, center-crop to 512x512
    3. For the model: resize to 128x128, grayscale, HOG features
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener
from skimage.color import rgb2gray
from skimage.feature import hog

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True


STANDARDIZE_SIZE = 512
HOG_SIZE = 128
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

ImageInput = Union[Image.Image, Path, str, bytes, BytesIO]


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image))
    if isinstance(image, BytesIO):
        return Image.open(image)
    return Image.open(Path(image))


def standardize_image(image: ImageInput, size: int = STANDARDIZE_SIZE) -> Image.Image:
    """Mirror of `model/preprocess.py` standardize step.

    EXIF-orient → RGB → resize short side to `size` → center crop `size x size`.
    """
    img = _to_pil(image)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_for_model(image: ImageInput) -> np.ndarray:
    """Standardize + extract HOG features as a 1xN float32 feature vector.

    Output matches the feature space the joblib models were trained on.
    """
    std = standardize_image(image)
    small = std.resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    gray = rgb2gray(arr)
    feats = hog(gray, **HOG_PARAMS).astype(np.float32)
    return feats.reshape(1, -1)
