"""Data utilities used by ``train_resnet.py``.

Kept as a standalone module so the training scripts can be *imported* (for
their ``nn.Module`` definitions) without needing the training pipeline to be
runnable. Inference code in ``predictor_app/torch_models.py`` only needs the
import to succeed — it never calls ``scan_dataset`` / ``stratified_split``.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ["negative", "positive"]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def scan_dataset(root: Path):
    """Walk ``root/{vehicle}/{class}/*`` and return (paths, labels)."""
    paths: list[Path] = []
    labels: list[int] = []
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
            for p in cls_dir.iterdir():
                if p.suffix.lower() in EXTS and p.is_file():
                    paths.append(p)
                    labels.append(label)
    return paths, labels


def stratified_split(labels, test_size: float, seed: int):
    idx = list(range(len(labels)))
    return train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
