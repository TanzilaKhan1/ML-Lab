"""Load PyTorch classifier weights for inference (no retraining).

Supports the original ``cnn``/``resnet`` checkpoints AND the newer generic
checkpoints that carry their own ``backbone`` name + tuned decision
``threshold`` (produced by ``model/train_cv.py`` / ``train_torch2.py``).

A checkpoint is a dict:
  {model: state_dict, classes: [...], img_size: int,
   backbone: "resnet50"|"convnext_tiny"|"efficientnet_b0"|"resnet18"|"smallcnn",
   threshold: float (P(unsafe) cut-off), tta: bool}

``backbone``/``threshold``/``tta`` are optional — old checkpoints still load.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def _resolve_device(device):
    """Pick inference device. env PREDICTOR_DEVICE in {auto,cuda,cpu} (default auto)."""
    if device is not None:
        return torch.device(device)
    pref = os.environ.get("PREDICTOR_DEVICE", "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_threshold(ck: dict) -> float:
    """Choose the active threshold by operating mode (env PREDICTOR_OP_MODE).

    balanced     -> max accuracy
    high_recall  -> catch >=95% of unsafe (default; safety-leaning)
    max_recall   -> zero-miss, 100% unsafe recall (flags more, lower accuracy)
    """
    mode = os.environ.get("PREDICTOR_OP_MODE", "high_recall").lower()
    key = {"balanced": "threshold_balanced",
           "high_recall": "threshold_high_recall",
           "max_recall": "threshold_max_recall"}.get(mode, "threshold_high_recall")
    return float(ck.get(key, ck.get("threshold", 0.5)))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models as tvm
from torchvision import transforms

from .config import MODEL_DIR

_TRAIN_SCRIPT_DIR = Path(MODEL_DIR).resolve()
if str(_TRAIN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SCRIPT_DIR))

from train_customised_cnn import SmallCNN  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------- architecture registry ---------------------------
def _build_arch(backbone: str, num_classes: int = 2) -> nn.Module:
    """Instantiate the matching architecture (weights=None — joblib has trained)."""
    if backbone in ("smallcnn", "cnn"):
        return SmallCNN(num_classes=num_classes)
    if backbone in ("resnet18", "resnet"):
        m = tvm.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        return m
    if backbone == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
        return m
    if backbone == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4),
                                     nn.Linear(m.classifier[1].in_features, num_classes))
        return m
    if backbone == "convnext_tiny":
        m = tvm.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    raise ValueError(f"Unknown backbone {backbone!r}")


def _eval_transform(backbone: str, img_size: int):
    if backbone in ("smallcnn", "cnn"):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    # all transfer backbones: Resize(1.15x) -> CenterCrop -> Normalize (matches training)
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert("RGB")
    raise TypeError(f"Cannot convert {type(x)} to PIL.Image")


class TorchClassifier:
    """Sklearn-like wrapper. Applies a tuned threshold + optional TTA (hflip)."""

    def __init__(self, model, *, classes, img_size, eval_transform,
                 threshold: float = 0.5, tta: bool = True, device=None):
        self.model = model.eval()
        self.classes_ = list(classes)
        self.img_size = img_size
        self.eval_transform = eval_transform
        self.threshold = float(threshold)
        self.tta = bool(tta)
        self.device = _resolve_device(device)
        self.model.to(self.device)

    def _stack(self, image) -> torch.Tensor:
        if isinstance(image, np.ndarray) and image.ndim == 4:
            t = [self.eval_transform(_to_pil(im)) for im in image]
            return torch.stack(t, 0).to(self.device)
        return self.eval_transform(_to_pil(image)).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_proba(self, image) -> np.ndarray:
        x = self._stack(image)
        p = torch.softmax(self.model(x), 1)
        if self.tta:
            p2 = torch.softmax(self.model(torch.flip(x, dims=[3])), 1)
            p = (p + p2) / 2
        return p.cpu().numpy()

    def predict(self, image) -> np.ndarray:
        proba = self.predict_proba(image)
        return (proba[:, 1] >= self.threshold).astype(int)


class EnsembleClassifier:
    """Averages predict_proba across members; applies its own tuned threshold."""

    def __init__(self, members: Sequence[TorchClassifier], *, classes,
                 threshold: float = 0.5):
        self.members = list(members)
        self.classes_ = list(classes)
        self.threshold = float(threshold)
        self.img_size = self.members[0].img_size if self.members else 224

    def predict_proba(self, image) -> np.ndarray:
        return np.mean([m.predict_proba(image) for m in self.members], axis=0)

    def predict(self, image) -> np.ndarray:
        return (self.predict_proba(image)[:, 1] >= self.threshold).astype(int)


def load_torch_checkpoint(path, kind: str | None = None, device=None) -> TorchClassifier:
    import joblib
    ckpt = joblib.load(path)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Not a torch checkpoint dict: {path}")
    classes = ckpt.get("classes", ["negative", "positive"])
    backbone = ckpt.get("backbone") or (kind if kind in ("cnn", "resnet") else "resnet")
    default_img = 128 if backbone in ("cnn", "smallcnn") else 224
    img_size = int(ckpt.get("img_size", default_img))
    threshold = _pick_threshold(ckpt)
    tta = bool(ckpt.get("tta", True))
    net = _build_arch(backbone, num_classes=len(classes))
    net.load_state_dict(ckpt["model"])
    return TorchClassifier(net, classes=classes, img_size=img_size,
                           eval_transform=_eval_transform(backbone, img_size),
                           threshold=threshold, tta=tta, device=device)


def load_ensemble(spec_path, device=None) -> EnsembleClassifier:
    """spec_path -> JSON {members: [filename...], threshold: float} in MODEL_DIR."""
    import json
    spec = json.loads(Path(spec_path).read_text())
    members = [load_torch_checkpoint(Path(MODEL_DIR) / f, device=device)
               for f in spec["members"]]
    classes = members[0].classes_ if members else ["negative", "positive"]
    return EnsembleClassifier(members, classes=classes,
                              threshold=_pick_threshold(spec))
