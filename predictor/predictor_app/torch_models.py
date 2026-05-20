"""Load the PyTorch CNN / ResNet-18 weights for inference, no retraining.

The saved ``.joblib`` files in ``predictor/model/`` contain only a state_dict
plus metadata. To run a forward pass we must instantiate the matching
architecture and pour the weights in. The architectures already exist in
the sibling training scripts — we just import them.

  ``SmallCNN``    ← imported from ``train_customised_cnn.py``
  ``resnet18``    ← from torchvision; FC head replaced exactly as
                    ``train_resnet.build_model`` does it, but without the
                    ImageNet weight download (the joblib already has trained
                    weights, so downloading the pretrained ones would be wasted
                    network I/O).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from .config import MODEL_DIR

# Make the sibling training scripts importable so we can reuse `SmallCNN`
# verbatim — no architecture duplication, no risk of drift.
_TRAIN_SCRIPT_DIR = Path(MODEL_DIR).resolve()
if str(_TRAIN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SCRIPT_DIR))

from train_customised_cnn import SmallCNN  # noqa: E402  (path mutated above)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_resnet18_head(num_classes: int = 2) -> nn.Module:
    """Same FC swap as ``train_resnet.build_model`` — but no pretrained download."""
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _cnn_eval_transform(img_size: int):
    """Eval branch of ``train_customised_cnn.build_transforms``."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _resnet_eval_transform(img_size: int):
    """Eval branch of ``train_resnet.build_transforms`` (Resize 256 → CenterCrop)."""
    return transforms.Compose([
        transforms.Resize(256),
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
    """Sklearn-like wrapper around a torch ``nn.Module`` classifier.

    Accepts a PIL image (single inference) or an ``(N, H, W, 3) uint8`` numpy
    array (LIME perturbation batch). Preprocessing matches the model's
    training eval transform exactly.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        classes: Sequence[str],
        img_size: int,
        eval_transform,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model.eval()
        self.classes_ = list(classes)
        self.img_size = img_size
        self.eval_transform = eval_transform
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)

    def _stack_tensor(self, image) -> torch.Tensor:
        if isinstance(image, np.ndarray) and image.ndim == 4:
            tensors = [self.eval_transform(_to_pil(im)) for im in image]
            return torch.stack(tensors, dim=0).to(self.device)
        return self.eval_transform(_to_pil(image)).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_proba(self, image) -> np.ndarray:
        x = self._stack_tensor(image)
        logits = self.model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, image) -> np.ndarray:
        return self.predict_proba(image).argmax(axis=1)


def load_torch_checkpoint(path, kind: str) -> TorchClassifier:
    """Load a ``{model, classes, img_size}`` joblib checkpoint into a wrapper.

    ``kind`` ∈ {``"cnn"``, ``"resnet"``} picks the architecture and the
    matching eval-time preprocessing pipeline.
    """
    import joblib

    ckpt = joblib.load(path)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Not a torch checkpoint dict: {path}")

    classes = ckpt.get("classes", ["negative", "positive"])
    img_size = int(ckpt.get("img_size", 128 if kind == "cnn" else 224))
    num_classes = len(classes)

    if kind == "cnn":
        net = SmallCNN(num_classes=num_classes)
        eval_tf = _cnn_eval_transform(img_size)
    elif kind == "resnet":
        net = _build_resnet18_head(num_classes=num_classes)
        eval_tf = _resnet_eval_transform(img_size)
    else:
        raise ValueError(f"Unknown torch model kind: {kind!r}")

    net.load_state_dict(ckpt["model"])
    return TorchClassifier(net, classes=classes, img_size=img_size, eval_transform=eval_tf)
