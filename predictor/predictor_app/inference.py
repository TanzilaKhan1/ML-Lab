"""Load trained models (sklearn pipelines or torch state_dicts) and predict."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from PIL import Image

from .config import MODEL_DIR
from .preprocess import ImageInput, preprocess_for_model, standardize_image

# Label semantics (matches scan_dataset's alphabetical ordering and the saved
# GaussianNB class_prior_ = [0.6677, 0.3322] — majority class is label 0):
#   0 = negative (safe)    — passengers are NOT hanging on the door
#   1 = positive (UNSAFE)  — passengers ARE hanging on the door
CLASS_NAMES = ["negative (safe)", "positive (UNSAFE)"]
LABEL_MEANING = {
    0: "Safe — no passengers hanging on the door",
    1: "UNSAFE — passengers are hanging on the door",
}
UNSAFE_LABEL = 1

# (display_name, joblib filename, kind). ``kind`` selects the load path:
#   "sklearn" → loaded directly; predict() feeds it HOG features.
#   "cnn" / "resnet" → torch state_dict; loaded via torch_models wrapper,
#     which preprocesses raw PIL images itself (no HOG).
AVAILABLE_MODELS: dict[str, tuple[str, str]] = {
    "Logistic Regression": ("logistic_model.joblib", "sklearn"),
    "SVM (RBF)": ("svm_model.joblib", "sklearn"),
    "Naive Bayes": ("naive_bayes_model.joblib", "sklearn"),
    "CNN": ("cnn_model.joblib", "cnn"),
    "ResNet": ("resnet_model.joblib", "resnet"),
}


@dataclass
class Prediction:
    model_name: str
    label: int
    class_name: str
    meaning: str
    probabilities: Optional[dict[str, float]]


def list_models() -> dict[str, Path]:
    """Return {display_name: absolute_path} for every model file that exists on disk."""
    return {
        name: (MODEL_DIR / fname)
        for name, (fname, _kind) in AVAILABLE_MODELS.items()
        if (MODEL_DIR / fname).exists()
    }


@lru_cache(maxsize=8)
def load_model(model_name: str):
    """Load a model by display name. Cached for repeated calls.

    sklearn pipelines are returned directly. Torch checkpoints are hydrated
    into a ``TorchClassifier`` wrapper that exposes the same predict-like API.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {list(AVAILABLE_MODELS)}"
        )
    fname, kind = AVAILABLE_MODELS[model_name]
    path = MODEL_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}")

    if kind == "sklearn":
        return joblib.load(path)
    if kind in ("cnn", "resnet"):
        # Lazy import so the app still loads sklearn-only when torch isn't installed.
        from .torch_models import load_torch_checkpoint
        return load_torch_checkpoint(path, kind=kind)
    raise ValueError(f"Unknown model kind for {model_name!r}: {kind!r}")


def is_torch_model(model) -> bool:
    """True if ``model`` came from the torch-checkpoint path."""
    # Avoid importing torch_models eagerly at module load.
    from .torch_models import TorchClassifier
    return isinstance(model, TorchClassifier)


def _image_to_pil(image: ImageInput) -> Image.Image:
    """Standardize an image input to a 512x512 RGB PIL for torch models.

    Mirrors the front-end's standardize_image step so torch models see the
    same framed input the UI shows. Their own resize is applied on top.
    """
    return standardize_image(image)


def predict(image: ImageInput, model_name: str = "Logistic Regression") -> Prediction:
    """Run the full pipeline: preprocess → load model → predict label + probs."""
    model = load_model(model_name)

    if is_torch_model(model):
        pil = _image_to_pil(image)
        probs_arr = model.predict_proba(pil)[0]
        pred = int(probs_arr.argmax())
        probs = {CLASS_NAMES[i]: float(probs_arr[i]) for i in range(len(CLASS_NAMES))}
    else:
        features = preprocess_for_model(image)
        pred = int(model.predict(features)[0])

        probs: Optional[dict[str, float]] = None
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(features)[0]
            probs = {CLASS_NAMES[i]: float(raw[i]) for i in range(len(CLASS_NAMES))}
        elif hasattr(model, "decision_function"):
            scores = np.atleast_1d(model.decision_function(features)[0])
            if scores.size == 1:
                s = float(scores[0])
                p1 = 1.0 / (1.0 + np.exp(-s))
                probs = {CLASS_NAMES[0]: float(1 - p1), CLASS_NAMES[1]: float(p1)}

    return Prediction(
        model_name=model_name,
        label=pred,
        class_name=CLASS_NAMES[pred],
        meaning=LABEL_MEANING[pred],
        probabilities=probs,
    )
