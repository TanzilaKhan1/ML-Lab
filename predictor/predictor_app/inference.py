"""Load trained joblib pipelines and run prediction on a preprocessed image."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .preprocess import ImageInput, preprocess_for_model

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "model"

AVAILABLE_MODELS: dict[str, str] = {
    "Logistic Regression": "logistic_model.joblib",
    "SVM (RBF)": "svm_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
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
        for name, fname in AVAILABLE_MODELS.items()
        if (MODEL_DIR / fname).exists()
    }


@lru_cache(maxsize=8)
def load_model(model_name: str):
    """Load a joblib pipeline by display name. Cached for repeated calls."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {list(AVAILABLE_MODELS)}"
        )
    path = MODEL_DIR / AVAILABLE_MODELS[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}")
    return joblib.load(path)


def predict(image: ImageInput, model_name: str = "Logistic Regression") -> Prediction:
    """Run the full pipeline: preprocess → load model → predict label + probs."""
    features = preprocess_for_model(image)
    model = load_model(model_name)
    pred = int(model.predict(features)[0])

    probs: Optional[dict[str, float]] = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(features)[0]
        probs = {CLASS_NAMES[i]: float(raw[i]) for i in range(len(CLASS_NAMES))}
    elif hasattr(model, "decision_function"):
        scores = np.atleast_1d(model.decision_function(features)[0])
        if scores.size == 1:
            # binary decision_function returns a single score for the positive class (label=1)
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
