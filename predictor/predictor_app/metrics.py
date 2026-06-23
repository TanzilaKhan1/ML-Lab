"""Load the analysis metrics (metrics.json) and derive avoidable bias / variance.

The app NEVER computes errors from the dataset (it isn't deployed with one).
It reads ``metrics.json`` — regenerated on the cluster by ``model/export_metrics.py``
after each retrain — and computes the *derived* quantities live:

    avoidable_bias = train_error - human_level_error
    variance       = dev_error   - train_error

So the Model Analysis tab updates automatically whenever a fresh metrics.json
is committed and the app is rebooted. Nothing here is hard-coded per model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .config import MODEL_DIR

METRICS_FILENAME = "metrics.json"


@dataclass
class ModelDiagnosis:
    name: str
    family: str
    accuracy: Optional[float]
    unsafe_recall: Optional[float]
    auc: Optional[float]
    human_error: float
    train_error: Optional[float]
    dev_error: Optional[float]
    dev_method: str
    test_error: Optional[float]
    val_contaminated: bool = False

    @property
    def held_out_error(self) -> Optional[float]:
        """Generalization error used for the variance estimate.

        Standard choice = val (dev). EXCEPTION: when ``val_contaminated`` is set
        — the model was retrained on train+val, so its val error is really
        training error — we fall back to TEST, the only honest held-out set for
        that model. ``held_out_label`` surfaces which one was used.
        """
        if self.val_contaminated and self.test_error is not None:
            return self.test_error
        if self.dev_error is not None:
            return self.dev_error
        return self.test_error

    @property
    def held_out_label(self) -> str:
        if self.val_contaminated and self.test_error is not None:
            return "test (val contaminated — trained on train+val)"
        if self.dev_error is not None:
            return self.dev_method or "val"
        return "test holdout"

    @property
    def avoidable_bias(self) -> Optional[float]:
        if self.train_error is None:
            return None
        return self.train_error - self.human_error

    @property
    def variance(self) -> Optional[float]:
        ho = self.held_out_error
        if self.train_error is None or ho is None:
            return None
        return ho - self.train_error

    @property
    def regime(self) -> str:
        """Plain-language read of where this model's error mainly comes from."""
        ab, var = self.avoidable_bias, self.variance
        if ab is None or var is None:
            ho = self.held_out_error
            if ho is None:
                return "not evaluated"
            gap = ho - self.human_error
            if gap > 0.15:
                return "far below human (likely bias-limited)"
            return "near human (train error needed to confirm)"
        if ab > 0.10 and ab >= var:
            return "bias-limited (underfitting)"
        if var > 0.05 and var > ab:
            return "variance-limited (overfitting / needs data)"
        return "well-balanced"


def _metrics_path() -> Optional[Path]:
    p = MODEL_DIR / METRICS_FILENAME
    return p if p.exists() else None


@lru_cache(maxsize=1)
def load_metrics() -> Optional[dict]:
    """Read metrics.json from the active model dir. Returns None if absent."""
    p = _metrics_path()
    if p is None:
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def model_diagnoses(data: dict) -> list[ModelDiagnosis]:
    """Build per-model diagnoses, injecting the shared human-level error."""
    human_error = float(data.get("human_level", {}).get("error", 0.04))
    out: list[ModelDiagnosis] = []
    for m in data.get("models", []):
        out.append(
            ModelDiagnosis(
                name=m.get("name", "?"),
                family=m.get("family", ""),
                accuracy=m.get("accuracy"),
                unsafe_recall=m.get("unsafe_recall"),
                auc=m.get("auc"),
                human_error=human_error,
                train_error=m.get("train_error"),
                dev_error=m.get("dev_error"),
                dev_method=m.get("dev_method", ""),
                test_error=m.get("test_error"),
                val_contaminated=bool(m.get("val_contaminated", False)),
            )
        )
    return out
