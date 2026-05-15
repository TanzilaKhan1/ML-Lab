"""LIME explainability for the HOG + sklearn-pipeline classifier.

LIME perturbs an image by masking superpixels, asks the model for a probability
on each perturbed copy, then fits a local linear surrogate. The result is a
per-superpixel weight telling us which regions pushed the prediction toward
each class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from lime import lime_image
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.segmentation import mark_boundaries, slic

from .inference import CLASS_NAMES, UNSAFE_LABEL, load_model
from .preprocess import HOG_PARAMS, HOG_SIZE, ImageInput, standardize_image


@dataclass
class LimeExplanation:
    standardized_rgb: np.ndarray         # (512, 512, 3) uint8 — image LIME was run on
    label: int                            # the label whose evidence we visualized
    label_name: str
    overlay: np.ndarray                   # (H, W, 3) uint8 — image + region outlines
    heatmap: np.ndarray                   # (H, W) float — weight per pixel (signed)
    positive_only: np.ndarray             # (H, W, 3) uint8 — only regions pushing TOWARD label
    top_regions_count: int
    notes: Optional[str] = None


def _batch_hog_features(images_uint8: np.ndarray) -> np.ndarray:
    """Replicate `preprocess_for_model` for a batch of (N, H, W, 3) uint8 arrays."""
    feats = []
    for img in images_uint8:
        pil = Image.fromarray(img).resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        gray = rgb2gray(arr)
        feats.append(hog(gray, **HOG_PARAMS).astype(np.float32))
    return np.stack(feats)


def explain_with_lime(
    image: ImageInput,
    model_name: str,
    *,
    label: int = UNSAFE_LABEL,
    num_samples: int = 600,
    top_regions: int = 5,
    random_state: int = 42,
) -> LimeExplanation:
    """Run LIME on a single image. Returns overlay + heatmap visualisations."""
    std_pil = standardize_image(image)
    std_rgb = np.asarray(std_pil, dtype=np.uint8)

    model = load_model(model_name)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        feats = _batch_hog_features(batch)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(feats)
        # Fallback for decision_function (e.g., SVC without probability=True)
        scores = np.atleast_2d(model.decision_function(feats))
        if scores.shape[1] == 1:
            scores = scores.ravel()
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - p1, p1])
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # Bigger segments → fewer perturbations needed → faster.
    segmentation_fn = lambda img: slic(
        img, n_segments=80, compactness=10, sigma=1, start_label=0
    )

    explainer = lime_image.LimeImageExplainer(random_state=random_state)
    explanation = explainer.explain_instance(
        std_rgb,
        classifier_fn=predict_fn,
        labels=(0, 1),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
        random_seed=random_state,
    )

    # Outlined view: top N regions both for and against the label, semi-transparent.
    img_with_boundaries, mask_signed = explanation.get_image_and_mask(
        label=label,
        positive_only=False,
        negative_only=False,
        num_features=top_regions * 2,
        hide_rest=False,
        min_weight=0.0,
    )
    overlay = mark_boundaries(
        img_with_boundaries.astype(np.uint8), mask_signed, color=(1, 1, 0), mode="thick"
    )
    overlay = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)

    # Positive-only view: regions pushing TOWARD the chosen label.
    pos_img, _ = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        num_features=top_regions,
        hide_rest=True,
    )
    pos_img = pos_img.astype(np.uint8)

    # Continuous heatmap: per-pixel weight from the local linear model.
    heatmap = np.zeros(std_rgb.shape[:2], dtype=np.float32)
    for seg_id, weight in explanation.local_exp[label]:
        heatmap[explanation.segments == seg_id] = float(weight)

    return LimeExplanation(
        standardized_rgb=std_rgb,
        label=label,
        label_name=CLASS_NAMES[label],
        overlay=overlay,
        heatmap=heatmap,
        positive_only=pos_img,
        top_regions_count=top_regions,
    )
