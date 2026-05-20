"""LIME explainability for the HOG + sklearn-pipeline classifier.

LIME perturbs an image by masking superpixels, asks the model for a probability
on each perturbed copy, then fits a local linear surrogate. The result is a
per-superpixel weight telling us which regions pushed the prediction toward
the chosen class.

Visual outputs (all RGB uint8, 512×512):
  * ``standardized_rgb`` — the image LIME ran on.
  * ``overlay`` — a tinted overlay where the top-N positive (supporting)
    regions are painted with a red wash and the top-N negative (opposing)
    regions with a green wash, both outlined for clarity. Matches the
    convention used by the matplotlib heatmap (RdYlGn_r).
  * ``heatmap`` — float, signed per-pixel weight from the local linear
    surrogate; rendered as a translucent matplotlib layer in the UI.
  * ``positive_only`` — only the top-N supporting regions retained, rest
    masked black.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lime import lime_image
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.segmentation import mark_boundaries, slic

from .inference import CLASS_NAMES, UNSAFE_LABEL, is_torch_model, load_model
from .preprocess import HOG_PARAMS, HOG_SIZE, ImageInput, standardize_image


@dataclass
class LimeExplanation:
    standardized_rgb: np.ndarray         # (512, 512, 3) uint8
    label: int                            # the class whose evidence we visualised
    label_name: str
    overlay: np.ndarray                   # (H, W, 3) uint8 — tinted overlay
    heatmap: np.ndarray                   # (H, W) float32 — signed per-pixel weight
    positive_only: np.ndarray             # (H, W, 3) uint8 — only supporting regions
    top_regions_count: int
    n_positive_regions: int               # how many top-N regions LIME actually surfaced
    n_negative_regions: int


# ---------------------------------------------------------------------------
# Model wrapper for LIME
# ---------------------------------------------------------------------------
def _batch_hog_features(images_uint8: np.ndarray) -> np.ndarray:
    """Replicate `preprocess_for_model` for a batch of (N, H, W, 3) uint8 arrays."""
    feats = []
    for img in images_uint8:
        pil = Image.fromarray(img).resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        gray = rgb2gray(arr)
        feats.append(hog(gray, **HOG_PARAMS).astype(np.float32))
    return np.stack(feats)


def _make_predict_fn(model):
    # Torch wrappers consume the raw (N, H, W, 3) uint8 batch directly — they
    # do their own resize/normalize, no HOG features involved.
    if is_torch_model(model):
        def predict_fn(batch: np.ndarray) -> np.ndarray:
            return model.predict_proba(batch)
        return predict_fn

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        feats = _batch_hog_features(batch)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(feats)
        # No predict_proba — fall back to decision_function. For binary
        # classifiers sklearn returns a 1D array of shape (N,) holding the
        # margin for the positive class; for multiclass it returns (N, C).
        scores = model.decision_function(feats)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - p1, p1])
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    return predict_fn


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def _build_overlay(
    base_rgb_uint8: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    pos_color=(0.86, 0.18, 0.18),   # red wash — supports the predicted class
    neg_color=(0.14, 0.62, 0.30),   # green wash — opposes the predicted class
    tint_alpha: float = 0.40,
    boundary_color=(1.0, 1.0, 1.0),  # white outline for crispness
) -> np.ndarray:
    """Render an overlay where positive regions are red-tinted and negative regions
    green-tinted, with thin white boundaries traced around both. Returns uint8."""
    base = base_rgb_uint8.astype(np.float32) / 255.0
    pos_rgb = np.array(pos_color, dtype=np.float32)
    neg_rgb = np.array(neg_color, dtype=np.float32)

    out = base.copy()
    if pos_mask is not None and np.any(pos_mask > 0):
        pixels = pos_mask > 0
        out[pixels] = (1 - tint_alpha) * base[pixels] + tint_alpha * pos_rgb
    if neg_mask is not None and np.any(neg_mask > 0):
        pixels = neg_mask > 0
        out[pixels] = (1 - tint_alpha) * base[pixels] + tint_alpha * neg_rgb

    # Draw boundaries on each non-empty mask. mark_boundaries requires float input
    # in [0, 1] for the boundary color to render correctly.
    if pos_mask is not None and np.any(pos_mask > 0):
        out = mark_boundaries(out, pos_mask.astype(np.int32), color=boundary_color,
                              mode="thick", background_label=0)
    if neg_mask is not None and np.any(neg_mask > 0):
        out = mark_boundaries(out, neg_mask.astype(np.int32), color=boundary_color,
                              mode="thick", background_label=0)

    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def explain_with_lime(
    image: ImageInput,
    model_name: str,
    *,
    label: int = UNSAFE_LABEL,
    num_samples: int = 600,
    top_regions: int = 5,
    n_segments: int = 80,
    batch_size: int = 64,
    random_state: int = 42,
) -> LimeExplanation:
    """Run LIME on a single image and return overlay + heatmap visualisations."""
    if num_samples < 50:
        raise ValueError(f"num_samples must be >= 50 to fit a stable surrogate, got {num_samples}")
    if top_regions < 1:
        raise ValueError(f"top_regions must be >= 1, got {top_regions}")

    std_pil = standardize_image(image)
    std_rgb = np.asarray(std_pil, dtype=np.uint8)

    model = load_model(model_name)
    predict_fn = _make_predict_fn(model)

    # Bigger segments → fewer perturbations needed → faster.
    def segmentation_fn(img):
        return slic(
            img, n_segments=n_segments, compactness=10, sigma=1, start_label=0,
        )

    explainer = lime_image.LimeImageExplainer(random_state=random_state)
    explanation = explainer.explain_instance(
        std_rgb,
        classifier_fn=predict_fn,
        # Only fit a surrogate for the label we'll visualise. LIME defaults to
        # top_labels=5 which would also engage labels we don't need; explicitly
        # disabling it forces use of `labels=(label,)`.
        labels=(label,),
        top_labels=None,
        # hide_color=None lets LIME replace each masked superpixel with the
        # segment's mean colour. Critical for HOG models: a black fill (0)
        # creates artificial high-gradient edges at every segment boundary,
        # which HOG over-responds to and skews the local linear surrogate.
        hide_color=None,
        num_samples=num_samples,
        # Larger than LIME's default of 10 → fewer Python-side predict calls.
        batch_size=batch_size,
        segmentation_fn=segmentation_fn,
        random_seed=random_state,
    )

    if label not in explanation.local_exp:
        raise ValueError(
            f"LIME did not produce an explanation for label={label}. "
            f"Available: {list(explanation.local_exp)}"
        )

    # Separate top-N positive and top-N negative masks (each returns 0/1 mask).
    _, pos_mask = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        negative_only=False,
        num_features=top_regions,
        hide_rest=False,
        min_weight=0.0,
    )
    _, neg_mask = explanation.get_image_and_mask(
        label=label,
        positive_only=False,
        negative_only=True,
        num_features=top_regions,
        hide_rest=False,
        min_weight=0.0,
    )

    # Tinted overlay (the main "where did the model look" visualisation).
    overlay = _build_overlay(std_rgb, pos_mask, neg_mask)

    # Positive-only view: only the top-N supporting regions are shown; rest is black.
    pos_only_img, _ = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        num_features=top_regions,
        hide_rest=True,
        min_weight=0.0,
    )
    pos_only_img = pos_only_img.astype(np.uint8)

    # Continuous per-pixel heatmap from the local linear surrogate weights.
    # Build a segment_id → weight lookup, then index once — O(pixels), vs.
    # the previous O(segments × pixels) boolean-mask loop.
    segments = explanation.segments
    seg_to_weight = np.zeros(int(segments.max()) + 1, dtype=np.float32)
    for seg_id, weight in explanation.local_exp[label]:
        seg_to_weight[int(seg_id)] = float(weight)
    heatmap = seg_to_weight[segments]

    # Count how many top-N regions LIME actually surfaced. This must match
    # `get_image_and_mask`'s filter-then-take-first-N semantics: it filters to
    # positives (or negatives), then takes the first `num_features` from the
    # already-by-|weight|-sorted list. So the answer is simply
    # min(top_regions, count_of_positives_in_local_exp).
    weights = np.asarray([w for _, w in explanation.local_exp[label]], dtype=np.float32)
    n_pos = int(min(top_regions, np.sum(weights > 0)))
    n_neg = int(min(top_regions, np.sum(weights < 0)))

    return LimeExplanation(
        standardized_rgb=std_rgb,
        label=label,
        label_name=CLASS_NAMES[label],
        overlay=overlay,
        heatmap=heatmap,
        positive_only=pos_only_img,
        top_regions_count=top_regions,
        n_positive_regions=n_pos,
        n_negative_regions=n_neg,
    )
