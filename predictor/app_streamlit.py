"""Streamlit entry point — orchestrates UI components and prediction calls.

Domain logic lives in `predictor_app/` (preprocess, inference, explain) and
visual building blocks in `predictor_app/ui/`. This module just wires them
together: page setup → header → sidebar → upload → predict → explain.
"""
from __future__ import annotations

import hashlib
from io import BytesIO

import streamlit as st

from predictor_app import (
    UNSAFE_LABEL,
    Prediction,
    explain_with_lime,
    list_models,
    predict,
    standardize_image,
)
from predictor_app.config import SUPPORTED_UPLOAD_EXTENSIONS
from predictor_app.ui import (
    SidebarSettings,
    apply_theme,
    page_config,
    render_explanation,
    render_header,
    render_prediction_card,
    render_sidebar,
    render_upload_panel,
)


# ---------------------------------------------------------------------------
# Cached compute
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_predict(image_hash: str, image_bytes: bytes, model_name: str) -> Prediction:
    return predict(BytesIO(image_bytes), model_name=model_name)


@st.cache_data(show_spinner=False)
def _cached_explain(
    image_hash: str, image_bytes: bytes, model_name: str, label: int, num_samples: int
):
    return explain_with_lime(
        BytesIO(image_bytes),
        model_name=model_name,
        label=label,
        num_samples=num_samples,
    )


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    page_config()
    apply_theme()

    render_header(
        title="🚌 Door-Hanging Safety Predictor",
        subtitle=(
            "Upload a bus or leguna image — the classifier flags whether passengers "
            "are hanging on the door, and LIME shows the regions that drove the call."
        ),
    )

    models = list_models()
    if not models:
        st.error(
            "No model `.joblib` files found under `model/`. "
            "Train a model first (see `model/train_*.py`)."
        )
        return

    settings: SidebarSettings = render_sidebar(models)

    uploaded = st.file_uploader(
        "Drop or browse an image",
        type=list(SUPPORTED_UPLOAD_EXTENSIONS),
        label_visibility="collapsed",
    )

    if uploaded is None:
        return

    raw_bytes = uploaded.read()
    image_hash = _hash_bytes(raw_bytes)

    try:
        standardized = standardize_image(BytesIO(raw_bytes))
    except Exception as e:
        st.error(f"Failed to decode image: {e}")
        return

    render_upload_panel(
        uploaded_bytes=raw_bytes,
        filename=uploaded.name,
        standardized_image=standardized,
    )

    with st.spinner("Running prediction…"):
        result = _cached_predict(image_hash, raw_bytes, settings.model_name)
    render_prediction_card(result)

    if not settings.show_lime:
        return

    try:
        with st.spinner(f"Running LIME with {settings.num_samples} perturbations…"):
            exp = _cached_explain(
                image_hash,
                raw_bytes,
                settings.model_name,
                result.label,
                settings.num_samples,
            )
    except Exception as e:
        st.error(f"LIME failed: {e}")
        return

    render_explanation(exp)


if __name__ == "__main__":
    main()
