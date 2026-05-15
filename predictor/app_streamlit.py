"""Streamlit UI for the bus/legua door-hanging safety classifier.

Each prediction also shows a LIME region-importance heatmap explaining which
parts of the image pushed the model toward its decision.
"""
from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from predictor_app import (
    CLASS_NAMES,
    UNSAFE_LABEL,
    Prediction,
    explain_with_lime,
    list_models,
    predict,
    standardize_image,
)


st.set_page_config(
    page_title="Door-Hanging Safety Predictor",
    page_icon="🚌",
    layout="wide",
)


def _badge(label: int) -> str:
    color = "#16a34a" if label != UNSAFE_LABEL else "#dc2626"
    text = "SAFE" if label != UNSAFE_LABEL else "UNSAFE"
    return (
        f"<span style='background:{color};color:white;padding:6px 14px;"
        f"border-radius:8px;font-weight:600;font-size:1.1rem;'>"
        f"Prediction: {text}</span>"
    )


def _render_result(result: Prediction) -> None:
    st.markdown(_badge(result.label), unsafe_allow_html=True)
    st.write(f"**Model:** {result.model_name}")
    st.write(f"**Predicted class:** `{result.class_name}` (label = {result.label})")
    st.write(f"**Meaning:** {result.meaning}")

    if result.probabilities:
        st.subheader("Class probabilities")
        for name in CLASS_NAMES:
            p = result.probabilities.get(name, 0.0)
            st.progress(min(max(p, 0.0), 1.0), text=f"{name}: {p*100:.2f}%")


def _heatmap_figure(rgb: np.ndarray, heatmap: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(rgb)
    vmax = float(np.abs(heatmap).max() or 1.0)
    ax.imshow(heatmap, cmap="RdYlGn_r", alpha=0.55, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def _cached_explain(image_bytes: bytes, model_name: str, label: int, num_samples: int):
    return explain_with_lime(
        BytesIO(image_bytes),
        model_name=model_name,
        label=label,
        num_samples=num_samples,
    )


def main() -> None:
    st.title("🚌 Door-Hanging Safety Predictor")
    st.caption(
        "Upload a bus or legua image. The classifier flags whether passengers "
        "are hanging on the door (UNSAFE) or not (SAFE). LIME shows which image "
        "regions drove the decision."
    )

    available = list_models()
    if not available:
        st.error(
            "No model .joblib files found under `model/`. "
            "Train a model first (see `model/train_*.py`)."
        )
        return

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Model", list(available.keys()))
        st.caption(f"Loaded from: `{available[model_name].name}`")
        st.divider()
        st.subheader("LIME")
        show_lime = st.checkbox("Show region-importance heatmap", value=True)
        num_samples = st.slider(
            "Perturbation samples",
            min_value=200,
            max_value=2000,
            value=600,
            step=100,
            help="More samples = smoother map, slower (a few seconds → ~30s).",
        )
        st.divider()
        st.markdown(
            "**Environment**  \n"
            f"PyTorch: `{torch.__version__}`  \n"
            f"CUDA available: `{torch.cuda.is_available()}`"
        )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "heic", "heif"],
    )

    if uploaded is None:
        st.info("Upload an image to run a prediction.")
        return

    raw_bytes = uploaded.read()
    standardized = standardize_image(BytesIO(raw_bytes))

    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        st.subheader("Uploaded image")
        st.image(raw_bytes, width="stretch")
        st.caption(f"Filename: `{uploaded.name}`")

    with col_out:
        st.subheader("Preprocessed (512x512 center crop)")
        st.image(standardized, width="stretch")

    st.divider()
    with st.spinner("Running prediction..."):
        result = predict(BytesIO(raw_bytes), model_name=model_name)
    _render_result(result)

    if not show_lime:
        return

    st.divider()
    st.subheader("Why did the model predict this? — LIME explanation")
    st.caption(
        f"Heatmap shows per-region influence on the **{CLASS_NAMES[result.label]}** "
        "decision. Red regions pushed the model **toward** this class; "
        "green regions pushed it **away**."
    )

    try:
        with st.spinner(f"Running LIME with {num_samples} perturbations…"):
            exp = _cached_explain(raw_bytes, model_name, result.label, num_samples)
    except Exception as e:
        st.error(f"LIME failed: {e}")
        return

    col_hm, col_pos, col_outline = st.columns(3, gap="medium")
    with col_hm:
        st.markdown("**Region-importance heatmap**")
        fig = _heatmap_figure(
            exp.standardized_rgb,
            exp.heatmap,
            f"Evidence for: {exp.label_name}",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_pos:
        st.markdown(f"**Top {exp.top_regions_count} regions supporting the prediction**")
        st.image(exp.positive_only, width="stretch")
        st.caption("Other regions are hidden — these are the strongest pieces of evidence.")

    with col_outline:
        st.markdown("**Outlined regions (positive + negative)**")
        st.image(exp.overlay, width="stretch")
        st.caption(
            "Yellow boundaries mark the most influential superpixels — colored mask shows "
            "support (positive) and opposition (negative) for the predicted class."
        )

    with st.expander("How to read this"):
        st.markdown(
            "- **LIME** treats the image as a set of *superpixels* (small connected regions), "
            "then masks subsets of them to see how the model's probability changes.\n"
            "- It fits a tiny linear model in that perturbation space. Each superpixel "
            "gets a positive or negative weight per class.\n"
            "- Red = pushed the model **toward** the predicted class.  \n"
            "  Green = pushed the model **against** the predicted class.\n"
            "- More perturbation samples = more stable map but slower."
        )


if __name__ == "__main__":
    main()
