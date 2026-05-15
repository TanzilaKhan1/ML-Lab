"""Reusable Streamlit components.

Each render_* function takes data + writes UI. The entry point composes them.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..config import LIME_DEFAULTS, PALETTE
from ..explain import LimeExplanation
from ..inference import CLASS_NAMES, UNSAFE_LABEL, Prediction


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
@dataclass
class SidebarSettings:
    model_name: str
    show_lime: bool
    num_samples: int


def render_sidebar(model_choices: dict[str, "object"]) -> SidebarSettings:
    """model_choices: dict[display_name -> Path]. Returns user-selected settings."""
    with st.sidebar:
        st.markdown("### Model")
        model_name = st.selectbox(
            "Classifier",
            list(model_choices.keys()),
            label_visibility="collapsed",
        )
        st.markdown(
            f'<span class="pp-chip">{model_choices[model_name].name}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Explanation")
        show_lime = st.toggle(
            "LIME region heatmap",
            value=True,
            help="Highlight which image regions drove the prediction.",
        )
        num_samples = st.slider(
            "Perturbation samples",
            min_value=LIME_DEFAULTS.num_samples_min,
            max_value=LIME_DEFAULTS.num_samples_max,
            value=LIME_DEFAULTS.num_samples,
            step=LIME_DEFAULTS.num_samples_step,
            help="More samples → smoother map, slower.",
            disabled=not show_lime,
        )

    return SidebarSettings(
        model_name=model_name,
        show_lime=show_lime,
        num_samples=num_samples,
    )


def render_environment_info(*, pytorch_version: str, cuda_available: bool) -> None:
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#64748b;">'
            f'pytorch {pytorch_version} · cuda {"on" if cuda_available else "off"}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="pp-hero">
          <h1>{_html.escape(title)}</h1>
          <p>{_html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Upload + image preview
# ---------------------------------------------------------------------------
def render_upload_panel(
    *, uploaded_bytes: bytes, filename: str, standardized_image
) -> None:
    """Two-column preview of the raw upload and the 512×512 preprocessed crop."""
    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        with st.container(border=True):
            st.markdown("**Uploaded image**")
            st.image(uploaded_bytes, width="stretch")
            st.caption(f"file: `{filename}`")
    with col_out:
        with st.container(border=True):
            st.markdown("**Preprocessed (512×512 center crop)**")
            st.image(standardized_image, width="stretch")
            st.caption("EXIF-oriented · RGB · centre-cropped")


# ---------------------------------------------------------------------------
# Prediction card
# ---------------------------------------------------------------------------
def _prob_bar(name: str, prob: float) -> str:
    pct = max(0.0, min(prob, 1.0)) * 100
    is_unsafe_class = "unsafe" in name.lower() or "positive" in name.lower()
    cls = "unsafe" if is_unsafe_class else "safe"
    return (
        '<div class="pp-prob-row">'
        f'  <div>{_html.escape(name)}</div>'
        f'  <div class="pp-prob-value">{pct:.1f}%</div>'
        f'  <div class="pp-prob-bar"><div class="pp-prob-fill {cls}" '
        f'       style="width:{pct:.2f}%"></div></div>'
        '</div>'
    )


def render_prediction_card(result: Prediction) -> None:
    is_unsafe = result.label == UNSAFE_LABEL
    card_cls = "unsafe" if is_unsafe else "safe"
    title = "UNSAFE — passengers hanging on the door" if is_unsafe else "SAFE — no door-hanging detected"

    st.markdown(
        f"""
        <div class="pp-verdict {card_cls}">
          <div class="pp-verdict-accent"></div>
          <div class="pp-verdict-body">
            <div class="pp-verdict-label">Verdict</div>
            <div class="pp-verdict-title">{_html.escape(title)}</div>
            <p class="pp-verdict-meaning">{_html.escape(result.meaning)}</p>
            <div style="margin-top:10px;">
              <span class="pp-chip">model: {_html.escape(result.model_name)}</span>
              <span class="pp-chip">class: {_html.escape(result.class_name)}</span>
              <span class="pp-chip">label = {result.label}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.probabilities:
        with st.container(border=True):
            st.markdown("**Class probabilities**")
            rows = "".join(
                _prob_bar(name, result.probabilities.get(name, 0.0))
                for name in CLASS_NAMES
            )
            st.markdown(rows, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LIME explanation
# ---------------------------------------------------------------------------
def _heatmap_figure(rgb: np.ndarray, heatmap: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    ax.imshow(rgb)
    vmax = float(np.abs(heatmap).max() or 1.0)
    ax.imshow(heatmap, cmap="RdYlGn_r", alpha=0.55, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10, color=PALETTE.text)
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_explanation(exp: LimeExplanation) -> None:
    st.markdown("### Why this prediction?")
    st.caption(
        f"LIME quantifies how each image region influenced the "
        f"**{exp.label_name}** decision. "
        "Red = pushed toward this class · green = pushed against."
    )

    tab_overlay, tab_heatmap, tab_top, tab_help = st.tabs(
        ["Overlay", "Heatmap", "Top regions", "How to read"]
    )

    with tab_overlay:
        with st.container(border=True):
            st.image(exp.overlay, width="stretch")
            st.caption(
                "Yellow boundaries mark the most influential superpixels. "
                "Coloured mask shows support (positive) and opposition (negative) "
                "for the predicted class."
            )

    with tab_heatmap:
        with st.container(border=True):
            fig = _heatmap_figure(
                exp.standardized_rgb,
                exp.heatmap,
                f"Evidence for: {exp.label_name}",
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("Continuous per-pixel weight from the local linear surrogate.")

    with tab_top:
        with st.container(border=True):
            st.markdown(f"**Top {exp.top_regions_count} regions supporting the prediction**")
            st.image(exp.positive_only, width="stretch")
            st.caption("All other regions are masked — these are the strongest pieces of evidence.")

    with tab_help:
        st.markdown(
            "- **LIME** treats the image as a set of *superpixels* (small connected regions), "
            "then masks subsets of them to see how the model's probability changes.\n"
            "- It fits a tiny linear model in that perturbation space. Each superpixel "
            "gets a positive or negative weight per class.\n"
            "- **Red** = pushed the model **toward** the predicted class. "
            "**Green** = pushed it **against**.\n"
            "- More perturbation samples = smoother map but slower (a few seconds → ~30s)."
        )
