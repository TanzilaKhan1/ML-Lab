"""Page config + theme injection for the Streamlit app.

Modest, modern look: Inter typography, slate neutrals, subtle borders,
soft elevation on cards, hidden Streamlit chrome. Streamlit primary colour
is set via `.streamlit/config.toml` so sliders/toggles aren't loud red.
"""
from __future__ import annotations

import streamlit as st

from ..config import PALETTE

# We intentionally do NOT set `font-family` via injected CSS. Any CSS rule
# that matches `span` or `[class*="st-"]` also matches Streamlit's
# Material-icon ligature spans (e.g. `keyboard_double_arrow_left` for the
# sidebar collapse, `file_upload` for the uploader). Overriding the font on
# those spans makes the ligature render as its raw text. The body font is
# set via `.streamlit/config.toml` (`font = "sans serif"`) which keeps the
# icon-font cascade intact.
_CSS = f"""
<style>
  h1, h2, h3, h4 {{
    letter-spacing: -0.01em;
    color: {PALETTE.text};
  }}

  /* ----- inline code in regular markdown: muted, not loud ----- */
  .stApp code {{
    background: {PALETTE.surface_muted};
    color: {PALETTE.text} !important;
    padding: 1px 6px;
    border-radius: 6px;
    font-size: 0.85em;
    border: 1px solid {PALETTE.border};
  }}

  /* ----- hide Streamlit chrome -----
     We intentionally do NOT hide [data-testid="stToolbar"] here: that
     container holds the Deploy button (kept visible for local development)
     and, in some Streamlit builds, the sidebar collapse/expand toggle that
     lets users reopen the sidebar after collapsing it. */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stDecoration"] {{ display: none; }}

  /* Make absolutely sure the chevron used to reopen a collapsed sidebar is
     visible regardless of any container-level rules above. */
  [data-testid="stSidebarCollapsedControl"] {{
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
  }}

  /* ----- page width ----- */
  .block-container {{
    padding-top: 2.25rem;
    padding-bottom: 4rem;
    max-width: 1240px;
  }}

  /* ----- bordered containers act as cards ----- */
  [data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: {PALETTE.border} !important;
    border-radius: 14px !important;
    background: {PALETTE.surface};
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  }}

  /* ----- header block ----- */
  .pp-hero {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 4px 0 18px 0;
    border-bottom: 1px solid {PALETTE.border};
    margin-bottom: 20px;
  }}
  .pp-hero h1 {{
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0;
  }}
  .pp-hero p {{
    color: {PALETTE.text_muted};
    margin: 0;
    font-size: 0.95rem;
  }}

  /* ----- verdict card with accent bar ----- */
  .pp-verdict {{
    display: flex;
    align-items: stretch;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {PALETTE.border};
    background: {PALETTE.surface};
    margin: 6px 0 4px 0;
  }}
  .pp-verdict-accent {{ width: 6px; }}
  .pp-verdict-body {{ padding: 18px 22px; flex: 1; }}
  .pp-verdict-label {{
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
    font-weight: 600;
    color: {PALETTE.text_muted};
    margin-bottom: 6px;
  }}
  .pp-verdict-title {{
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: {PALETTE.text};
  }}
  .pp-verdict-meaning {{
    color: {PALETTE.text_muted};
    font-size: 0.95rem;
    margin: 0;
  }}
  .pp-verdict.safe {{ background: {PALETTE.safe_soft}; }}
  .pp-verdict.safe .pp-verdict-accent {{ background: {PALETTE.safe}; }}
  .pp-verdict.unsafe {{ background: {PALETTE.unsafe_soft}; }}
  .pp-verdict.unsafe .pp-verdict-accent {{ background: {PALETTE.unsafe}; }}

  /* ----- chips inside the verdict card ----- */
  .pp-chip {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: {PALETTE.surface};
    border: 1px solid {PALETTE.border};
    font-size: 0.78rem;
    color: {PALETTE.text_muted};
    margin-right: 6px;
  }}

  /* ----- probability bars ----- */
  .pp-prob-row {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 6px 12px;
    margin: 10px 0 6px 0;
    font-size: 0.88rem;
    color: {PALETTE.text};
  }}
  .pp-prob-value {{ font-variant-numeric: tabular-nums; color: {PALETTE.text_muted}; }}
  .pp-prob-bar {{
    grid-column: 1 / 3;
    height: 6px;
    background: {PALETTE.surface_muted};
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid {PALETTE.border};
  }}
  .pp-prob-fill {{ height: 100%; border-radius: 999px; background: {PALETTE.accent}; }}
  .pp-prob-fill.safe   {{ background: {PALETTE.safe}; }}
  .pp-prob-fill.unsafe {{ background: {PALETTE.unsafe}; }}

  /* ----- file uploader: keep defaults, just spacing ----- */
  [data-testid="stFileUploaderDropzone"] {{
    min-height: 96px;
    padding: 10px 16px;
  }}

  /* ----- tabs polish ----- */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0;
    padding: 8px 14px;
  }}

  /* ----- caption color ----- */
  [data-testid="stCaptionContainer"] {{ color: {PALETTE.text_muted}; }}

  /* ----- soften sidebar divider ----- */
  [data-testid="stSidebar"] hr {{
    border-color: {PALETTE.border};
    margin: 1rem 0;
  }}
</style>
"""


def page_config() -> None:
    st.set_page_config(
        page_title="Door-Hanging Safety Predictor",
        page_icon="🚌",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
