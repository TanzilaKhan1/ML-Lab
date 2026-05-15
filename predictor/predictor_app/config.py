"""Centralised configuration: paths, defaults, and UI palette."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
MODEL_DIR = REPO_ROOT / "model"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "images"

SUPPORTED_UPLOAD_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "bmp", "tiff", "heic", "heif")


@dataclass(frozen=True)
class LimeDefaults:
    num_samples: int = 600
    num_samples_min: int = 200
    num_samples_max: int = 2000
    num_samples_step: int = 100
    top_regions: int = 5


@dataclass(frozen=True)
class Palette:
    """Muted, modern palette. Used by `ui/theme.py` and matplotlib helpers."""
    safe: str = "#10b981"        # emerald-500
    safe_soft: str = "#ecfdf5"   # emerald-50
    unsafe: str = "#ef4444"      # red-500
    unsafe_soft: str = "#fef2f2" # red-50
    accent: str = "#6366f1"      # indigo-500
    surface: str = "#ffffff"
    surface_muted: str = "#f8fafc"  # slate-50
    border: str = "#e2e8f0"      # slate-200
    text: str = "#0f172a"        # slate-900
    text_muted: str = "#64748b"  # slate-500


LIME_DEFAULTS = LimeDefaults()
PALETTE = Palette()
