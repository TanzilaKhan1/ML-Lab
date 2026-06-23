"""Centralised configuration: paths, defaults, and UI palette."""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent

# Model-weights lookup. We prefer an in-app `predictor/model/` directory so
# Streamlit Cloud deployments are self-contained (no need to also clone the
# repo-root training tree). If that folder doesn't exist or is empty we fall
# back to the repo-root `model/` directory used during local development.
_LOCAL_MODEL_DIR = PROJECT_ROOT / "model"
_REPO_MODEL_DIR = REPO_ROOT / "model"


def _resolve_model_dir() -> Path:
    if _LOCAL_MODEL_DIR.is_dir() and any(_LOCAL_MODEL_DIR.glob("*.joblib")):
        return _LOCAL_MODEL_DIR
    return _REPO_MODEL_DIR


MODEL_DIR = _resolve_model_dir()
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "images"

SUPPORTED_UPLOAD_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "bmp", "tiff", "heic", "heif")

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("MODEL_DOWNLOAD_TIMEOUT_SECONDS", "900"))
_MODEL_BASE_URL_ENV = "MODEL_BASE_URL"
_MODEL_DOWNLOAD_TOKEN_ENV = "MODEL_DOWNLOAD_TOKEN"
_DEFAULT_MODEL_BASE_URL = (
    "https://github.com/TanzilaKhan1/ML-Lab/releases/download/model-weights-v1"
)
_DEFAULT_MODEL_SHA256_BY_FILE = {
    "resnet50.joblib": "e012f9466f9273f15448d4dffc2eec5a52f5180b69ffecd46660c02c77469d35",
    "convnext_tiny.joblib": "075f6823a489243831773de080997c5b4f4a68eb64659ab7e3f4e1992167d371",
}

# Primary env names are what Streamlit Cloud users should set in app secrets.
# Alternate names keep the loader readable if this app is moved to another host.
_MODEL_URL_ENV_BY_FILE = {
    "resnet50.joblib": ("RESNET50_JOBLIB_URL", "MODEL_RESNET50_URL"),
    "convnext_tiny.joblib": ("CONVNEXT_TINY_JOBLIB_URL", "MODEL_CONVNEXT_TINY_URL"),
}
_MODEL_SHA256_ENV_BY_FILE = {
    "resnet50.joblib": ("RESNET50_JOBLIB_SHA256", "MODEL_RESNET50_SHA256"),
    "convnext_tiny.joblib": ("CONVNEXT_TINY_JOBLIB_SHA256", "MODEL_CONVNEXT_TINY_SHA256"),
}


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def _download_url_for(filename: str) -> str | None:
    explicit = _first_env(_MODEL_URL_ENV_BY_FILE.get(filename, ()))
    if explicit:
        return explicit

    base_url = os.environ.get(_MODEL_BASE_URL_ENV) or _DEFAULT_MODEL_BASE_URL
    if base_url:
        return f"{base_url.rstrip('/')}/{filename}"
    return None


def _expected_sha256_for(filename: str) -> str | None:
    value = _first_env(_MODEL_SHA256_ENV_BY_FILE.get(filename, ()))
    return value.lower() if value else _DEFAULT_MODEL_SHA256_BY_FILE.get(filename)


def is_git_lfs_pointer(path: Path) -> bool:
    """True when Streamlit has cloned a Git LFS pointer instead of the binary."""
    if not path.is_file():
        return False
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(_LFS_POINTER_PREFIX)
    except OSError:
        return False


def is_usable_model_file(path: Path) -> bool:
    return path.is_file() and not is_git_lfs_pointer(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model_file(url: str, destination: Path, expected_sha256: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.download")

    headers = {"User-Agent": "ML-Lab-Streamlit-model-loader"}
    token = os.environ.get(_MODEL_DOWNLOAD_TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with tmp_path.open("wb") as fh:
                shutil.copyfileobj(response, fh)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"download failed from {url}: {exc}") from exc

    if is_git_lfs_pointer(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded another Git LFS pointer from {url}")

    if expected_sha256:
        actual_sha256 = _sha256(tmp_path)
        if actual_sha256 != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {destination.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    tmp_path.replace(destination)


def ensure_model_file(filename: str, *, required: bool = True) -> Path | None:
    """Return a usable model path, downloading configured remote artifacts once.

    Streamlit Community Cloud can clone Git LFS pointer files instead of large
    binaries. This treats those pointers as missing and replaces them from a
    release/bucket URL configured via environment variables or Streamlit secrets.
    """
    path = MODEL_DIR / filename
    if is_usable_model_file(path):
        return path

    url = _download_url_for(filename)
    if not url:
        if required:
            if is_git_lfs_pointer(path):
                url_envs = ", ".join(_MODEL_URL_ENV_BY_FILE.get(filename, ()))
                raise RuntimeError(
                    f"{path} is a Git LFS pointer, not model weights. "
                    f"Set {_MODEL_BASE_URL_ENV} or {url_envs}."
                )
            raise FileNotFoundError(f"Model weights not found at {path}")
        return None

    try:
        _download_model_file(url, path, _expected_sha256_for(filename))
    except Exception:
        if required:
            raise
        return None

    if is_usable_model_file(path):
        return path
    if required:
        raise RuntimeError(f"Downloaded file is not usable model weights: {path}")
    return None


@dataclass(frozen=True)
class LimeDefaults:
    # Lowered 600 -> 200: each sample is a 512x512 image pushed through the
    # model, so the perturbation batch is a large transient memory spike.
    # Keeping the default small avoids OOM on memory-capped hosts (Streamlit
    # Cloud ~1 GB); users can still raise it via the sidebar slider.
    num_samples: int = 200
    num_samples_min: int = 100
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
