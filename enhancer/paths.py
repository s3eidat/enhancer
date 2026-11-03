"""Where things live, in a source checkout and in a frozen (PyInstaller) build."""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
PKG = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "enhancer" if FROZEN \
    else Path(__file__).resolve().parent
ROOT = PKG.parent


def web_dir() -> Path:
    return PKG / "web"


def data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    d = base / "S3eidat Enhancer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    """ENHANCER_MODELS wins; then ./models next to the code (dev, or shipped with the exe); then the data dir."""
    env = os.environ.get("ENHANCER_MODELS")
    if env:
        return Path(env)
    local = (Path(sys.executable).parent if FROZEN else ROOT) / "models"
    if (local / "manifest.json").exists():
        return local
    d = data_dir() / "models"
    d.mkdir(exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(exist_ok=True)
    return d


def default_output_dir() -> Path:
    return Path.home() / "Pictures" / "Enhanced"
