"""First-run model download: fetch the ONNX files listed in the manifest, check each SHA-256, keep them in the
app's data folder. Nothing is downloaded without the user pressing the button."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Callable

from .models import Catalog

log = logging.getLogger(__name__)

# Where release builds get the models. Set when the models are published; ENHANCER_MODEL_URL overrides it.
MODEL_URL = "https://github.com/s3eidat/enhancer/releases/download/models-v1/"

ESSENTIAL = ["realesr-general-x4", "realesrgan-x4", "realesrgan-x2", "realesrgan-anime-x4", "nafnet-denoise-32",
             "nafnet-deblur-32", "rife-4.25", "rife-4.25-uhd", "yunet"]
BEST = ["nafnet-denoise-64", "nafnet-deblur-64", "gfpgan-1.4"]


def base_url() -> str:
    url = os.environ.get("ENHANCER_MODEL_URL") or MODEL_URL
    return url if url.endswith("/") else url + "/"


def missing(cat: Catalog, ids: list[str]) -> list[str]:
    out = []
    for mid in ids:
        if mid not in cat.models:
            continue
        f = cat.models[mid].files["fp32"]
        p = cat.root / f["file"]
        if not p.exists() or p.stat().st_size != f["bytes"]:
            out.append(mid)
    return out


def size_of(cat: Catalog, ids: list[str]) -> int:
    return sum(cat.models[m].files["fp32"]["bytes"] for m in ids if m in cat.models)


class Cancelled(Exception):
    pass


def download(cat: Catalog, ids: list[str], progress: Callable[[int, int, str], None] | None = None,
             cancel: Callable[[], bool] | None = None, url: str | None = None) -> None:
    """Download the given models into cat.root. progress(bytes_done, bytes_total, model_id)."""
    todo = missing(cat, ids)
    total, done = size_of(cat, todo), 0
    base = url or base_url()
    cat.root.mkdir(parents=True, exist_ok=True)
    for mid in todo:
        f = cat.models[mid].files["fp32"]
        dst = cat.root / f["file"]
        part = dst.with_suffix(dst.suffix + ".part")
        h = hashlib.sha256()
        log.info("downloading %s", base + f["file"])
        req = urllib.request.Request(base + f["file"], headers={"User-Agent": "S3eidat-Enhancer"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:
                while True:
                    if cancel and cancel():
                        raise Cancelled()
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total, mid)
            if h.hexdigest() != f["sha256"]:
                raise ValueError(f"{f['file']} did not match its checksum; it was not saved. Try again.")
            shutil.move(str(part), str(dst))
        finally:
            part.unlink(missing_ok=True)


def install_manifest(bundled: Path, root: Path) -> None:
    """A frozen build ships manifest.json inside the app; the models live in the data folder next to a copy of it."""
    root.mkdir(parents=True, exist_ok=True)
    dst = root / "manifest.json"
    if bundled.exists() and (not dst.exists() or dst.read_bytes() != bundled.read_bytes()):
        shutil.copy2(bundled, dst)
