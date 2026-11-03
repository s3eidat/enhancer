"""One function per enhancement step. Each takes and returns an HxWx3 float32 image in [0,1]."""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from .runtime import Runtime, is_oom
from .tiling import run_tiled

log = logging.getLogger(__name__)
Progress = Callable[[int, int], None] | None
Cancel = Callable[[], bool] | None

UPSCALE_MODELS = {  # mode -> {native scale: model id}
    "standard": {4: "realesrgan-x4", 2: "realesrgan-x2"},
    "art": {4: "realesrgan-anime-x4"},
    "fast": {4: "realesr-general-x4"},
}
TILE = {"realesrgan-x4": 256, "realesrgan-x2": 384, "realesrgan-anime-x4": 384, "realesr-general-x4": 512,
        "nafnet-denoise-32": 512, "nafnet-denoise-64": 384, "nafnet-deblur-32": 512, "nafnet-deblur-64": 384}
OVERLAP = {"upscale": 16, "denoise": 32, "sharpen": 32}


def tiled(rt: Runtime, mid: str, img: np.ndarray, progress: Progress = None, cancel: Cancel = None) -> np.ndarray:
    """Run a model tile by tile; on GPU out-of-memory, halve the tile size and try again."""
    info = rt.catalog.get(mid)
    tile = TILE.get(mid, 256)
    while True:
        try:
            out = run_tiled(rt.fn(mid), img, info.scale, tile=tile, overlap=OVERLAP.get(info.kind, 16),
                            multiple=info.multiple, progress=progress, cancel=cancel)
            return np.clip(out, 0, 1, out=out)
        except Exception as e:
            if not is_oom(e) or tile <= 64:
                raise
            tile //= 2
            log.warning("out of GPU memory on %s, retrying with %dpx tiles", mid, tile)
            rt.release()


def blend(base: np.ndarray, result: np.ndarray, strength: float) -> np.ndarray:
    if strength >= 1:
        return result
    return base + np.float32(strength) * (result - base)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    down = w < img.shape[1]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if down else cv2.INTER_CUBIC)


def nafnet(rt: Runtime, task: str, quality: str) -> str:
    """The width-64 model when asked for and downloaded, else the width-32 one (part of the essential download)."""
    best, fast = f"nafnet-{task}-64", f"nafnet-{task}-32"
    return best if quality == "best" and rt.catalog.available(best) else fast


def denoise(rt: Runtime, img: np.ndarray, strength: float, quality: str = "best",
            progress: Progress = None, cancel: Cancel = None) -> np.ndarray:
    return blend(img, tiled(rt, nafnet(rt, "denoise", quality), img, progress, cancel), strength)


def sharpen(rt: Runtime, img: np.ndarray, strength: float, quality: str = "best",
            progress: Progress = None, cancel: Cancel = None) -> np.ndarray:
    return blend(img, tiled(rt, nafnet(rt, "deblur", quality), img, progress, cancel), strength)


def upscale(rt: Runtime, img: np.ndarray, scale: int, mode: str = "standard", detail: float = 1.0,
            progress: Progress = None, cancel: Cancel = None) -> np.ndarray:
    """detail < 1 mixes the AI result with a plain bicubic upscale (less 'AI look')."""
    models = UPSCALE_MODELS[mode]
    native = scale if scale in models else 4
    out = tiled(rt, models[native], img, progress, cancel)
    h, w = img.shape[:2]
    if native != scale:
        out = resize(out, w * scale, h * scale)
    if detail < 1:
        out = blend(resize(img, w * scale, h * scale), out, detail)
    return out
