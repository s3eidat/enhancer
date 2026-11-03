"""Load and save images without losing what matters: EXIF, colour profile, alpha and 16-bit depth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

INPUT_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}
FORMATS = {  # id -> (extension, bit depth, keeps alpha)
    "png": (".png", 8, True),
    "jpg": (".jpg", 8, False),
    "webp": (".webp", 8, True),
    "tiff": (".tif", 8, True),
    "png16": (".png", 16, True),
    "tiff16": (".tif", 16, True),
}
ORIENTATION = 0x0112


@dataclass
class Loaded:
    rgb: np.ndarray                 # HxWx3 float32 in [0,1]
    alpha: np.ndarray | None = None  # HxW float32 in [0,1]
    exif: bytes | None = None
    icc: bytes | None = None
    bits: int = 8
    meta: dict = field(default_factory=dict)


def _read_bytes(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths, unlike cv2.imread


def _load_16bit(path: Path) -> Loaded | None:
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        return None
    arr = cv2.imdecode(_read_bytes(path), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.dtype != np.uint16:
        return None
    arr = arr.astype(np.float32) / 65535.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    alpha = None
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3].copy()
        arr = arr[:, :, :3]
    rgb = np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr[:, :, ::-1]
    return Loaded(rgb=np.ascontiguousarray(rgb), alpha=alpha, bits=16)


def load(path: str | Path) -> Loaded:
    path = Path(path)
    deep = _load_16bit(path)
    if deep is not None:
        return deep
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        if im.mode == "CMYK":
            icc = None  # the profile describes CMYK; after conversion it would be wrong
        im = ImageOps.exif_transpose(im)  # rotate once, then mark the EXIF as upright
        exif = im.getexif()
        exif_bytes = None
        if len(exif):
            exif[ORIENTATION] = 1
            exif_bytes = exif.tobytes()
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3].copy() if has_alpha else None
    if alpha is not None and alpha.min() >= 1.0:
        alpha = None
    return Loaded(rgb=np.ascontiguousarray(arr[:, :, :3]), alpha=alpha, exif=exif_bytes, icc=icc)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def output_path(src: Path, out_dir: Path, fmt: str, suffix: str = "_enhanced", ext: str | None = None) -> Path:
    ext = ext or FORMATS[fmt][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{src.stem}{suffix}{ext}"
    n = 2
    while p.exists():
        p = out_dir / f"{src.stem}{suffix}-{n}{ext}"
        n += 1
    return p


def save(rgb: np.ndarray, path: str | Path, fmt: str = "png", alpha: np.ndarray | None = None,
         exif: bytes | None = None, icc: bytes | None = None, quality: int = 95) -> Path:
    path = Path(path)
    ext, bits, keeps_alpha = FORMATS[fmt]
    if not keeps_alpha:
        alpha = None
    if bits == 16:
        arr = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16)[:, :, ::-1]
        if alpha is not None:
            a = (np.clip(alpha, 0, 1) * 65535 + 0.5).astype(np.uint16)
            arr = np.dstack([arr, a])
        ok, buf = cv2.imencode(ext, arr)
        if not ok:
            raise OSError(f"Could not encode {path.name}")
        buf.tofile(str(path))
        return path
    data = to_uint8(rgb)
    if alpha is not None:
        data = np.dstack([data, to_uint8(alpha)])
    im = Image.fromarray(data, "RGBA" if alpha is not None else "RGB")
    kw: dict = {}
    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    if fmt == "jpg":
        kw.update(quality=quality, subsampling=0, optimize=True)
        im.save(path, "JPEG", **kw)
    elif fmt == "webp":
        kw.update(quality=quality, method=4)
        im.save(path, "WEBP", **kw)
    elif fmt == "tiff":
        im.save(path, "TIFF", compression="tiff_lzw", **kw)
    else:
        im.save(path, "PNG", compress_level=4, **kw)
    return path


def preview_jpeg(rgb: np.ndarray, path: Path, max_side: int = 0, quality: int = 92) -> Path:
    """A browser-friendly copy for the compare view. max_side=0 keeps full size."""
    img = to_uint8(rgb)
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    Image.fromarray(img).save(path, "JPEG", quality=quality)
    return path
