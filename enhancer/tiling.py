"""Run a model over an image of any size in overlapping tiles, with no visible seams.

Every tile sent to the model has the same shape (DirectML compiles a graph per input shape, so a
changing shape would be slow). Overlaps are cross-faded with per-axis weights that sum to exactly 1,
so the result needs only one output buffer.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

ModelFn = Callable[[np.ndarray], np.ndarray]  # (1,3,h,w) float32 -> (1,3,h*s,w*s) float32


class Cancelled(Exception):
    pass


def _ceil_to(n: int, m: int) -> int:
    return -(-n // m) * m


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Evenly spread tile starts so tiles cover [0, length) and neighbours overlap by at least 2*overlap."""
    if length <= tile:
        return [0]
    step = max(1, tile - 2 * overlap)
    n = math.ceil((length - tile) / step) + 1
    return sorted({round(i * (length - tile) / (n - 1)) for i in range(n)})


def axis_weights(starts: list[int], tile: int, length: int, ramp: int) -> list[np.ndarray]:
    """1D weights per tile (length `tile`) that sum to 1 at every position along the axis."""
    raw = []
    for s in starts:
        w = np.ones(tile, np.float64)
        if ramp > 0:
            r = (np.arange(ramp) + 0.5) / ramp
            if s > 0:
                w[:ramp] = np.minimum(w[:ramp], r)
            if s + tile < length:
                w[-ramp:] = np.minimum(w[-ramp:], r[::-1])
        raw.append(np.maximum(w, 1e-4))
    total = np.zeros(length, np.float64)
    for s, w in zip(starts, raw):
        total[s:s + tile] += w
    return [(w / total[s:s + tile]).astype(np.float32) for s, w in zip(starts, raw)]


def _pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ph, pw = h - img.shape[0], w - img.shape[1]
    if ph <= 0 and pw <= 0:
        return img
    mode = "reflect" if ph < img.shape[0] and pw < img.shape[1] else "edge"
    return np.pad(img, ((0, max(ph, 0)), (0, max(pw, 0)), (0, 0)), mode=mode)


def tile_count(h: int, w: int, tile: int, overlap: int, multiple: int = 1) -> int:
    tile = max(multiple, tile // multiple * multiple)
    th, tw = min(tile, _ceil_to(h, multiple)), min(tile, _ceil_to(w, multiple))
    o = min(overlap, th // 4, tw // 4)
    return len(tile_starts(max(h, th), th, o)) * len(tile_starts(max(w, tw), tw, o))


def run_tiled(fn: ModelFn, img: np.ndarray, scale: int, tile: int = 256, overlap: int = 16, multiple: int = 1,
              progress: Callable[[int, int], None] | None = None,
              cancel: Callable[[], bool] | None = None) -> np.ndarray:
    """img: HxWx3 float32 in [0,1]. Returns (H*scale)x(W*scale)x3 float32."""
    h, w = img.shape[:2]
    tile = max(multiple, tile // multiple * multiple)
    th, tw = min(tile, _ceil_to(h, multiple)), min(tile, _ceil_to(w, multiple))
    padded = _pad_to(img, max(h, th), max(w, tw))
    ph, pw = padded.shape[:2]
    o = min(overlap, th // 4, tw // 4)
    ys, xs = tile_starts(ph, th, o), tile_starts(pw, tw, o)
    wy = axis_weights([y * scale for y in ys], th * scale, ph * scale, o * scale)
    wx = axis_weights([x * scale for x in xs], tw * scale, pw * scale, o * scale)

    out = np.zeros((h * scale, w * scale, 3), np.float32)
    total, done = len(ys) * len(xs), 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if cancel and cancel():
                raise Cancelled()
            patch = np.ascontiguousarray(padded[y:y + th, x:x + tw].transpose(2, 0, 1)[None])
            res = fn(patch)[0].transpose(1, 2, 0)
            oy, ox = y * scale, x * scale
            # crop the part of this tile that lands inside the real (unpadded) output
            hh, ww = min(th * scale, h * scale - oy), min(tw * scale, w * scale - ox)
            if hh > 0 and ww > 0:
                weight = wy[iy][:hh, None, None] * wx[ix][None, :ww, None]
                out[oy:oy + hh, ox:ox + ww] += res[:hh, :ww] * weight
            done += 1
            if progress:
                progress(done, total)
    return out
