"""Autopilot: look at a photo and suggest settings, with a plain reason for each."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .faces import FaceDetector
from .runtime import Runtime

_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_sigma(rgb: np.ndarray) -> float:
    """Per-channel noise level on a 0-255 scale (Immerkaer 1996), measured on flat areas only so texture
    is not mistaken for noise."""
    img = np.clip(rgb, 0, 1).astype(np.float32) * 255
    h, w = img.shape[:2]
    if max(h, w) > 2048:  # a centre crop keeps it fast without resampling away the noise
        cy, cx, r = h // 2, w // 2, 1024
        img = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resp = np.abs(cv2.filter2D(img, -1, _KERNEL))[1:-1, 1:-1].mean(axis=2)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2)  # find flat areas on a blurred copy so the noise itself does not bias the pick
    edges = cv2.Sobel(smooth, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(smooth, cv2.CV_32F, 0, 1) ** 2
    edges = edges[1:-1, 1:-1]
    flat = edges <= np.percentile(edges, 50)
    if flat.sum() < 100:
        flat = np.ones_like(flat)
    return float(math.sqrt(math.pi / 2) / 6 * resp[flat].mean())


def blur_score(rgb: np.ndarray) -> float:
    """Variance of the Laplacian at ~1 megapixel; low means soft or blurry."""
    gray = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY) * 255
    h, w = gray.shape
    s = min(1.0, 1200 / max(h, w))
    if s < 1:
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def suggest(rgb: np.ndarray, rt: Runtime | None = None, is_video: bool = False) -> dict:
    """For a video, rgb is one frame. Videos get the fast upscaler and smaller sizes: every frame costs time."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    s: dict = {}
    reasons: list[str] = []

    if is_video:
        if long_side < 720:
            s["upscale"], why = 4, "low resolution"
        elif long_side < 1600:
            s["upscale"], why = 2, "HD or less"
        else:
            s["upscale"], why = 1, "already Full HD or more"
        s["model"] = "fast"
        reasons.append(f"Upscale {s['upscale']}x: {why} ({w}x{h}), Fast model for video")
    elif long_side < 1200:
        s["upscale"], why = 4, f"small image ({w}x{h})"
    elif long_side < 3000:
        s["upscale"], why = 2, f"medium image ({w}x{h})"
    else:
        s["upscale"], why = 1, f"already large ({w}x{h})"
    if not is_video:
        reasons.append(f"Upscale {s['upscale']}x: {why}")

    sigma = noise_sigma(rgb)
    s["denoise"] = 0.0 if sigma < 1.5 else 0.5 if sigma < 4 else 0.75 if sigma < 8 else 1.0
    reasons.append(f"Denoise {int(s['denoise'] * 100)}%: noise level {sigma:.1f}")

    blur = blur_score(rgb)
    s["sharpen"] = 0.5 if blur < 60 else 0.0
    reasons.append(f"Sharpen {int(s['sharpen'] * 100)}%: " + ("looks soft" if s["sharpen"] else "looks sharp")
                   + f" ({blur:.0f})")

    s["faces"] = 0.0
    if rt is not None and rt.catalog.available("yunet") and rt.catalog.available("gfpgan-1.4"):
        faces = FaceDetector(rt).detect(rgb)
        if faces:
            biggest = max(f.eye_dist for f in faces)
            # GFPGAN works at ~126px between the eyes. Faces smaller than that in the source gain real
            # detail; big, clean faces only lose some of theirs.
            s["faces"] = 0.8 if biggest < 130 or sigma >= 6 else 0.5 if biggest < 200 else 0.0
            if is_video:  # restored faces can shimmer from frame to frame, so stay gentle
                s["faces"] = min(s["faces"], 0.5)
            reasons.append(f"Faces {int(s['faces'] * 100)}%: {len(faces)} found, "
                           + ("small or noisy" if s["faces"] >= 0.8 else "medium" if s["faces"] else "already detailed"))
        else:
            reasons.append("Faces off: none found")
    return {"settings": s, "reasons": reasons, "noise": sigma, "blur": blur}
