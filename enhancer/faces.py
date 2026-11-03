"""Face recovery: find faces (YuNet), align each to the 512px FFHQ layout, restore (GFPGAN), paste back softly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .runtime import Runtime
from .tiling import Cancelled

FACE_SIZE = 512
# FFHQ 5-point layout at 512px (eye on the image left first), as used by GFPGAN's alignment.
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
DETECT_MAX_SIDE = 1600
MIN_EYE_DIST = 10  # px in the image; smaller faces are left alone


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in image pixels
    landmarks: np.ndarray                    # 5x2: eye (image left), eye, nose, mouth left, mouth right
    score: float

    @property
    def eye_dist(self) -> float:
        return float(np.linalg.norm(self.landmarks[1] - self.landmarks[0]))


class FaceDetector:
    def __init__(self, rt: Runtime, score: float = 0.7):
        path = rt.catalog.get("yunet").path("fp32", rt.catalog.root)
        self._net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 5000)

    def detect(self, rgb: np.ndarray) -> list[Face]:
        h, w = rgb.shape[:2]
        s = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) \
            if s < 1 else rgb
        bgr = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        self._net.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = self._net.detect(bgr)
        faces = []
        for row in (found if found is not None else []):
            lm = row[4:14].reshape(5, 2) / s
            # YuNet gives the subject's right eye / mouth corner first, which is on the image left.
            faces.append(Face(box=tuple(float(v) / s for v in row[:4]), landmarks=lm.astype(np.float32),
                              score=float(row[14])))
        return [f for f in faces if f.eye_dist >= MIN_EYE_DIST]


def _soft_mask() -> np.ndarray:
    m = np.zeros((FACE_SIZE, FACE_SIZE), np.float32)
    pad = 26
    m[pad:-pad, pad:-pad] = 1
    return cv2.GaussianBlur(m, (0, 0), 11)


_MASK = _soft_mask()


def restore_faces(rt: Runtime, img: np.ndarray, strength: float = 1.0,
                  progress: Callable[[int, int], None] | None = None,
                  cancel: Callable[[], bool] | None = None,
                  faces: list[Face] | None = None) -> tuple[np.ndarray, int]:
    """Returns (image, number of faces restored). img is HxWx3 float32 and is not modified."""
    if faces is None:
        faces = FaceDetector(rt).detect(img)
    out = img.copy()
    h, w = img.shape[:2]
    for i, face in enumerate(faces):
        if cancel and cancel():
            raise Cancelled()
        m, _ = cv2.estimateAffinePartial2D(face.landmarks, TEMPLATE, method=cv2.LMEDS)
        if m is None:
            continue
        crop = cv2.warpAffine(img, m, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT101)
        restored = rt.run("gfpgan-1.4", crop.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
        restored = np.clip(restored, 0, 1)
        restored = crop + strength * (restored - crop)

        inv = cv2.invertAffineTransform(m)
        corners = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]], np.float32)
        pts = corners @ inv.T
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        local = inv.copy()
        local[:, 2] -= (x0, y0)  # warp straight into the face's region, not the whole image
        size = (x1 - x0, y1 - y0)
        face_back = cv2.warpAffine(restored, local, size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)
        mask = cv2.warpAffine(_MASK, local, size, flags=cv2.INTER_LINEAR, borderValue=0)[:, :, None]
        region = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = region * (1 - mask) + np.clip(face_back, 0, 1) * mask
        if progress:
            progress(i + 1, len(faces))
    return out, len(faces)
