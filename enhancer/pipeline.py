"""The enhancement pipeline: denoise -> sharpen -> upscale -> faces, each with its own strength."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable

import numpy as np

from . import ops
from .faces import restore_faces
from .runtime import Runtime

StageProgress = Callable[[float, str], None] | None


@dataclass
class Settings:
    upscale: int = 2          # 1, 2 or 4
    model: str = "standard"   # standard | art | fast
    detail: float = 1.0       # 0..1, how much of the AI upscale to keep
    denoise: float = 0.0      # 0 = off
    sharpen: float = 0.0      # 0 = off
    faces: float = 0.0        # 0 = off
    quality: str = "best"     # best | fast (NAFNet width 64 or 32)
    format: str = "png"
    suffix: str = "_enhanced"

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        d = d or {}
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.upscale = int(s.upscale) if int(s.upscale) in (1, 2, 4) else 2
        s.model = s.model if s.model in ops.UPSCALE_MODELS else "standard"
        s.quality = s.quality if s.quality in ("best", "fast") else "best"
        for k in ("detail", "denoise", "sharpen", "faces"):
            setattr(s, k, float(min(1.0, max(0.0, float(getattr(s, k))))))
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    def steps(self) -> list[tuple[str, float]]:
        """(step, relative cost) in run order; costs only steer the progress bar."""
        out = []
        if self.denoise > 0:
            out.append(("denoise", 1.0))
        if self.sharpen > 0:
            out.append(("sharpen", 1.0))
        if self.upscale > 1:
            out.append(("upscale", {"standard": 6.0, "art": 2.0, "fast": 0.5}[self.model]))
        if self.faces > 0:
            out.append(("faces", 0.5))
        return out


LABELS = {"denoise": "Removing noise", "sharpen": "Sharpening", "upscale": "Upscaling", "faces": "Recovering faces"}


def process(img: np.ndarray, s: Settings, rt: Runtime, progress: StageProgress = None,
            cancel: Callable[[], bool] | None = None, report: dict | None = None) -> np.ndarray:
    steps = s.steps()
    total = sum(c for _, c in steps) or 1.0
    done = 0.0
    report = report if report is not None else {}
    for name, cost in steps:
        def tick(d, t, _base=done, _cost=cost, _name=name):
            if progress:
                progress((_base + _cost * d / max(t, 1)) / total, LABELS[_name])

        tick(0, 1)
        if name == "denoise":
            img = ops.denoise(rt, img, s.denoise, s.quality, tick, cancel)
        elif name == "sharpen":
            img = ops.sharpen(rt, img, s.sharpen, s.quality, tick, cancel)
        elif name == "upscale":
            img = ops.upscale(rt, img, s.upscale, s.model, s.detail, tick, cancel)
        elif name == "faces":
            img, n = restore_faces(rt, img, s.faces, tick, cancel)
            report["faces"] = n
        done += cost
    if progress:
        progress(1.0, "Saving")
    return img
