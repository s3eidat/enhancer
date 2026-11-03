"""Batch queue: one worker runs jobs in order on the GPU; the UI watches job state through callbacks."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import imageio, paths
from .ops import resize
from .pipeline import Settings, process
from .runtime import Runtime
from .tiling import Cancelled

log = logging.getLogger(__name__)
BROWSER_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}
AFTER_PREVIEW_MAX = 8192


@dataclass
class Job:
    src: Path
    settings: Settings
    out_dir: Path
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = ""
    out: Path | None = None
    error: str = ""
    in_size: tuple[int, int] | None = None
    out_size: tuple[int, int] | None = None
    seconds: float = 0.0
    faces: int | None = None
    before: Path | None = None  # browser-viewable files for the compare view
    after: Path | None = None
    notice: str = ""
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.src.name, "src": str(self.src), "status": self.status,
                "progress": round(self.progress, 4), "stage": self.stage, "out": str(self.out) if self.out else "",
                "error": self.error, "in_size": self.in_size, "out_size": self.out_size,
                "seconds": round(self.seconds, 1), "faces": self.faces, "settings": self.settings.to_dict(),
                "notice": self.notice}


def upscale_alpha(alpha: np.ndarray | None, w: int, h: int) -> np.ndarray | None:
    if alpha is None or alpha.shape[:2] == (h, w):
        return alpha
    return resize(np.repeat(alpha[:, :, None], 3, 2), w, h)[:, :, 0]


class JobQueue:
    def __init__(self, rt: Runtime, on_change: Callable[[Job], None] | None = None,
                 preview_dir: Path | None = None):
        self.rt = rt
        self.on_change = on_change or (lambda job: None)
        self.preview_dir = preview_dir or (paths.data_dir() / "previews")
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._wake = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="jobs", daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------------
    def add(self, srcs: list[str | Path], settings: Settings, out_dir: str | Path) -> list[Job]:
        added = []
        with self._wake:
            for s in srcs:
                p = Path(s)
                if p.suffix.lower() not in imageio.INPUT_EXTS or not p.is_file():
                    continue
                job = Job(src=p, settings=settings, out_dir=Path(out_dir))
                self.jobs[job.id] = job
                self._order.append(job.id)
                added.append(job)
            self._wake.notify_all()
        for job in added:
            self.on_change(job)
        return added

    def list(self) -> list[Job]:
        return [self.jobs[i] for i in self._order if i in self.jobs]

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def cancel(self, jid: str) -> None:
        job = self.jobs.get(jid)
        if not job:
            return
        job._cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            self.on_change(job)

    def retry(self, jid: str, settings: Settings | None = None) -> None:
        job = self.jobs.get(jid)
        if not job or job.status in ("queued", "running"):
            return
        with self._wake:
            if settings:
                job.settings = settings
            job.status, job.progress, job.stage, job.error = "queued", 0.0, "", ""
            job._cancel = threading.Event()
            self._wake.notify_all()
        self.on_change(job)

    def remove(self, jid: str) -> None:
        job = self.jobs.get(jid)
        if not job:
            return
        if job.status == "running":
            job._cancel.set()
        with self._wake:
            self.jobs.pop(jid, None)
            self._order = [i for i in self._order if i != jid]
        for p in (job.before, job.after):
            if p and p.parent == self.preview_dir:
                p.unlink(missing_ok=True)

    def clear_finished(self) -> list[str]:
        gone = [j.id for j in self.list() if j.status in ("done", "error", "cancelled")]
        for jid in gone:
            self.remove(jid)
        return gone

    def wait_idle(self, timeout: float = 600) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if not any(j.status in ("queued", "running") for j in self.list()):
                return True
            time.sleep(0.05)
        return False

    def shutdown(self) -> None:
        with self._wake:
            self._stop = True
            for j in self.jobs.values():
                j._cancel.set()
            self._wake.notify_all()

    # -- worker ---------------------------------------------------------------
    def _next(self) -> Job | None:
        for jid in self._order:
            job = self.jobs.get(jid)
            if job and job.status == "queued":
                return job
        return None

    def _loop(self) -> None:
        while True:
            with self._wake:
                job = self._next()
                while job is None and not self._stop:
                    self._wake.wait()
                    job = self._next()
                if self._stop:
                    return
                job.status = "running"
            self._run(job)

    def _run(self, job: Job) -> None:
        t0 = time.time()
        last = [0.0]

        def progress(frac: float, stage: str) -> None:
            job.progress, job.stage = frac, stage
            now = time.time()
            if now - last[0] > 0.1 or frac >= 1:
                last[0] = now
                self.on_change(job)

        self.on_change(job)
        try:
            loaded = imageio.load(job.src)
            h, w = loaded.rgb.shape[:2]
            job.in_size = (w, h)
            report: dict = {}
            out = process(loaded.rgb, job.settings, self.rt, progress, job._cancel.is_set, report)
            oh, ow = out.shape[:2]
            alpha = upscale_alpha(loaded.alpha, ow, oh)
            fmt = job.settings.format if job.settings.format in imageio.FORMATS else "png"
            dst = imageio.output_path(job.src, job.out_dir, fmt, job.settings.suffix)
            imageio.save(out, dst, fmt, alpha=alpha, exif=loaded.exif, icc=loaded.icc)
            job.out, job.out_size, job.faces = dst, (ow, oh), report.get("faces")
            job.before = job.src if job.src.suffix.lower() in BROWSER_EXTS and loaded.bits == 8 else \
                imageio.preview_jpeg(loaded.rgb, self.preview_dir / f"{job.id}_before.jpg")
            job.after = imageio.preview_jpeg(out, self.preview_dir / f"{job.id}_after.jpg", AFTER_PREVIEW_MAX)
            job.status, job.progress, job.stage = "done", 1.0, "Done"
        except Cancelled:
            job.status, job.stage = "cancelled", "Cancelled"
        except MemoryError:
            job.status, job.error = "error", "Not enough memory for this size. Try a smaller upscale."
        except Exception as e:
            log.exception("job %s failed", job.src)
            job.status, job.error = "error", str(e) or e.__class__.__name__
        job.seconds = time.time() - t0
        job.notice = self.rt.notice
        self.on_change(job)
