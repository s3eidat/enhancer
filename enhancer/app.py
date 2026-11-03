"""Desktop app: a pywebview window over the local UI server, with a JS API into the job queue."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import APP_NAME, __version__, imageio, paths
from .autopilot import suggest
from .jobs import JobQueue
from .ops import resize
from .pipeline import Settings, process
from .runtime import Runtime
from .server import LocalServer

log = logging.getLogger("enhancer")
LIVE_OUT = 768  # live preview output size in px
THUMB = 160


def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.handlers.RotatingFileHandler(paths.logs_dir() / "enhancer.log", maxBytes=1_000_000,
                                              backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    root.addHandler(fh)
    if sys.stderr and not paths.FROZEN:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)
    logging.getLogger("pywebview").setLevel(logging.WARNING)
    sys.excepthook = lambda t, e, tb: log.critical("Unhandled exception", exc_info=(t, e, tb))


class Prefs:
    def __init__(self):
        self.path = paths.data_dir() / "settings.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def save(self, **kw) -> None:
        self.data.update(kw)
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            log.warning("could not save settings")


class Api:
    """Methods callable from JavaScript as window.pywebview.api.<name>(...)."""

    def __init__(self, app: "EnhancerApp"):
        self._app = app

    def ready(self) -> dict:
        a = self._app
        log.info("UI loaded (%s)", a.rt.device_name)
        rt = a.rt
        return {"version": __version__, "name": APP_NAME, "device": rt.device_name, "gpu": rt.use_gpu,
                "notice": rt.notice, "settings": Settings.from_dict(a.prefs.data.get("settings")).to_dict(),
                "out_dir": a.prefs.data.get("out_dir") or str(paths.default_output_dir()),
                "formats": list(imageio.FORMATS), "credits": rt.catalog.credits(),
                "faces_available": rt.catalog.available("gfpgan-1.4") and rt.catalog.available("yunet"),
                "exts": sorted(imageio.INPUT_EXTS)}

    def pick_files(self) -> list[str]:
        if self._app.window is None:
            return []
        import webview
        exts = ";".join(f"*{e}" for e in sorted(imageio.INPUT_EXTS))
        res = self._app.window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                                  file_types=(f"Images ({exts})", "All files (*.*)"))
        return list(res or [])

    def pick_folder(self) -> str:
        if self._app.window is None:
            return ""
        import webview
        res = self._app.window.create_file_dialog(webview.FileDialog.FOLDER)
        return res[0] if res else ""

    def save_prefs(self, settings: dict, out_dir: str) -> None:
        self._app.prefs.save(settings=Settings.from_dict(settings).to_dict(), out_dir=out_dir)

    def add(self, files: list[str], settings: dict, out_dir: str) -> list[dict]:
        s = Settings.from_dict(settings)
        self._app.prefs.save(settings=s.to_dict(), out_dir=out_dir)
        return [j.to_dict() for j in self._app.queue.add(files, s, out_dir)]

    def stage(self, files: list[str]) -> list[dict]:
        """Files dropped or picked but not started yet: just metadata + thumbnail for the list."""
        out = []
        for f in files:
            p = Path(f)
            if p.suffix.lower() not in imageio.INPUT_EXTS or not p.is_file():
                continue
            try:
                out.append({"path": str(p), "name": p.name, **self._app.thumb(p)})
            except Exception as e:
                log.warning("cannot read %s: %s", p, e)
                out.append({"path": str(p), "name": p.name, "error": f"Cannot read this image: {e}"})
        return out

    def jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._app.queue.list()]

    def cancel(self, jid: str) -> None:
        self._app.queue.cancel(jid)

    def retry(self, jid: str, settings: dict | None = None) -> None:
        self._app.queue.retry(jid, Settings.from_dict(settings) if settings else None)

    def remove(self, jid: str) -> None:
        self._app.queue.remove(jid)

    def clear_finished(self) -> list[str]:
        return self._app.queue.clear_finished()

    def compare(self, jid: str) -> dict:
        job = self._app.queue.get(jid)
        if not job or job.status != "done":
            return {}
        srv = self._app.server
        return {"before": srv.file_url(job.before), "after": srv.file_url(job.after),
                "in_size": job.in_size, "out_size": job.out_size}

    def original(self, path: str) -> dict:
        return self._app.original(Path(path))

    def preview(self, path: str, settings: dict, cx: float, cy: float) -> dict:
        try:
            return self._app.live_preview(Path(path), Settings.from_dict(settings), cx, cy)
        except Exception as e:
            log.exception("preview failed")
            return {"error": str(e)}

    def autopilot(self, path: str) -> dict:
        try:
            tip = suggest(self._app.load_cached(Path(path)).rgb, self._app.rt)
            return {"settings": tip["settings"], "reasons": tip["reasons"]}
        except Exception as e:
            log.exception("autopilot failed")
            return {"error": str(e)}

    def reveal(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            subprocess.Popen(["explorer", "/select,", str(p)])

    def open_folder(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(str(p))  # noqa: S606 (Windows only, local folder the user chose)


class EnhancerApp:
    def __init__(self, device: str = "auto", port: int = 0):
        self.prefs = Prefs()
        self.rt = Runtime(device=device)
        self.preview_dir = paths.data_dir() / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        for old in self.preview_dir.glob("*"):
            old.unlink(missing_ok=True)
        self.queue = JobQueue(self.rt, preview_dir=self.preview_dir)
        self.server = LocalServer(port=port)
        self.window = None
        self._cache: tuple[Path, float, imageio.Loaded] | None = None
        self._cache_lock = threading.Lock()

    def load_cached(self, p: Path) -> imageio.Loaded:
        """The last opened image stays in memory so preview clicks and Autopilot are instant."""
        with self._cache_lock:
            mtime = p.stat().st_mtime
            if self._cache and self._cache[0] == p and self._cache[1] == mtime:
                return self._cache[2]
            loaded = imageio.load(p)
            self._cache = (p, mtime, loaded)
            return loaded

    def thumb(self, p: Path) -> dict:
        from PIL import Image, ImageOps
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            size = im.size
            im.thumbnail((THUMB, THUMB))
            dst = self.preview_dir / f"thumb_{abs(hash(str(p)))}.jpg"
            im.convert("RGB").save(dst, "JPEG", quality=85)
        return {"thumb": self.server.file_url(dst), "size": size}

    def original(self, p: Path) -> dict:
        loaded = self.load_cached(p)
        h, w = loaded.rgb.shape[:2]
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"} and loaded.bits == 8:
            src = p
        else:
            src = imageio.preview_jpeg(loaded.rgb, self.preview_dir / f"orig_{abs(hash(str(p)))}.jpg")
        return {"url": self.server.file_url(src), "size": [w, h]}

    def live_preview(self, p: Path, s: Settings, cx: float, cy: float) -> dict:
        loaded = self.load_cached(p)
        img = loaded.rgb
        h, w = img.shape[:2]
        side = max(32, LIVE_OUT // s.upscale)
        cw, ch = min(side, w), min(side, h)
        x0 = int(min(max(0, cx * w - cw / 2), w - cw))
        y0 = int(min(max(0, cy * h - ch / 2), h - ch))
        crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
        t0 = time.time()
        out = process(crop, s, self.rt)
        secs = time.time() - t0
        before = resize(crop, out.shape[1], out.shape[0]) if s.upscale > 1 else crop
        stamp = int(t0 * 1000)
        b = imageio.preview_jpeg(before, self.preview_dir / f"live_{stamp}_b.jpg", quality=95)
        a = imageio.preview_jpeg(out, self.preview_dir / f"live_{stamp}_a.jpg", quality=95)
        return {"before": self.server.file_url(b), "after": self.server.file_url(a),
                "box": [x0 / w, y0 / h, cw / w, ch / h], "size": [out.shape[1], out.shape[0]],
                "seconds": round(secs, 2), "device": self.rt.device_name, "notice": self.rt.notice}

    def on_drop(self, e: dict) -> None:
        files = [f.get("pywebviewFullPath") for f in e.get("dataTransfer", {}).get("files", [])]
        files = [f for f in files if f]
        expanded: list[str] = []
        for f in files:
            p = Path(f)
            if p.is_dir():
                expanded += [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in imageio.INPUT_EXTS]
            else:
                expanded.append(str(p))
        if expanded and self.window:
            self.window.evaluate_js(f"window.onFilesDropped && window.onFilesDropped({json.dumps(expanded)})")

    def run(self, debug: bool = False) -> None:
        import webview
        from webview.dom import DOMEventHandler

        self.window = webview.create_window(APP_NAME, self.server.ui_url(), js_api=Api(self), width=1440,
                                            height=900, min_size=(1080, 700), background_color="#0c0d10")

        def started():
            self.window.events.loaded.wait(15)
            self.window.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True)
            self.window.dom.document.events.drop += DOMEventHandler(self.on_drop, True, True)

        self.window.events.closed += lambda: self.queue.shutdown()
        webview.start(started, debug=debug, private_mode=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="enhancer")
    ap.add_argument("--cpu", action="store_true", help="never use the GPU")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--browser", action="store_true", help="dev: no window, serve the UI + API on localhost")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)
    setup_logging(a.debug)
    log.info("%s %s starting", APP_NAME, __version__)
    app = EnhancerApp(device="cpu" if a.cpu else "auto", port=a.port)
    if a.browser:
        app.server.api = Api(app)
        print(f"UI: {app.server.ui_url()}?browser=1", flush=True)
        threading.Event().wait()
    else:
        app.run(debug=a.debug)


if __name__ == "__main__":
    main()
