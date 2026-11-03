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

from . import APP_NAME, __version__, download, imageio, paths, video
from .autopilot import suggest
from .jobs import JobQueue
from .models import Catalog
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


def supported(p: Path) -> bool:
    return p.suffix.lower() in imageio.INPUT_EXTS or video.is_video(p)


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
        return a.info()

    def start_download(self, best: bool = False) -> dict:
        return self._app.start_download(best)

    def download_state(self) -> dict:
        return dict(self._app.dl)

    def cancel_download(self) -> None:
        self._app.dl["cancel"] = True

    def pick_files(self) -> list[str]:
        if self._app.window is None:
            return []
        import webview
        imgs = ";".join(f"*{e}" for e in sorted(imageio.INPUT_EXTS))
        vids = ";".join(f"*{e}" for e in sorted(video.VIDEO_EXTS))
        res = self._app.window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=(f"Photos and videos ({imgs};{vids})", f"Photos ({imgs})", f"Videos ({vids})",
                        "All files (*.*)"))
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
            if not supported(p) or not p.is_file():
                continue
            kind = "video" if video.is_video(p) else "image"
            try:
                out.append({"path": str(p), "name": p.name, "kind": kind, **self._app.thumb(p)})
            except Exception as e:
                log.warning("cannot read %s: %s", p, e)
                out.append({"path": str(p), "name": p.name, "kind": kind, "error": f"Cannot read this {kind}: {e}"})
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

    def original(self, path: str, at: float | None = None) -> dict:
        """at: for videos, the time in seconds of the frame to show (and to preview and run Autopilot on)."""
        p = Path(path)
        if at is not None and video.is_video(p):
            self._app.frame_time[p] = float(at)
        return self._app.original(p)

    def preview(self, path: str, settings: dict, cx: float, cy: float) -> dict:
        try:
            return self._app.live_preview(Path(path), Settings.from_dict(settings), cx, cy)
        except Exception as e:
            log.exception("preview failed")
            return {"error": str(e)}

    def autopilot(self, path: str) -> dict:
        try:
            tip = suggest(self._app.load_cached(Path(path)).rgb, self._app.rt, is_video=video.is_video(path))
            return {"settings": tip["settings"], "reasons": tip["reasons"]}
        except Exception as e:
            log.exception("autopilot failed")
            return {"error": str(e)}

    def reveal(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            subprocess.Popen(["explorer", "/select,", str(p)])

    def open_file(self, path: str) -> None:
        p = Path(path)
        if p.is_file():
            os.startfile(str(p))  # noqa: S606 (plays a result the app wrote)

    def open_folder(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(str(p))  # noqa: S606 (Windows only, local folder the user chose)


class EnhancerApp:
    def __init__(self, device: str = "auto", port: int = 0):
        self.prefs = Prefs()
        if paths.FROZEN:  # the app ships the manifest; the models themselves live in the data folder
            download.install_manifest(paths.PKG.parent / "models" / "manifest.json", paths.models_dir())
        self.rt = Runtime(device=device)
        log.info("models in %s, ffmpeg %s", self.rt.catalog.root, video.ffmpeg_path())
        self.dl = {"running": False, "done": 0, "total": 0, "model": "", "error": "", "cancel": False}
        self.preview_dir = paths.data_dir() / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        for old in self.preview_dir.glob("*"):
            old.unlink(missing_ok=True)
        self.queue = JobQueue(self.rt, preview_dir=self.preview_dir)
        self.server = LocalServer(port=port)
        self.window = None
        self._cache: tuple[tuple, imageio.Loaded] | None = None
        self._cache_lock = threading.Lock()
        self.frame_time: dict[Path, float] = {}  # video -> time of the frame shown and previewed
        self._vinfo: dict[Path, video.VideoInfo] = {}

    def info(self) -> dict:
        rt = self.rt
        cat = rt.catalog
        need, best = download.missing(cat, download.ESSENTIAL), download.missing(cat, download.BEST)
        return {"version": __version__, "name": APP_NAME, "device": rt.device_name, "gpu": rt.use_gpu,
                "notice": rt.notice, "settings": Settings.from_dict(self.prefs.data.get("settings")).to_dict(),
                "out_dir": self.prefs.data.get("out_dir") or str(paths.default_output_dir()),
                "formats": list(imageio.FORMATS), "credits": cat.credits(),
                "faces_available": cat.available("gfpgan-1.4") and cat.available("yunet"),
                "exts": sorted(imageio.INPUT_EXTS), "video_exts": sorted(video.VIDEO_EXTS),
                "ffmpeg": video.ffmpeg_path() is not None,
                "interp_available": cat.available("rife-4.25") and cat.available("rife-4.25-uhd"),
                "models": {"missing": need, "missing_bytes": download.size_of(cat, need),
                           "best_missing": best, "best_bytes": download.size_of(cat, best),
                           "folder": str(cat.root)}}

    def start_download(self, best: bool = False) -> dict:
        if self.dl["running"]:
            return dict(self.dl)
        ids = download.ESSENTIAL + (download.BEST if best else [])
        self.dl.update(running=True, done=0, total=download.size_of(self.rt.catalog, download.missing(
            self.rt.catalog, ids)), model="", error="", cancel=False)

        def work():
            try:
                download.download(self.rt.catalog, ids,
                                  lambda d, t, m: self.dl.update(done=d, total=t, model=m),
                                  lambda: self.dl["cancel"])
            except download.Cancelled:
                self.dl["error"] = "Download cancelled."
            except Exception as e:
                log.exception("model download failed")
                self.dl["error"] = f"Download failed: {e}"
            finally:
                self.rt.catalog = Catalog(self.rt.catalog.root)
                self.rt.release()
                self.dl["running"] = False

        threading.Thread(target=work, name="models", daemon=True).start()
        return dict(self.dl)

    def video_info(self, p: Path) -> video.VideoInfo:
        if p not in self._vinfo:
            self._vinfo[p] = video.probe(p)
        return self._vinfo[p]

    def load_cached(self, p: Path) -> imageio.Loaded:
        """The last opened image (or video frame) stays in memory so preview clicks and Autopilot are instant."""
        with self._cache_lock:
            is_vid = video.is_video(p)
            if is_vid and p not in self.frame_time:
                self.frame_time[p] = self.video_info(p).duration / 3
            key = (p, p.stat().st_mtime, self.frame_time.get(p) if is_vid else None)
            if self._cache and self._cache[0] == key:
                return self._cache[1]
            if is_vid:
                loaded = imageio.Loaded(rgb=video.read_frame(p, key[2], self.video_info(p)))
            else:
                loaded = imageio.load(p)
            self._cache = (key, loaded)
            return loaded

    def thumb(self, p: Path) -> dict:
        from PIL import Image, ImageOps
        if video.is_video(p):
            info = self.video_info(p)
            frame = self.load_cached(p).rgb
            dst = imageio.preview_jpeg(frame, self.preview_dir / f"thumb_{abs(hash(str(p)))}.jpg", THUMB)
            return {"thumb": self.server.file_url(dst), "size": (info.width, info.height),
                    "video": {"duration": round(info.duration, 2), "fps": round(float(info.fps), 3),
                              "frames": info.frames, "audio": info.audio, "at": self.frame_time[p]}}
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
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"} and loaded.bits == 8 \
                and not video.is_video(p):
            src = p
        else:
            stamp = f"{abs(hash(str(p)))}_{int(self.frame_time.get(p, 0) * 1000)}"
            src = imageio.preview_jpeg(loaded.rgb, self.preview_dir / f"orig_{stamp}.jpg")
        return {"url": self.server.file_url(src), "size": [w, h], "at": self.frame_time.get(p)}

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
                expanded += [str(x) for x in sorted(p.iterdir()) if supported(x)]
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
