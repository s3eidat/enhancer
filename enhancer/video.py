"""Video: decode frames with ffmpeg, enhance each one, optionally add in-between frames with RIFE, encode, keep audio.

Three threads keep the GPU busy: one reads raw frames from the decoder, the caller's thread enhances and
interpolates, one writes raw frames into the encoder.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import paths
from .pipeline import LABELS, Settings, process
from .runtime import Runtime, is_oom
from .tiling import Cancelled

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".mts",
              ".m2ts", ".3gp"}
CONTAINER = {"h264": ".mp4", "h265": ".mp4", "prores": ".mov"}
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"}
MOV_AUDIO = {"aac", "mp3", "alac", "pcm_s16le", "pcm_s24le"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SCENE_CUT_SSIM = 0.2  # same threshold as Practical-RIFE: below it, frames are a cut, not motion


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


# -- ffmpeg -------------------------------------------------------------------
def ffmpeg_path(tool: str = "ffmpeg") -> str | None:
    """ENHANCER_FFMPEG (folder or exe), then PATH, then C:/ffmpeg/*/bin, then the app's data folder."""
    exe = tool + (".exe" if os.name == "nt" else "")
    env = os.environ.get("ENHANCER_FFMPEG")
    cands = []
    if env:
        p = Path(env)
        cands.append(p.parent / exe if p.is_file() else p / exe)
    found = shutil.which(tool)
    if found:
        cands.append(Path(found))
    cands += [Path(p) for p in sorted(glob.glob(f"C:/ffmpeg/*/bin/{exe}"), reverse=True)]
    cands += [Path("C:/ffmpeg/bin") / exe, paths.data_dir() / "ffmpeg" / exe]
    for c in cands:
        if c.is_file():
            return str(c)
    return None


class NoFFmpeg(RuntimeError):
    pass


def _need(tool: str) -> str:
    p = ffmpeg_path(tool)
    if not p:
        raise NoFFmpeg("ffmpeg was not found. Install it (or set ENHANCER_FFMPEG) to enhance videos.")
    return p


_NVENC: dict[str, bool] = {}


def has_nvenc(codec: str) -> bool:
    """True when this PC can encode with the NVIDIA hardware encoder (tested once with a tiny encode)."""
    if codec not in _NVENC:
        try:
            r = subprocess.run([_need("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256",
                                "-frames:v", "1", "-c:v", codec, "-f", "null", "-"],
                               capture_output=True, timeout=20, creationflags=NO_WINDOW)
            _NVENC[codec] = r.returncode == 0
        except Exception:
            _NVENC[codec] = False
    return _NVENC[codec]


@dataclass
class VideoInfo:
    width: int          # as displayed (rotation applied, like ffmpeg's decoder output)
    height: int
    fps: Fraction
    frames: int         # best estimate, for progress
    duration: float
    pix_fmt: str
    color_space: str
    audio: str | None   # codec of the first audio stream
    codec: str

    @property
    def high_bit(self) -> bool:
        return any(t in self.pix_fmt for t in ("10", "12", "16"))


def _fraction(s: str | None) -> Fraction | None:
    try:
        f = Fraction(s)
        return f if 0 < f < 1000 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe(path: str | Path) -> VideoInfo:
    r = subprocess.run([_need("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        str(path)], capture_output=True, timeout=60, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise ValueError(f"Not a readable video: {r.stderr.decode(errors='replace').strip()[:200]}")
    data = json.loads(r.stdout or b"{}")
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        raise ValueError("This file has no video stream.")
    v = vids[0]
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate")) or Fraction(30)
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else math.ceil(duration * fps)
    audio = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
    return VideoInfo(w, h, fps, max(1, frames), duration, v.get("pix_fmt", ""), v.get("color_space", ""), audio,
                     v.get("codec_name", ""))


def _matrix(color_space: str, height: int) -> str:
    """swscale matrix name. Untagged video: HD is BT.709, SD is BT.601 (what players assume)."""
    cs = color_space or ""
    if cs in ("bt709",):
        return "bt709"
    if cs in ("bt470bg", "smpte170m", "bt601"):
        return "bt601"
    if cs.startswith("bt2020"):
        return "bt2020"
    return "bt709" if height >= 720 else "bt601"


def read_frame(path: str | Path, at: float = 0.0, info: VideoInfo | None = None) -> np.ndarray:
    """One frame (HxWx3 float32 RGB) at `at` seconds, for previews and Autopilot."""
    info = info or probe(path)
    at = max(0.0, min(at, max(0.0, info.duration - 0.1)))
    m = _matrix(info.color_space, info.height)
    cmd = [_need("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{at:.3f}", "-i", str(path), "-map", "0:V:0",
           "-frames:v", "1", "-vf", f"scale=in_color_matrix={m}:flags=bicubic+accurate_rnd+full_chroma_int",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=60, creationflags=NO_WINDOW)
    n = info.width * info.height * 3
    if r.returncode != 0 or len(r.stdout) < n:
        raise ValueError(f"Could not read a frame: {r.stderr.decode(errors='replace').strip()[:200]}")
    return np.frombuffer(r.stdout[:n], np.uint8).reshape(info.height, info.width, 3).astype(np.float32) / 255


# -- frame timing -------------------------------------------------------------
def plan_rate(info: VideoInfo, s: Settings) -> tuple[Fraction, Fraction]:
    """(output fps, source frames advanced per output frame).

    fps "2x"/"4x" multiply the frame rate; a number ("60") is a target rate. slowmo stretches time instead:
    the output plays at the source rate but every source frame becomes `slowmo` frames."""
    fps = info.fps
    if s.slowmo > 1:
        return fps, Fraction(1, s.slowmo)
    if s.fps in ("2x", "4x"):
        out = fps * int(s.fps[0])
    elif s.fps.isdigit() and int(s.fps) > fps:
        out = Fraction(int(s.fps))
    else:
        return fps, Fraction(1)
    return out, fps / out


def output_frames(n_src: int, step: Fraction) -> int:
    """Each source frame covers 1/fps seconds, the last one included, so the duration matches the audio."""
    return math.ceil(n_src / step)


# -- interpolation --------------------------------------------------------------
def _thumb(img: np.ndarray) -> np.ndarray:
    """32x32 grayscale in [0,1] from a float, 8-bit or 16-bit frame."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(img.dtype, 1.0)
    return g.astype(np.float64) / scale


def ssim_small(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM of two 32x32 grayscale thumbnails (Gaussian window), cheap enough to run on every frame pair."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


class Interpolator:
    """Each frame is used in two pairs (as B, then as A), and in several pairs at 4x, so its padded
    channels-first copy is kept for the frames seen last instead of being rebuilt every call."""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self._prepared: list[tuple[np.ndarray, np.ndarray]] = []
        self._x: np.ndarray | None = None

    def model_for(self, h: int, w: int) -> str:
        """4K-class frames estimate motion at half resolution (Practical-RIFE's UHD mode)."""
        return "rife-4.25-uhd" if max(h, w) > 2200 else "rife-4.25"

    def _chw(self, img: np.ndarray, ph: int, pw: int) -> np.ndarray:
        for src, chw in self._prepared:
            if src is img:
                return chw
        h, w = img.shape[:2]
        chw = np.empty((3, ph, pw), np.float32)
        chw[:, :h, :w] = img.transpose(2, 0, 1)
        chw[:, h:, :w] = chw[:, h - 1:h, :w]  # repeat the edge into the padding
        chw[:, :, w:] = chw[:, :, w - 1:w]
        self._prepared = (self._prepared + [(img, chw)])[-3:]
        return chw

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape[:2]
        mid = self.model_for(h, w)
        m = self.rt.catalog.get(mid).multiple
        ph, pw = -(-h // m) * m, -(-w // m) * m
        if self._x is None or self._x.shape != (1, 7, ph, pw):
            self._x = np.empty((1, 7, ph, pw), np.float32)
        x = self._x
        x[0, 0:3] = self._chw(a, ph, pw)
        x[0, 3:6] = self._chw(b, ph, pw)
        x[0, 6] = t
        y = self.rt.run(mid, x)[0, :, :h, :w].transpose(1, 2, 0)
        return np.clip(y, 0, 1)


# -- the job ------------------------------------------------------------------
def _encoder_args(codec: str, out_w: int, out_h: int, high_bit: bool, matrix: str) -> list[str]:
    tags = {"bt709": "bt709", "bt601": "smpte170m", "bt2020": "bt2020nc"}[matrix]
    prim = {"bt709": "bt709", "bt601": "smpte170m", "bt2020": "bt2020"}[matrix]
    trc = {"bt709": "bt709", "bt601": "smpte170m", "bt2020": "bt709"}[matrix]
    color = ["-colorspace", tags, "-color_primaries", prim, "-color_trc", trc, "-color_range", "tv"]
    if codec == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"] + color
    if codec == "h265":
        if has_nvenc("hevc_nvenc") and max(out_w, out_h) <= 8192:
            return ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0",
                    "-pix_fmt", "p010le" if high_bit else "yuv420p", "-tag:v", "hvc1"] + color
        return ["-c:v", "libx265", "-preset", "medium", "-crf", "19", "-pix_fmt",
                "yuv420p10le" if high_bit else "yuv420p", "-tag:v", "hvc1", "-x265-params", "log-level=error"] + color
    if has_nvenc("h264_nvenc") and max(out_w, out_h) <= 4096:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0",
                "-pix_fmt", "yuv420p", "-profile:v", "high"] + color
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p"] + color


def _audio_args(info: VideoInfo, codec: str, keep: bool) -> list[str]:
    if not keep or not info.audio:
        return ["-an"]
    ok = MOV_AUDIO if codec == "prores" else MP4_AUDIO
    if info.audio in ok:
        return ["-map", "1:a:0", "-c:a", "copy"]
    return ["-map", "1:a:0"] + (["-c:a", "pcm_s16le"] if codec == "prores" else ["-c:a", "aac", "-b:a", "192k"])


def output_size(info: VideoInfo, s: Settings) -> tuple[int, int]:
    """Upscaled size rounded down to even numbers (4:2:0 video needs even width and height)."""
    return info.width * s.upscale // 2 * 2, info.height * s.upscale // 2 * 2


Progress = Callable[[float, str], None] | None


def enhance_video(src: str | Path, dst: str | Path, s: Settings, rt: Runtime, progress: Progress = None,
                  cancel: Callable[[], bool] | None = None, report: dict | None = None,
                  preview_at: float = 1 / 3) -> Path:
    """Enhance every frame of `src` with `s` (and change the frame rate if asked), writing `dst`.
    report gets: frames_in, frames_out, fps_in, fps_out, size, encoder, cuts, before/after preview frames."""
    src, dst = Path(src), Path(dst)
    report = report if report is not None else {}
    cancel = cancel or (lambda: False)
    info = probe(src)
    out_fps, step = plan_rate(info, s)
    interp = step != 1
    if interp and not (rt.catalog.available("rife-4.25") and rt.catalog.available("rife-4.25-uhd")):
        raise RuntimeError("The frame interpolation model (RIFE) is missing. Run tools/convert/convert.py.")
    ow, oh = output_size(info, s)
    matrix_in = _matrix(info.color_space, info.height)
    matrix_out = "bt2020" if matrix_in == "bt2020" else ("bt709" if oh >= 720 else "bt601")
    bits16 = info.high_bit and s.vcodec in ("h265", "prores")
    pix_in, dtype, maxval = ("rgb48le", np.uint16, 65535.0) if bits16 else ("rgb24", np.uint8, 255.0)
    n_in = info.frames
    n_out = output_frames(n_in, step)
    enc = _encoder_args(s.vcodec, ow, oh, bits16, matrix_out)
    keep_audio = s.slowmo <= 1
    report.update(frames_in=n_in, frames_out=n_out, fps_in=float(info.fps), fps_out=float(out_fps), size=(ow, oh),
                  encoder=enc[1], cuts=0, audio=bool(keep_audio and info.audio))

    ff = _need("ffmpeg")
    dec_cmd = [ff, "-v", "error", "-nostdin", "-i", str(src), "-map", "0:V:0", "-fps_mode", "cfr", "-r",
               str(info.fps), "-vf", f"scale=in_color_matrix={matrix_in}:flags=bicubic+accurate_rnd+full_chroma_int",
               "-f", "rawvideo", "-pix_fmt", pix_in, "-"]
    enc_cmd = [ff, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{ow}x{oh}",
               "-framerate", str(out_fps), "-i", "-", "-i", str(src), "-map", "0:v:0",
               "-vf", f"scale=out_color_matrix={matrix_out}:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int",
               *enc, *_audio_args(info, s.vcodec, keep_audio), "-map_metadata", "1", "-map_chapters", "-1",
               "-movflags", "+faststart"]
    if keep_audio and info.audio:
        enc_cmd.append("-shortest")
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    enc_cmd.append(str(tmp))
    log.info("video %s: %dx%d %.3f fps -> %dx%d %.3f fps, %s", src.name, info.width, info.height, info.fps,
             ow, oh, out_fps, enc[1])

    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    encp = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    frame_bytes = info.width * info.height * 3 * np.dtype(dtype).itemsize
    inq: queue.Queue = queue.Queue(maxsize=6)
    outq: queue.Queue = queue.Queue(maxsize=6)
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                buf = dec.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                inq.put(np.frombuffer(buf, dtype).reshape(info.height, info.width, 3))
        except Exception as e:  # pragma: no cover - pipe errors surface via the return code below
            errors.append(e)
        finally:
            inq.put(None)

    def writer():
        try:
            while True:
                f = outq.get()
                if f is None:
                    break
                encp.stdin.write(f.tobytes())
        except Exception as e:
            errors.append(e)
            stop.set()
        finally:
            try:
                encp.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)]
    for t in threads:
        t.start()

    steps = s.steps()
    interp_fn = Interpolator(rt) if interp else None
    preview_idx = min(n_in - 1, int(n_in * preview_at))
    stage = LABELS[steps[0][0]] if steps else "Adding frames"
    done_out = 0
    cuts = 0
    at_source = False  # set when interpolating the big enhanced frames ran out of GPU memory

    def to_out(img: np.ndarray) -> np.ndarray:
        img = img[:oh, :ow]
        return (np.clip(img, 0, 1) * maxval + 0.5).astype(dtype)

    def emit(img: np.ndarray) -> None:
        nonlocal done_out
        while True:
            if stop.is_set() or cancel():
                raise Cancelled()
            try:
                outq.put(to_out(img), timeout=0.5)
                break
            except queue.Full:
                continue
        done_out += 1

    def enhance(raw: np.ndarray, i: int) -> np.ndarray:
        img = raw.astype(np.float32) / maxval

        def tick(frac, label):
            if progress:
                progress(min(0.999, (done_out + frac * (1 / step if interp else 1)) / n_out),
                         stage if label == "Saving" else label)

        out = process(img, s, rt, tick, cancel) if steps else img
        if i == preview_idx:
            report["before"], report["after"] = img, out
        return out

    try:
        prev_raw = prev = prev_thumb = None
        k = 0  # next output frame; its source position is k * step
        i = -1
        while True:
            if cancel():
                raise Cancelled()
            raw = inq.get()
            if raw is None:
                break
            i += 1
            cur = enhance(raw, i)
            thumb = _thumb(raw) if interp else None
            if prev is not None:
                cut = interp and ssim_small(prev_thumb, thumb) < SCENE_CUT_SSIM
                cuts += bool(cut)
                while k * step < i:  # output frames between source frames i-1 and i
                    frac = float(k * step - (i - 1))
                    if frac < 1e-6 or cut:
                        emit(prev)
                    elif not at_source:
                        try:
                            emit(interp_fn(prev, cur, frac))
                        except Exception as e:
                            if not is_oom(e):
                                raise
                            log.warning("interpolation at %dx%d ran out of GPU memory; interpolating before "
                                        "enhancing instead", ow, oh)
                            rt.release()
                            at_source = True
                    if at_source and frac >= 1e-6 and not cut:
                        mid = interp_fn(prev_raw.astype(np.float32) / maxval, raw.astype(np.float32) / maxval, frac)
                        emit(process(mid, s, rt, None, cancel) if steps else mid)
                    k += 1
                    if progress:
                        progress(min(0.999, done_out / n_out), "Adding frames" if interp and frac > 1e-6 else stage)
            prev_raw, prev, prev_thumb = raw, cur, thumb
            if not interp:
                emit(cur)
                k += 1
                if progress:
                    progress(min(0.999, done_out / max(n_out, i + 1)), stage)
        if prev is not None and interp:  # the last source frame holds for its own frame time
            while k * step < i + 1:
                emit(prev)
                k += 1
        if i < 0:
            raise ValueError("No frames could be decoded from this video.")
        if "before" not in report:  # fewer frames than the container said
            report["before"], report["after"] = prev_raw.astype(np.float32) / maxval, prev
        outq.put(None)
        threads[1].join()
        encp.wait()
        dec.wait()
        if errors or encp.returncode != 0:
            msg = encp.stderr.read().decode(errors="replace").strip()[-300:] or str(errors[0] if errors else "")
            raise RuntimeError(f"Encoding failed: {msg}")
        os.replace(tmp, dst)
        report.update(frames_in=i + 1, frames_out=done_out, cuts=cuts, interp_at_source=at_source)
        if progress:
            progress(1.0, "Done")
        return dst
    except BaseException:
        stop.set()
        for p in (dec, encp):
            try:
                p.kill()
            except OSError:
                pass
        try:
            while True:
                inq.get_nowait()
        except queue.Empty:
            pass
        try:
            outq.put_nowait(None)
        except queue.Full:
            pass
        for p in (dec, encp):
            p.wait()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        for p in (dec, encp):
            for pipe in (p.stdout, p.stderr):
                if pipe:
                    pipe.close()
