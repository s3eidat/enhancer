import subprocess
from fractions import Fraction

import numpy as np
import pytest

from enhancer import video
from enhancer.pipeline import Settings
from enhancer.tiling import Cancelled
from tests.conftest import photo_like

needs_ffmpeg = pytest.mark.skipif(video.ffmpeg_path() is None, reason="ffmpeg not installed")


def info(fps=24, frames=24):
    return video.VideoInfo(320, 180, Fraction(fps), frames, frames / fps, "yuv420p", "", "aac", "h264")


@pytest.mark.parametrize("fps,slowmo,src,out,step", [
    ("off", 1, 24, 24, 1), ("2x", 1, 24, 48, Fraction(1, 2)), ("4x", 1, 30, 120, Fraction(1, 4)),
    ("60", 1, 24, 60, Fraction(2, 5)), ("30", 1, 60, 60, 1),  # a target rate never lowers the rate
    ("2x", 4, 24, 24, Fraction(1, 4)),  # slow motion wins and keeps the source rate
])
def test_plan_rate(fps, slowmo, src, out, step):
    s = Settings.from_dict({"fps": fps, "slowmo": slowmo})
    assert video.plan_rate(info(src), s) == (Fraction(out), step)


def test_output_frames_keep_duration():
    assert video.output_frames(24, Fraction(1)) == 24
    assert video.output_frames(24, Fraction(1, 2)) == 48
    assert video.output_frames(24, Fraction(2, 5)) == 60  # 24 fps -> 60 fps, same 1 s


def test_scene_cut_detection():
    a = photo_like(90, 160)
    assert video.ssim_small(video._thumb(a), video._thumb(a)) > 0.99
    assert video.ssim_small(video._thumb(a), video._thumb(np.roll(a, 3, axis=1))) > 0.5  # motion, not a cut
    b = np.random.default_rng(1).random((90, 160, 3)).astype(np.float32)
    assert video.ssim_small(video._thumb(a), video._thumb(b)) < video.SCENE_CUT_SSIM


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if video.ffmpeg_path() is None:
        pytest.skip("ffmpeg not installed")
    p = tmp_path_factory.mktemp("v") / "clip.mp4"
    subprocess.run([video.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=1",
                    "-f", "lavfi", "-i", "sine=f=440:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(p)], check=True)
    return p


@needs_ffmpeg
def test_probe(clip):
    i = video.probe(clip)
    assert (i.width, i.height, i.fps, i.audio) == (320, 180, 24, "aac")
    assert 23 <= i.frames <= 25


@needs_ffmpeg
def test_upscale_and_double_fps_keeps_audio(clip, tmp_path, rt):
    dst = tmp_path / "out.mp4"
    report = {}
    s = Settings.from_dict({"upscale": 2, "model": "fast", "fps": "2x"})
    video.enhance_video(clip, dst, s, rt, report=report)
    o = video.probe(dst)
    assert (o.width, o.height) == (640, 360)
    assert o.fps == 48 and o.audio == "aac"
    assert abs(o.frames - 2 * report["frames_in"]) <= 1
    assert abs(o.duration - 1.0) < 0.1
    assert report["before"].shape == (180, 320, 3) and report["after"].shape == (360, 640, 3)
    assert not list(tmp_path.glob("*.part*"))


@needs_ffmpeg
def test_interpolated_frame_is_between_neighbours(clip, tmp_path, rt):
    """A frame RIFE made should look like its two neighbours (the real frames), not like a black or random one."""
    dst = tmp_path / "smooth.mp4"
    video.enhance_video(clip, dst, Settings.from_dict({"upscale": 1, "fps": "2x"}), rt)
    a, mid, b = (video.read_frame(dst, t) for t in (10 / 48, 11 / 48, 12 / 48))
    d_neigh = np.abs(a - b).mean()
    assert np.abs(mid - (a + b) / 2).mean() < max(0.02, d_neigh)


@needs_ffmpeg
def test_slowmo_drops_audio_and_stretches(clip, tmp_path, rt):
    dst = tmp_path / "slow.mp4"
    video.enhance_video(clip, dst, Settings.from_dict({"upscale": 1, "slowmo": 2}), rt)
    o = video.probe(dst)
    assert o.fps == 24 and o.audio is None and abs(o.duration - 2.0) < 0.15


@needs_ffmpeg
def test_cancel_leaves_no_partial_file(clip, tmp_path, rt):
    dst = tmp_path / "c.mp4"
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 5

    with pytest.raises(Cancelled):
        video.enhance_video(clip, dst, Settings.from_dict({"upscale": 2, "model": "fast"}), rt, cancel=cancel)
    assert not dst.exists() and not list(tmp_path.glob("*.part*"))
