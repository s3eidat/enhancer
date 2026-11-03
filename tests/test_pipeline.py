import numpy as np
import pytest

from enhancer import ops
from enhancer.autopilot import noise_sigma, suggest
from enhancer.pipeline import Settings, process
from tests.conftest import photo_like


def test_settings_are_clamped():
    s = Settings.from_dict({"upscale": 3, "denoise": 7, "faces": -1, "model": "nope", "bogus": 1})
    assert (s.upscale, s.denoise, s.faces, s.model) == (2, 1.0, 0.0, "standard")


def test_step_order():
    s = Settings(upscale=4, denoise=0.5, sharpen=0.5, faces=0.5)
    assert [n for n, _ in s.steps()] == ["denoise", "sharpen", "upscale", "faces"]
    assert Settings(upscale=1).steps() == []


def test_order_and_strength(monkeypatch):
    calls = []

    def fake(name):
        def f(rt, img, strength, *a, **k):
            calls.append((name, strength))
            return ops.blend(img, np.ones_like(img), strength)
        return f

    monkeypatch.setattr(ops, "denoise", fake("denoise"))
    monkeypatch.setattr(ops, "sharpen", fake("sharpen"))
    img = photo_like()
    out = process(img, Settings(upscale=1, denoise=0.5, sharpen=0.25), rt=None)
    assert [c[0] for c in calls] == ["denoise", "sharpen"]
    np.testing.assert_allclose(out, 1 - (1 - img) * 0.5 * 0.75, atol=1e-6)


def test_blend_zero_is_identity():
    a, b = photo_like(), np.zeros((96, 128, 3), np.float32)
    np.testing.assert_array_equal(ops.blend(a, b, 0.0), a)


def test_noise_estimate_tracks_noise():
    clean, noisy = noise_sigma(photo_like(256, 256)), noise_sigma(photo_like(256, 256, noise=10 / 255))
    assert clean < 1.5 and 7 < noisy < 13


def test_autopilot_without_models():
    tip = suggest(photo_like(600, 800, noise=12 / 255))
    assert tip["settings"]["upscale"] == 4 and tip["settings"]["denoise"] >= 0.75


@pytest.mark.parametrize("scale,mode", [(2, "standard"), (4, "fast"), (2, "art")])
def test_real_upscale_size(rt, scale, mode):
    out = ops.upscale(rt, photo_like(40, 52), scale, mode)
    assert out.shape == (40 * scale, 52 * scale, 3) and 0 <= out.min() and out.max() <= 1


def test_real_denoise_reduces_noise(rt):
    noisy = photo_like(128, 128, noise=15 / 255)
    out = ops.denoise(rt, noisy, 1.0, "fast")
    clean = photo_like(128, 128)
    assert np.abs(out - clean).mean() < 0.5 * np.abs(noisy - clean).mean()
