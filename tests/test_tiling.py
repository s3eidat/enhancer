import numpy as np
import pytest

from enhancer.tiling import Cancelled, axis_weights, run_tiled, tile_count, tile_starts


def identity(x):
    return x.copy()


def nearest(scale):
    return lambda x: x.repeat(scale, axis=2).repeat(scale, axis=3)


def rand(h, w, seed=0):
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


@pytest.mark.parametrize("h,w", [(1, 1), (7, 13), (64, 64), (100, 37), (513, 1031)])
@pytest.mark.parametrize("multiple", [1, 2, 16])
def test_identity_is_exact(h, w, multiple):
    img = rand(h, w)
    out = run_tiled(identity, img, 1, tile=64, overlap=8, multiple=multiple)
    np.testing.assert_allclose(out, img, atol=1e-5)


@pytest.mark.parametrize("scale", [2, 4])
def test_upscale_matches_whole_image(scale):
    img = rand(150, 211, seed=1)
    out = run_tiled(nearest(scale), img, scale, tile=48, overlap=6, multiple=2)
    ref = img.repeat(scale, 0).repeat(scale, 1)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_every_tile_has_the_same_shape():
    shapes = set()

    def spy(x):
        shapes.add(x.shape)
        return x

    run_tiled(spy, rand(300, 170), 1, tile=64, overlap=8, multiple=16)
    assert shapes == {(1, 3, 64, 64)}


def test_weights_sum_to_one():
    starts = tile_starts(1000, 128, 16)
    ws = axis_weights(starts, 128, 1000, 16)
    total = np.zeros(1000)
    for s, w in zip(starts, ws):
        total[s:s + 128] += w
    np.testing.assert_allclose(total, 1, atol=1e-6)


def test_progress_and_count():
    seen = []
    run_tiled(identity, rand(200, 300), 1, tile=64, overlap=8, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == seen[-1][1] == tile_count(200, 300, 64, 8)


def test_cancel():
    with pytest.raises(Cancelled):
        run_tiled(identity, rand(200, 200), 1, tile=64, overlap=8, cancel=lambda: True)
