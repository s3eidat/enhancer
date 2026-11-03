"""GPU output must match CPU output. fp16 on DirectML once silently produced garbage for NAFNet/GFPGAN."""
import numpy as np
import pytest

from enhancer.runtime import Runtime
from tests.conftest import photo_like

MODELS = [("realesrgan-x4", 64), ("realesrgan-x2", 64), ("realesrgan-anime-x4", 64), ("realesr-general-x4", 64),
          ("nafnet-denoise-32", 128), ("nafnet-denoise-64", 128), ("nafnet-deblur-32", 128),
          ("nafnet-deblur-64", 128), ("gfpgan-1.4", 512), ("rife-4.25", 128), ("rife-4.25-uhd", 256)]


@pytest.fixture(scope="module")
def pair(catalog):
    gpu = Runtime(catalog)
    if not gpu.use_gpu:
        pytest.skip("no DirectML GPU")
    return gpu, Runtime(catalog, device="cpu")


@pytest.mark.parametrize("mid,size", MODELS)
def test_gpu_matches_cpu(pair, mid, size):
    gpu, cpu = pair
    if not gpu.catalog.available(mid):
        pytest.skip(f"{mid} not converted")
    x = photo_like(size, size, noise=0.01).transpose(2, 0, 1)[None].copy()
    if gpu.catalog.get(mid).kind == "interp":  # two frames 4 px apart + the time channel
        x = np.concatenate([x, np.roll(x, 4, axis=3), np.full_like(x[:, :1], 0.5)], axis=1)
    g, c = gpu.run(mid, x), cpu.run(mid, x)
    assert gpu.use_gpu, gpu.notice
    diff = np.abs(g - c)
    # DirectML picks different conv kernels, so tiny differences are normal; broken fp16 gave a mean of ~1.
    assert diff.mean() < 0.002 and diff.max() < 0.1, f"{mid} differs on GPU: mean {diff.mean()}, max {diff.max()}"
