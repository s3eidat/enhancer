import numpy as np
import pytest

from enhancer.models import Catalog
from enhancer.runtime import Runtime


@pytest.fixture(scope="session")
def catalog():
    c = Catalog()
    if not c.available("realesr-general-x4"):
        pytest.skip("models not converted (run tools/convert/convert.py)")
    return c


@pytest.fixture(scope="session")
def rt(catalog):
    return Runtime(catalog)


def photo_like(h=96, w=128, noise=0.0, seed=0):
    """Smooth gradients + hard edges + optional Gaussian noise, float32 HxWx3 in [0,1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([xx / w, yy / h, 0.5 + 0.4 * np.sin(xx / 7) * np.cos(yy / 11)], axis=2)
    img[h // 3:h // 2, :, :] *= 0.3
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)
