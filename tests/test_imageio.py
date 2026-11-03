import numpy as np
from PIL import Image

from enhancer import imageio
from tests.conftest import photo_like


def _jpeg_with_exif(path, orientation=6):
    im = Image.fromarray(imageio.to_uint8(photo_like(40, 60)))
    exif = Image.Exif()
    exif[0x0110] = "TestCam"
    exif[imageio.ORIENTATION] = orientation
    im.save(path, "JPEG", exif=exif.tobytes(), icc_profile=b"\0fake-icc")
    return path


def test_orientation_applied_once_and_exif_kept(tmp_path):
    src = _jpeg_with_exif(tmp_path / "a.jpg")
    loaded = imageio.load(src)
    assert loaded.rgb.shape == (60, 40, 3)  # orientation 6 = rotated 90 degrees
    out = imageio.save(loaded.rgb, tmp_path / "b.jpg", "jpg", exif=loaded.exif, icc=loaded.icc)
    with Image.open(out) as im:
        assert im.size == (40, 60)
        ex = im.getexif()
        assert ex[0x0110] == "TestCam" and ex[imageio.ORIENTATION] == 1
        assert im.info["icc_profile"] == b"\0fake-icc"


def test_alpha_round_trip(tmp_path):
    rgba = np.zeros((20, 30, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[:, :15, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    loaded = imageio.load(tmp_path / "a.png")
    assert loaded.alpha is not None and loaded.alpha[0, 0] == 1 and loaded.alpha[0, 29] == 0
    imageio.save(loaded.rgb, tmp_path / "b.png", "png", alpha=loaded.alpha)
    assert Image.open(tmp_path / "b.png").mode == "RGBA"


def test_16bit_round_trip_and_unicode_path(tmp_path):
    img = photo_like(32, 48)
    p = imageio.save(img, tmp_path / "صورة.png", "png16")
    loaded = imageio.load(p)
    assert loaded.bits == 16
    np.testing.assert_allclose(loaded.rgb, img, atol=1 / 65535 + 1e-6)


def test_output_path_never_overwrites(tmp_path):
    a = imageio.output_path(tmp_path / "x.jpg", tmp_path, "png")
    a.write_bytes(b"1")
    b = imageio.output_path(tmp_path / "x.jpg", tmp_path, "png")
    assert a != b and b.name == "x_enhanced-2.png"
