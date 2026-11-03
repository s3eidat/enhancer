"""Convert the official PyTorch weights to ONNX (fp32 + fp16) and write models/manifest.json.

Dev only: needs torch + onnx + onnxconverter-common. The app itself never imports torch.

    .venv/Scripts/python -I tools/convert/convert.py [--only id1,id2] [--weights DIR]

Every exported model takes and returns RGB float in [0, 1], NCHW. Each export is checked against
PyTorch on the same input and the max abs difference is printed. fp16 copies are opt-in (--fp16): on
DirectML they were slower for Real-ESRGAN and numerically broken for NAFNet and GFPGAN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convert"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnxconverter_common import float16  # noqa: E402
from torch import nn  # noqa: E402

from vendor.gfpganv1_clean_arch import GFPGANv1Clean  # noqa: E402
from vendor.NAFNet_arch import NAFNet  # noqa: E402
from vendor.rrdbnet_arch import RRDBNet  # noqa: E402
from vendor.srvgg_arch import SRVGGNetCompact  # noqa: E402

MODELS_DIR = ROOT / "models"

ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download"
GDRIVE = "https://drive.google.com/file/d/{}/view"
LIC = {
    "esrgan": {"license": "BSD-3-Clause", "project": "https://github.com/xinntao/Real-ESRGAN",
               "credit": "Real-ESRGAN, Xintao Wang et al."},
    "nafnet": {"license": "MIT", "project": "https://github.com/megvii-research/NAFNet",
               "credit": "NAFNet, Liangyu Chen et al. (Megvii)"},
    "gfpgan": {"license": "Apache-2.0", "project": "https://github.com/TencentARC/GFPGAN",
               "credit": "GFPGAN v1.4, Xintao Wang et al. (Tencent ARC)",
               "note": "Clean architecture only. Weights trained on FFHQ (CC BY-NC-SA 4.0 dataset)."},
    "yunet": {"license": "MIT", "project": "https://github.com/opencv/opencv_zoo",
              "credit": "YuNet face detector, OpenCV Zoo"},
}

NAF_DENOISE = dict(enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
NAF_DEBLUR = dict(enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])

# id -> spec. kind: what the pipeline uses it for. scale: output/input size. multiple: input H/W must divide by it.
SPECS = {
    "realesrgan-x4": dict(kind="upscale", scale=4, multiple=1, lic="esrgan", weights="RealESRGAN_x4plus.pth",
                          source=f"{ESRGAN}/v0.1.0/RealESRGAN_x4plus.pth",
                          build=lambda: RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)),
    "realesrgan-x2": dict(kind="upscale", scale=2, multiple=2, lic="esrgan", weights="RealESRGAN_x2plus.pth",
                          source=f"{ESRGAN}/v0.2.1/RealESRGAN_x2plus.pth",
                          build=lambda: RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)),
    "realesrgan-anime-x4": dict(kind="upscale", scale=4, multiple=1, lic="esrgan",
                                weights="RealESRGAN_x4plus_anime_6B.pth",
                                source=f"{ESRGAN}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                                build=lambda: RRDBNet(3, 3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)),
    "realesr-general-x4": dict(kind="upscale", scale=4, multiple=1, lic="esrgan", weights="realesr-general-x4v3.pth",
                               source=f"{ESRGAN}/v0.2.5.0/realesr-general-x4v3.pth",
                               build=lambda: SRVGGNetCompact(3, 3, num_feat=64, num_conv=32, upscale=4,
                                                             act_type="prelu")),
    "realesr-general-wdn-x4": dict(kind="upscale", scale=4, multiple=1, lic="esrgan",
                                   weights="realesr-general-wdn-x4v3.pth",
                                   source=f"{ESRGAN}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                                   build=lambda: SRVGGNetCompact(3, 3, num_feat=64, num_conv=32, upscale=4,
                                                                 act_type="prelu")),
    "nafnet-denoise-32": dict(kind="denoise", scale=1, multiple=16, lic="nafnet", weights="NAFNet-SIDD-width32.pth",
                              source=GDRIVE.format("1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ"),
                              build=lambda: NAFNet(img_channel=3, width=32, **NAF_DENOISE)),
    "nafnet-denoise-64": dict(kind="denoise", scale=1, multiple=16, lic="nafnet", weights="NAFNet-SIDD-width64.pth",
                              source=GDRIVE.format("14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR"),
                              build=lambda: NAFNet(img_channel=3, width=64, **NAF_DENOISE)),
    "nafnet-deblur-32": dict(kind="sharpen", scale=1, multiple=16, lic="nafnet", weights="NAFNet-GoPro-width32.pth",
                             source=GDRIVE.format("1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj"),
                             build=lambda: NAFNet(img_channel=3, width=32, **NAF_DEBLUR)),
    "nafnet-deblur-64": dict(kind="sharpen", scale=1, multiple=16, lic="nafnet", weights="NAFNet-GoPro-width64.pth",
                             source=GDRIVE.format("1S0PVRbyTakYY9a82kujgZLbMihfNBLfC"),
                             build=lambda: NAFNet(img_channel=3, width=64, **NAF_DEBLUR)),
    "gfpgan-1.4": dict(kind="face", scale=1, multiple=512, fixed=512, lic="gfpgan", weights="GFPGANv1.4.pth",
                       source="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                       build=lambda: GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,
                                                   decoder_load_path=None, fix_decoder=False, num_mlp=8,
                                                   input_is_latent=True, different_w=True, narrow=1,
                                                   sft_half=True)),
}


class NoPad(nn.Module):
    """NAFNet pads to a multiple of 16 inside forward(); the engine pads instead, so export without it."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        net.check_image_size = lambda x: x

    def forward(self, x):
        return self.net(x)


class FaceWrap(nn.Module):
    """GFPGAN works in [-1, 1]; wrap it so every model is [0, 1] in and out."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x * 2 - 1, return_rgb=False, randomize_noise=False)[0]
        return (out + 1) * 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(state, dict) and key in state:
            return state[key]
    return state


def test_input(size: int) -> np.ndarray:
    """A smooth image-like tile (gradients + edges + mild noise) so fp16 is tested on realistic values."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    rgb = np.stack([xx, yy, (np.sin(xx * 20) * np.cos(yy * 13) + 1) / 2])
    rgb[:, size // 3: size // 2, :] *= 0.2
    rgb += np.random.default_rng(0).normal(0, 0.03, rgb.shape)
    return np.clip(rgb, 0, 1)[None].astype(np.float32)


def run_ort(path: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x})[0]


def convert(mid: str, spec: dict, weights_dir: Path, fp16: bool = False) -> dict:
    wpath = weights_dir / spec["weights"]
    net = spec["build"]()
    net.load_state_dict(load_state(wpath), strict=True)
    if spec["kind"] in ("denoise", "sharpen"):
        net = NoPad(net)
    if spec["kind"] == "face":
        net = FaceWrap(net)
    net.eval()

    fixed = spec.get("fixed")
    size = fixed or 64
    x = test_input(size)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()

    out32 = MODELS_DIR / f"{mid}.onnx"
    dyn = None if fixed else {"input": {2: "h", 3: "w"}, "output": {2: "oh", 3: "ow"}}
    torch.onnx.export(net, torch.from_numpy(x), str(out32), input_names=["input"], output_names=["output"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    m = onnx.load(str(out32))
    onnx.checker.check_model(m)
    diff32 = float(np.abs(run_ort(out32, x) - ref).max())

    files = {"fp32": {"file": out32.name, "bytes": out32.stat().st_size, "sha256": sha256(out32), "max_diff": diff32}}
    msg = f"{mid:24s} fp32 diff {diff32:.2e}"
    if fp16:
        out16 = MODELS_DIR / f"{mid}.fp16.onnx"
        onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), str(out16))
        y16 = run_ort(out16, x)
        diff16 = float(np.abs(y16 - ref).max()) if np.isfinite(y16).all() else float("inf")
        files["fp16"] = {"file": out16.name, "bytes": out16.stat().st_size, "sha256": sha256(out16), "max_diff": diff16}
        msg += f"  fp16 diff {diff16:.2e}"

    if not fixed:  # dynamic shapes really work: a non-square size gives the right output size
        h, w = 48, 80
        y = run_ort(out32, np.zeros((1, 3, h, w), np.float32))
        assert y.shape == (1, 3, h * spec["scale"], w * spec["scale"]), y.shape

    print(msg, flush=True)
    entry = {k: spec[k] for k in ("kind", "scale", "multiple")}
    if fixed:
        entry["fixed"] = fixed
    entry.update(source=spec["source"], weights_sha256=sha256(wpath), **LIC[spec["lic"]])
    entry["files"] = files
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--fp16", action="store_true", help="also write fp16 copies (not used by the app)")
    ap.add_argument("--weights", default=str(ROOT / ".cache" / "weights"))
    args = ap.parse_args()
    weights = Path(args.weights)
    MODELS_DIR.mkdir(exist_ok=True)
    manifest_path = MODELS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}
    wanted = [m for m in args.only.split(",") if m] or list(SPECS)
    for mid in wanted:
        t = time.time()
        manifest["models"][mid] = convert(mid, SPECS[mid], weights, args.fp16)
        print(f"  ({time.time() - t:.0f}s)", flush=True)

    yunet = weights / "face_detection_yunet_2023mar.onnx"
    if yunet.exists():
        dst = MODELS_DIR / "yunet.onnx"
        dst.write_bytes(yunet.read_bytes())
        manifest["models"]["yunet"] = {
            "kind": "detect", "scale": 1, "multiple": 1, **LIC["yunet"],
            "source": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                      "face_detection_yunet_2023mar.onnx",
            "files": {"fp32": {"file": dst.name, "bytes": dst.stat().st_size, "sha256": sha256(dst)}},
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
