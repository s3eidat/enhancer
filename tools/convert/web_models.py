"""Make the browser copies of the models: smaller stored weights, maths still in fp32.

Each large fp32 weight is stored smaller and expanded back to fp32 by a few nodes that run once at load, so the
network computes as before with rounded weights. (Converting the whole graph to fp16 broke NAFNet on DirectML; this
does not touch the activations.) Most models store fp16. The two NAFNets store 8 bits per weight with a range per
output channel, which keeps the tool under the S3eidat Tools size cap (77 dB and 61 dB against fp32 on a real photo
crop; the Photo model only reached 45 dB, so it stays fp16). Each result is checked against the fp32 model on the CPU.

    .venv/Scripts/python -I tools/convert/web_models.py [--out browser/models]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
WEB = ["realesr-general-x4", "realesrgan-x4", "realesrgan-anime-x4", "nafnet-denoise-32", "nafnet-deblur-32"]
U8_STORAGE = {"nafnet-denoise-32", "nafnet-deblur-32"}
MIN_ELEMS = 1024  # small tensors (biases, norms) stay fp32: no size gain, and they are the sensitive ones


def fp16_storage(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    keep, casts = [], []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < MIN_ELEMS:
            keep.append(init)
            continue
        arr = numpy_helper.to_array(init)
        half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__cast"))
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return model


def u8_storage(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    keep, nodes = [], []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < MIN_ELEMS:
            keep.append(init)
            continue
        a, n = numpy_helper.to_array(init), init.name
        axes = tuple(range(1, a.ndim))
        lo = a.min(axis=axes, keepdims=True)
        step = np.maximum(a.max(axis=axes, keepdims=True) - lo, 1e-12) / 255
        keep += [numpy_helper.from_array(np.round((a - lo) / step).astype(np.uint8), n + "__u8"),
                 numpy_helper.from_array(step.astype(np.float32), n + "__step"),
                 numpy_helper.from_array(lo.astype(np.float32), n + "__lo")]
        nodes += [helper.make_node("Cast", [n + "__u8"], [n + "__f"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", [n + "__f", n + "__step"], [n + "__m"]),
                  helper.make_node("Add", [n + "__m", n + "__lo"], [n])]
    del g.initializer[:]
    g.initializer.extend(keep)
    old = list(g.node)
    del g.node[:]
    g.node.extend(nodes + old)
    return model


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "browser" / "models"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text())["models"]
    web = {}
    rng = np.random.default_rng(0)
    for mid in WEB:
        info = manifest[mid]
        src = ROOT / "models" / info["files"]["fp32"]["file"]
        dst = out / f"{mid}.onnx"
        store = u8_storage if mid in U8_STORAGE else fp16_storage
        onnx.save(store(onnx.load(str(src))), str(dst))
        ch = info.get("channels", 3)
        size = max(64, info["multiple"])
        yy, xx = np.mgrid[0:size, 0:size] / size
        img = np.stack([xx, yy, (np.sin(xx * 17) + 1) / 2]) + rng.normal(0, 0.02, (3, size, size))
        x = np.clip(img, 0, 1)
        if ch == 7:
            x = np.concatenate([x, np.roll(x, 5, axis=2), np.full((1, size, size), 0.5)])
        x = x[None].astype(np.float32)
        ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
        got = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
        diff = np.abs(ref - got)
        print(f"{mid:22s} {src.stat().st_size / 1e6:6.1f} MB -> {dst.stat().st_size / 1e6:6.1f} MB  "
              f"mean diff {diff.mean():.1e} max {diff.max():.1e}", flush=True)
        assert diff.mean() < 2e-3 and diff.max() < 0.05, mid
        web[mid] = {k: info[k] for k in ("kind", "scale", "multiple", "license", "credit", "project")
                    if k in info}
        web[mid].update(file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst), channels=ch,
                        max_diff_vs_fp32=float(diff.max()))
    (out / "models.json").write_text(json.dumps({"models": web}, indent=2) + "\n")
    print("wrote", out / "models.json")


if __name__ == "__main__":
    main()
