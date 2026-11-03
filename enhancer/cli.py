"""Command line: python -m enhancer.cli photo.jpg [more...] -o OUTDIR --upscale 4 --denoise 0.7 --faces 0.8"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import imageio
from .autopilot import suggest
from .pipeline import Settings, process
from .runtime import Runtime


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="enhancer")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--upscale", type=int, default=2, choices=[1, 2, 4])
    ap.add_argument("--model", default="standard", choices=["standard", "art", "fast"])
    ap.add_argument("--detail", type=float, default=1.0)
    ap.add_argument("--denoise", type=float, default=0.0)
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--faces", type=float, default=0.0)
    ap.add_argument("--quality", default="best", choices=["best", "fast"])
    ap.add_argument("--format", default="png", choices=list(imageio.FORMATS))
    ap.add_argument("--auto", action="store_true", help="let Autopilot pick the settings")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    rt = Runtime(device="cpu" if a.cpu else "auto")
    print(f"device: {rt.device_name}", flush=True)
    failed = 0
    for src in map(Path, a.inputs):
        t0 = time.time()
        try:
            loaded = imageio.load(src)
            s = Settings.from_dict({k: getattr(a, k) for k in
                                    ("upscale", "model", "detail", "denoise", "sharpen", "faces", "quality", "format")})
            if a.auto:
                tip = suggest(loaded.rgb, rt)
                s = Settings.from_dict({**s.to_dict(), **tip["settings"]})
                print("  autopilot:", "; ".join(tip["reasons"]))
            report: dict = {}
            last = [-1]

            def show(frac, stage):
                pct = int(frac * 100)
                if pct != last[0]:
                    last[0] = pct
                    print(f"\r  {src.name}: {stage} {pct}%   ", end="", flush=True)

            out = process(loaded.rgb, s, rt, show, report=report)
            alpha = loaded.alpha
            if alpha is not None and s.upscale > 1:
                from .ops import resize
                alpha = resize(alpha[:, :, None].repeat(3, 2), out.shape[1], out.shape[0])[:, :, 0]
            dst = imageio.output_path(src, Path(a.out), s.format, s.suffix)
            imageio.save(out, dst, s.format, alpha=alpha, exif=loaded.exif, icc=loaded.icc)
            extra = f", {report['faces']} faces" if "faces" in report else ""
            print(f"\r  {src.name} -> {dst} ({out.shape[1]}x{out.shape[0]}, {time.time() - t0:.1f}s{extra})")
        except Exception as e:  # one bad file must not stop the batch
            failed += 1
            print(f"\r  {src.name}: FAILED: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
