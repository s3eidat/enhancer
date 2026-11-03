"""Fill browser/ with the files that are not in git: onnxruntime-web, the fp16-storage models and (for local runs)
the S3eidat Tools kit.

    .venv/Scripts/python -I tools/build_browser.py

onnxruntime-web comes from the official npm tarball in .cache/ortweb (sha512 checked against the registry when it
was downloaded). The kit is copied from a built S3eidat Tools repo; when the tool moves into S3eidat-Tools,
scripts/sync_kit.py writes kit/ instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "browser"
ORT_VERSION = "1.30.0"
ORT_DIST = ROOT / ".cache" / "ortweb" / "x" / "package" / "dist"
ORT_FILES = ["ort.webgpu.min.js", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]
KIT_FROM = ROOT.parent / "S3eidat-Tools" / "repos" / "image-upscale" / "kit"


def main() -> None:
    lib = B / "lib"
    lib.mkdir(exist_ok=True)
    for f in ORT_FILES:
        shutil.copy2(ORT_DIST / f, lib / f)
    (lib / "ort.LICENSE").write_text(
        f"onnxruntime-web {ORT_VERSION}, unmodified build files ({', '.join(ORT_FILES)})\n"
        "https://github.com/microsoft/onnxruntime, MIT License, Copyright (c) Microsoft Corporation.\n",
        encoding="utf-8")
    models = json.loads((B / "models" / "models.json").read_text()) if (B / "models" / "models.json").exists() else None
    if not models or not all((B / "models" / m["file"]).exists() for m in models["models"].values()):
        subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "convert" / "web_models.py")], check=True)
    if KIT_FROM.exists():
        shutil.copytree(KIT_FROM, B / "kit", dirs_exist_ok=True)
        shutil.copy2(KIT_FROM.parent.parent.parent / "_kit" / "sw.js", B / "sw.js")  # offline support, as sync_kit does
    else:
        print("kit not found at", KIT_FROM, "- the page needs kit/kit.js and kit/kit.css to run")
    size = sum(p.stat().st_size for p in B.rglob("*") if p.is_file())
    print(f"browser/ ready: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
