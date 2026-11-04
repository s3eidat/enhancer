"""Install the browser version into an S3eidat Tools tree as repos/ai-enhancer and list it in _kit/tools.json.

    .venv/Scripts/python -I tools/stage_site_tool.py "<path to S3eidat-Tools or a scratch copy>"

This only copies files. It does not push anything: after it, run from that tree's scripts folder
    python sync_kit.py ai-enhancer ; python check.py ai-enhancer ; python smoke.py ai-enhancer
and only then, with the owner's OK, publish.py ai-enhancer. The tool ships its own lib/ and models/ because they are not
in _vendor, so its tools.json entry lists none and sync_kit leaves both folders alone.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "ai-enhancer"
ENTRY = {
    "slug": SLUG,
    "name": "AI Photo Enhancer",
    "category": "Images",
    "tagline": "Upscale, denoise and sharpen photos with AI, free",
    "spec": "Free alternative to paid AI enhancers. Real-ESRGAN upscale 2x/4x (Photo, Art, Fast), NAFNet noise removal "
            "and sharpening with onnxruntime-web on WebGPU, WASM fallback. Built outside the kit from enhancer/browser; "
            "ships its own lib/ and models/.",
    "libs": [],
}
OWN = ["index.html", "app.js", "app.css", "worker.js", "tiling.js", "README.md", "manifest.webmanifest"]


def main(tree: Path) -> None:
    src = ROOT / "browser"
    dst = tree / "repos" / SLUG
    dst.mkdir(parents=True, exist_ok=True)
    for f in OWN:
        shutil.copy2(src / f, dst / f)
    for folder in ("lib", "models"):
        if (dst / folder).exists():
            shutil.rmtree(dst / folder)
        shutil.copytree(src / folder, dst / folder)
    tools = tree / "_kit" / "tools.json"
    text = tools.read_text(encoding="utf-8")
    if not any(t["slug"] == SLUG for t in json.loads(text)):
        # add text at the end instead of re-dumping, so the file's formatting and every other entry stay untouched
        end = text.rstrip().rfind("]")
        head = text[:end].rstrip()
        tools.write_text(f"{head},\n{json.dumps(ENTRY, ensure_ascii=False)}\n]\n", encoding="utf-8", newline="\n")
        json.loads(tools.read_text(encoding="utf-8"))
    print("staged", dst)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
