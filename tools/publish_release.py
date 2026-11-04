"""Publish S3eidat Enhancer: public repo s3eidat/enhancer, the models-v1 release (what the app downloads on first run)
and the v0.1.0 release (the Windows zip and FFmpeg's source).

    .venv/Scripts/python -I tools/publish_release.py         # dry run: checks everything, changes nothing
    .venv/Scripts/python -I tools/publish_release.py --go    # really publishes. Only with the owner's OK.

The assets come from models/manifest.json, so a model removed from the manifest (and the app rebuilt) is not uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from enhancer import __version__  # noqa: E402
from enhancer.download import MODEL_URL  # noqa: E402

REPO = "s3eidat/enhancer"
NOREPLY = "161432703+s3eidat@users.noreply.github.com"
TAG_MODELS = "models-v1"
TAG_APP = f"v{__version__}"
ZIP = ROOT / "release" / "S3eidat-Enhancer-Windows.zip"
FFMPEG_SRC = ROOT / "release" / "ffmpeg-source-469aad3897.tar.gz"
DESCRIPTION = "Free, private AI photo and video enhancer: upscale, denoise, sharpen, recover faces, 60 fps. Runs on your GPU."
SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}|sk-[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}"
                    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")
NOTES_MODELS = "AI models the S3eidat Enhancer app downloads on first run. Each file's SHA-256 is in manifest.json; " \
               "sources and licenses are in THIRD-PARTY-NOTICES.md."
NOTES_APP = """Free AI photo and video enhancer for Windows. Everything runs on your own PC.

**Install:** download the zip, unzip it, run `S3eidat Enhancer.exe`. On first start it downloads its AI models (389 MB, \
optional 1.1 GB more). Windows may warn about an unknown publisher because the app is not code-signed.

**What it does:** upscale photos and videos 2x/4x, remove noise, sharpen, recover faces, raise video frame rate or make \
slow motion, in a batch queue with a before/after view. Uses the GPU through DirectML, or the CPU without one.

**Known limits:** tested on one PC (Windows 11, RTX 3050). HDR video is not converted and will look washed out. \
See the README for the full tested and not-tested list.

`ffmpeg-source-469aad3897.tar.gz` is the source of the bundled FFmpeg (GPL v3)."""


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise SystemExit(f"{' '.join(args[:5])} failed: {(r.stderr or r.stdout).strip()[:500]}")
    return r


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight() -> tuple[list[str], list[Path]]:
    """Returns (problems, model assets)."""
    problems: list[str] = []
    if run("git", "status", "--porcelain").stdout.strip():
        problems.append("the git tree has uncommitted changes")
    authors = set(run("git", "log", "--format=%ae").stdout.split())
    if authors != {NOREPLY}:
        problems.append(f"unexpected commit emails: {authors - {NOREPLY}}")
    for f in run("git", "ls-files").stdout.splitlines():
        p = ROOT / f
        if p.suffix in (".onnx", ".pth", ".pkl", ".exe", ".wasm") or p.stat().st_size > 2_000_000:
            problems.append(f"tracked file should not be public or is too big: {f}")
        elif p.suffix in (".py", ".js", ".html", ".md", ".json", ".ps1", ".spec", ".css", ".txt", ".bat") \
                and SECRET.search(p.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"looks like a secret in {f}")
    if MODEL_URL != f"https://github.com/{REPO}/releases/download/{TAG_MODELS}/":
        problems.append(f"download.MODEL_URL does not match this release: {MODEL_URL}")

    manifest_path = ROOT / "models" / "manifest.json"
    assets = [manifest_path]
    for mid, m in json.loads(manifest_path.read_text())["models"].items():
        f = m["files"]["fp32"]
        p = ROOT / "models" / f["file"]
        if not p.exists() or p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
            problems.append(f"model file wrong or missing: {f['file']}")
        assets.append(p)

    if not FFMPEG_SRC.exists():
        problems.append(f"missing {FFMPEG_SRC.name}")
    if not ZIP.exists():
        problems.append(f"missing {ZIP.name}: run build_windows.ps1")
    else:
        head_time = int(run("git", "log", "-1", "--format=%ct").stdout.strip())
        if ZIP.stat().st_mtime < head_time - 86400:
            problems.append("the zip is more than a day older than the last commit: rebuild it")
        with zipfile.ZipFile(ZIP) as z:
            names = set(z.namelist())
            pre = "S3eidat Enhancer/"
            for need in (pre + "S3eidat Enhancer.exe", pre + "THIRD-PARTY-NOTICES.md", pre + "LICENSE",
                         pre + "_internal/bin/ffmpeg.exe", pre + "_internal/bin/ffmpeg-LICENSE.txt",
                         pre + "_internal/bin/ffmpeg-README.txt"):
                if need not in names:
                    problems.append(f"zip lacks {need}")
            bundled = pre + "_internal/models/manifest.json"
            if bundled not in names:
                problems.append(f"zip lacks {bundled}")
            elif z.read(bundled).replace(b"\r\n", b"\n") != manifest_path.read_bytes().replace(b"\r\n", b"\n"):
                problems.append("the manifest in the zip differs from models/manifest.json: rebuild the zip")
    if run("gh", "auth", "status", check=False).returncode:
        problems.append("gh is not logged in")
    return problems, assets


def release(tag: str, title: str, notes: str, assets: list[Path]) -> None:
    files = [str(a) for a in assets]
    if run("gh", "release", "view", tag, "--repo", REPO, check=False).returncode == 0:
        run("gh", "release", "upload", tag, *files, "--repo", REPO, "--clobber")
    else:
        run("gh", "release", "create", tag, *files, "--repo", REPO, "--title", title, "--notes", notes)


def verify(assets: list[Path]) -> None:
    """From the outside, like a user's app: the models must download anonymously and match their hashes."""
    for p in assets:
        with urllib.request.urlopen(urllib.request.Request(MODEL_URL + p.name, method="HEAD"), timeout=60) as r:
            if int(r.headers.get("content-length", 0)) != p.stat().st_size:
                raise SystemExit(f"wrong size online: {p.name}")
    small = min((a for a in assets if a.suffix == ".onnx"), key=lambda a: a.stat().st_size)
    with urllib.request.urlopen(MODEL_URL + small.name, timeout=60) as r:
        if hashlib.sha256(r.read()).hexdigest() != sha256(small):
            raise SystemExit(f"hash mismatch online: {small.name}")
    print(f"verified {len(assets)} model files online")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="really publish")
    a = ap.parse_args()
    problems, assets = preflight()
    exists = run("gh", "repo", "view", REPO, "--json", "name", check=False).returncode == 0
    gb = sum(p.stat().st_size for p in assets) / 1e9
    print(f"repo {REPO}: {'exists, would push' if exists else 'would be created PUBLIC'}")
    print(f"release {TAG_MODELS}: {len(assets)} files, {gb:.2f} GB")
    print(f"release {TAG_APP}: {ZIP.name} ({ZIP.stat().st_size / 1e6 if ZIP.exists() else 0:.0f} MB), {FFMPEG_SRC.name}")
    if problems:
        print("\nNOT READY:\n- " + "\n- ".join(problems))
        raise SystemExit(1)
    print("\nAll checks pass.")
    if not a.go:
        print("Dry run: nothing was changed. --go publishes.")
        return
    branch = run("git", "branch", "--show-current").stdout.strip()
    if exists:
        run("git", "push", "-u", "origin", branch)
    else:
        run("gh", "repo", "create", REPO, "--public", "--description", DESCRIPTION, "--source", ".", "--remote", "origin",
            "--push")
    release(TAG_MODELS, "AI models v1", NOTES_MODELS, assets)
    release(TAG_APP, f"S3eidat Enhancer {__version__}", NOTES_APP, [ZIP, FFMPEG_SRC])
    verify(assets)
    print(f"Published: https://github.com/{REPO}/releases/tag/{TAG_APP}")


if __name__ == "__main__":
    main()
