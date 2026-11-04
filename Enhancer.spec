# PyInstaller spec for S3eidat Enhancer (one-folder build). Run build_windows.ps1 -> dist\S3eidat Enhancer\
# The AI models are not bundled: the app downloads them on first run (checked by SHA-256 against the manifest).
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(SPECPATH)
sys.path.insert(0, str(ROOT))
from enhancer import APP_NAME, __version__  # noqa: E402

bindir = ROOT / "bin" / "win"
helpers = ["ffmpeg.exe", "ffprobe.exe", "ffmpeg-LICENSE.txt", "ffmpeg-README.txt"]
missing = [h for h in helpers if not (bindir / h).exists()]
if missing:
    raise SystemExit(f"Missing {missing} in {bindir} (run build_windows.ps1)")

datas = [
    (str(ROOT / "enhancer" / "web"), "enhancer/web"),
    (str(ROOT / "models" / "manifest.json"), "models"),
    (str(ROOT / "assets" / "icon.png"), "assets"),
] + [(str(bindir / h), "bin") for h in helpers]  # plain files: no dependency scan of 200 MB ffmpeg builds

a = Analysis(
    [str(ROOT / "enhancer" / "__main__.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=["enhancer.app"],
    excludes=["tkinter", "unittest", "pydoc", "test", "pytest", "torch", "torchvision", "onnx", "onnxconverter_common",
              "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy", "gi", "cefpython3", "matplotlib", "IPython",
              "scipy", "pandas", "Pythonwin", "win32com", "pythoncom", "win32ui",
              "pkg_resources", "setuptools"],  # nothing at runtime needs them; their PyInstaller hook breaks the start
    noarchive=False,
)
pyz = PYZ(a.pure)

from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,  # noqa: E402
                                                 VarFileInfo, VarStruct, VSVersionInfo)
nums = tuple(int(x) for x in (__version__.split(".") + ["0", "0", "0"])[:4])
version_file = VSVersionInfo(
    ffi=FixedFileInfo(filevers=nums, prodvers=nums),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", "S3eidat"),
            StringStruct("FileDescription", "S3eidat Enhancer - free AI photo and video enhancer"),
            StringStruct("FileVersion", __version__),
            StringStruct("InternalName", "S3eidat Enhancer"),
            StringStruct("OriginalFilename", "S3eidat Enhancer.exe"),
            StringStruct("ProductName", APP_NAME),
            StringStruct("ProductVersion", __version__),
        ])]),
        VarFileInfo([VarStruct("Translation", [1033, 1200])]),
    ],
)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, debug=False, strip=False, upx=False,
          console=False, icon=str(ROOT / "assets" / "icon.ico"), version=version_file)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
