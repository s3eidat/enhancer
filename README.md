# S3eidat Enhancer

A free, private AI photo enhancer that runs on your own GPU: upscale 2×/4×, remove noise, sharpen/deblur and
recover faces, with a batch queue, a live spot preview and a before/after slider. Nothing leaves the PC.
Local repo only: not published anywhere.

## Run (Windows)
Double-click `Run Enhancer.bat`, or:
```
.venv\Scripts\python -m enhancer            # desktop window
.venv\Scripts\python -m enhancer.cli photo.jpg -o out --upscale 4 --denoise 0.7 --faces 0.8
.venv\Scripts\python -m enhancer.cli photo.jpg --auto          # Autopilot picks the settings
.venv\Scripts\python -m enhancer --browser --port 7770         # dev: same UI in a normal browser
```
Output goes to `Pictures\Enhanced` by default (changeable in the app). Originals are never touched; output names
get `_enhanced` and never overwrite.

## Setup from scratch
```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-convert.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# download the official weights into .cache\weights (URLs are in models\manifest.json -> "source")
.venv\Scripts\python -I tools\convert\convert.py
```
The app itself needs only onnxruntime-directml, numpy, pillow, opencv and pywebview. PyTorch is used once, to
convert the official weights to ONNX (`tools/convert`).

## Models and licenses
| Step | Model | License |
|---|---|---|
| Upscale "Standard" 2×/4× | Real-ESRGAN x2plus / x4plus | BSD-3-Clause |
| Upscale "Art & CG" | RealESRGAN_x4plus_anime_6B | BSD-3-Clause |
| Upscale "Fast" | realesr-general-x4v3 | BSD-3-Clause |
| Remove noise | NAFNet-SIDD (width 64 "Best", 32 "Fast") | MIT |
| Sharpen / deblur | NAFNet-GoPro (width 64 / 32) | MIT |
| Face detection | YuNet (OpenCV Zoo) | MIT |
| Face recovery | GFPGAN v1.4, clean architecture | Apache-2.0 (see note) |

Note on faces: GFPGAN's repo also contains NVIDIA StyleGAN2 code (non-commercial) and DFDNet code (CC BY-NC-SA);
we use only the clean v1.4 architecture, which has neither. The weights were trained on FFHQ, a CC BY-NC-SA
dataset, a grey area every face model shares. Decide before shipping face recovery publicly.
CodeFormer is deliberately not used (non-commercial license). No Topaz code, models, assets or branding are used.

## What is tested (3 Nov 2026, Omar's PC: RTX 3050 8 GB, Windows 11, Python 3.11)
- 45 automated tests pass: seamless tiling (exact for identity/upscale models), settings, pipeline order and
  strength blending, EXIF/ICC/orientation/alpha/16-bit/non-ASCII paths, queue order + one failure does not stop
  the batch + cancel, noise estimator, real-model upscale/denoise, and GPU output == CPU output for all 9 models.
- Every ONNX model matches PyTorch (max abs diff ≤ 1e-5).
- Real photos from this PC through the UI (browser dev mode, same UI and API as the window):
  - 960×1280 WhatsApp photo, 2× + denoise 50%: 6.6 s. With face recovery (CLI): 11 s, face clearly restored, no seam.
  - 2272×1704 Canon PowerShot photos, 2×: ~19 s each.
  - Live spot preview: ~5 s first time (loads models), Autopilot, compare slider, zoom/pan, divider drag.
- GPU is about 12× faster than CPU for Real-ESRGAN x4 (0.73 s vs 8.75 s per 256px tile).
- The desktop window starts and loads the UI on the GPU (checked through the app log).

## Not tested yet / known limits
- The native window itself was not clicked through (screen access was declined): native file picker, folder
  picker and drag-and-drop from Explorer (they use pywebview's APIs; the same UI works in the browser test).
- fp16 is not used: on DirectML it was slower for Real-ESRGAN and gave wrong output for NAFNet and GFPGAN.
- Very large 4× jobs: the result is held in RAM as float32 (e.g. 24 MP × 4× = 384 MP ≈ 4.6 GB). Fine for
  typical photos on 16 GB; huge ones may run out of memory (the job fails with a message, the app keeps going).
- NAFNet runs per tile, so its global colour statistics are per 384–512px tile; no seams seen, but not TLC-exact.
- The dev venv uses Microsoft Store Python, so logs/settings land in
  `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\Local\S3eidat Enhancer`.
- No installer yet; no video yet; no browser version yet (see the spec's phases).

## Layout
`enhancer/` app (runtime, tiling, ops, faces, pipeline, autopilot, jobs, server, app, web UI) ·
`tools/convert/` weight conversion · `models/manifest.json` sources, hashes, licenses ·
`docs/superpowers/` spec and plan · `tests/`.
