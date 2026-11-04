# S3eidat Enhancer

A free, private AI photo and video enhancer that runs on your own GPU: upscale 2×/4×, remove noise, sharpen/deblur,
recover faces, and make video smoother (2×/4×/60/120 fps) or slow motion, with a batch queue, a live spot preview
and a before/after slider. Nothing leaves the PC. There is also a browser version for S3eidat Tools (`browser/`).
MIT licensed; third-party licenses are in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Run (Windows)
Double-click `Run Enhancer.bat`, or:
```
.venv\Scripts\python -m enhancer            # desktop window
.venv\Scripts\python -m enhancer.cli photo.jpg -o out --upscale 4 --denoise 0.7 --faces 0.8
.venv\Scripts\python -m enhancer.cli clip.mp4 -o out --upscale 2 --model fast --fps 60
.venv\Scripts\python -m enhancer.cli photo.jpg --auto          # Autopilot picks the settings
.venv\Scripts\python -m enhancer --browser --port 7770         # dev: same UI in a normal browser
```
Output goes to `Pictures\Enhanced` by default (changeable in the app). Originals are never touched; output names
get `_enhanced` and never overwrite.

## Build the Windows app
```
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```
Gives `dist\S3eidat Enhancer\S3eidat Enhancer.exe` and `release\S3eidat-Enhancer-Windows.zip` (~250 MB, ffmpeg
included). The models are not in the zip: on first start the app offers to download them (389 MB essential,
+1.1 GB for best-quality noise/sharpen and faces), checks each SHA-256 against `models/manifest.json`, and keeps
them in `%LOCALAPPDATA%\S3eidat Enhancer\models`. They download from `download.MODEL_URL`
(the `models-v1` GitHub release; set `ENHANCER_MODEL_URL` to test with any folder served over HTTP).
`tools/publish_release.py` checks everything in a dry run and publishes only with `--go`.

## Browser version (S3eidat Tools)
```
.venv\Scripts\python -I tools\build_browser.py     # fills browser\lib, browser\models, browser\kit
```
Then serve `browser\` (launch config `enhancer-web`, port 7771). onnxruntime-web 1.30.0 on WebGPU, WASM fallback.
Photos only: upscale (Photo / Art / Fast), remove noise, sharpen, spot preview, compare view, batch + zip.
Models store smaller weights (16 bits, 8 for the two NAFNets) with fp32 maths (`tools/convert/web_models.py`).
`tools/stage_site_tool.py <S3eidat-Tools folder>` installs it as `repos/ai-enhancer` (122 MB, under the 130 MiB cap).

## Setup from scratch
```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt pyinstaller
.venv\Scripts\pip install -r requirements-convert.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# download the official weights into .cache\weights (URLs are in models\manifest.json -> "source")
.venv\Scripts\python -I tools\convert\convert.py
```
The app itself needs only onnxruntime-directml, numpy, pillow, opencv and pywebview (and ffmpeg for video).
PyTorch is used once, to convert the official weights to ONNX (`tools/convert`).

## Models and licenses
| Step | Model | License |
|---|---|---|
| Upscale "Standard" 2×/4× | Real-ESRGAN x2plus / x4plus | BSD-3-Clause |
| Upscale "Art & CG" | RealESRGAN_x4plus_anime_6B | BSD-3-Clause |
| Upscale "Fast" | realesr-general-x4v3 | BSD-3-Clause |
| Remove noise | NAFNet-SIDD (width 64 "Best", 32 "Fast") | MIT |
| Sharpen / deblur | NAFNet-GoPro (width 64 / 32) | MIT |
| Video frame rate / slow motion | RIFE v4.25 (Practical-RIFE) | MIT |
| Face detection | YuNet (OpenCV Zoo) | MIT |
| Face recovery | GFPGAN v1.4, clean architecture | Apache-2.0 (see note) |

Note on faces: GFPGAN's repo also contains NVIDIA StyleGAN2 code (non-commercial) and DFDNet code (CC BY-NC-SA);
we use only the clean v1.4 architecture, which has neither. The weights were trained on FFHQ, a CC BY-NC-SA
dataset, a grey area every face model shares. Decide before shipping face recovery publicly. The browser version
has no face recovery.
CodeFormer is deliberately not used (non-commercial license). No Topaz code, models, assets or branding are used.
The bundled ffmpeg is the GPL v3 "full" build from gyan.dev; the zip carries its license, and the release carries
its source.

## What is tested (3 Nov 2026, the development PC: RTX 3050 8 GB, Windows 11, Python 3.11)
- 64 automated tests pass: seamless tiling, settings, pipeline order and strength blending,
  EXIF/ICC/orientation/alpha/16-bit/non-ASCII paths, queue order + failure isolation + cancel, noise estimator,
  real-model upscale/denoise, GPU output == CPU output for the photo models, video frame-rate maths, scene-cut
  detection, real video runs (upscale + 2× fps with audio, slow motion, cancel leaves no partial file), and the
  model downloader (checksum mismatch is not saved, cancel).
- Every ONNX model matches PyTorch (max abs diff ≤ 1e-5; RIFE 5.5e-6, RIFE UHD 7.9e-5). RIFE GPU == CPU (6.6e-6).
- Photos through the UI (browser dev mode, same UI and API as the window):
  - 960×1280 WhatsApp photo, 2× + denoise 50%: 6.6 s. With face recovery (CLI): 11 s, face restored, no seam.
  - 2272×1704 Canon photos, 2×: ~19 s each. 24 MP iPhone photo (5712×4284), 2× + denoise: 203 s, 98 MP out.
- Real videos:
  - 4 s 360p phone clip, 4× Fast + 2× fps (CLI): 1440×2560 48 fps with audio, 109 s. Real scene cut detected.
  - Same clip, 2× + denoise + 2× fps through the queue UI: 122 s, sound kept.
  - 4 s 1080×1920 clip, 24 → 60 fps, H.265 (NVENC): 64 s.
  - RIFE in-between frames checked by eye (no ghosting) and by a motion test (5× closer to the true frame than
    a plain blend).
- Browser version (preview browser, WebGPU on the RTX 3050): self-test passes (tiling maths + a real model run),
  960×1280 photo 2× Photo model + denoise: 75 s including the first model download; CPU-only mode works; no
  noise + sharpen + Fast 2× with the final smaller models: 20.5 s, no console errors; no
  horizontal scroll at phone width; the S3eidat Tools sync, check and smoke scripts pass on a scratch copy.
- Packaged app (`dist\S3eidat Enhancer`): starts on the GPU, finds its bundled ffmpeg, first-run model download
  from a local test server, then a 360p video job (2× + 2× fps, sound kept, NVENC) in 72 s using its bundled
  ffmpeg; the packaged window opens and loads the UI on the GPU (log).

## Not tested yet / known limits
- The native window itself was not clicked through (screen access was declined): native file picker, folder
  picker and drag-and-drop from Explorer. The same UI and API work in the browser test, and the window loads the UI
  on the GPU (checked through the app log). The packaged exe was tested in its `--browser` mode for the same reason.
- Tested on one PC only. The Windows exe is not code-signed, so Windows may warn about an unknown publisher.
- Browser version: the Photo model is slow on WebGPU (~3–5 s per 256 px tile on the RTX 3050, 5–7× slower than DirectML).
- Video: 8-bit SDR pipeline (10-bit sources keep 10 bits only with H.265/ProRes); HDR (PQ/HLG) is not converted
  and will look washed out; variable frame rate is converted to constant; subtitles are not copied.
- Video face recovery runs per frame and can shimmer, so Autopilot keeps it at 50% or less for video.
- fp16 compute is not used: on DirectML it was slower for Real-ESRGAN and gave wrong output for NAFNet and GFPGAN.
- Very large 4× photo jobs are held in RAM as float32 (24 MP × 4× = 384 MP ≈ 4.6 GB).
- NAFNet runs per tile, so its global colour statistics are per 384–512 px tile; no seams seen, but not TLC-exact.
- The dev venv uses Microsoft Store Python, so dev logs/settings land in
  `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\Local\S3eidat Enhancer`
  (the packaged app uses the normal `%LOCALAPPDATA%\S3eidat Enhancer`).

## Layout
`enhancer/` app (runtime, tiling, ops, faces, pipeline, autopilot, video, download, jobs, server, app, web UI) ·
`browser/` S3eidat Tools version · `tools/convert/` weight conversion + web models · `models/manifest.json`
sources, hashes, licenses · `Enhancer.spec` + `build_windows.ps1` packaging · `tools/publish_release.py` releases · `docs/superpowers/` spec, plan · `tests/`.
