# S3eidat Enhancer: design

Date: 3 Nov 2026. Status: approved by Omar ("do ur absolute best on it").

## Goal
A free, private AI photo and video enhancer that matches Topaz Photo AI / Gigapixel / Video AI feature for feature
and beats it on price (free), privacy (nothing leaves the device) and control. It ships two ways from one set of models:
a Windows desktop app (GPU) and a browser tool on S3eidat Tools (WebGPU).

Hard rule: no Topaz code, models, UI assets or branding. Only its public feature list is used as a checklist.

## Features (Topaz parity checklist)
| Topaz feature | Ours | Model |
|---|---|---|
| Upscale 2x / 4x (Gigapixel "Standard", "High fidelity") | Upscale 1x/2x/4x, any target size via resize after | Real-ESRGAN x4plus / x2plus |
| Gigapixel "Art & CG" | Anime/art mode | RealESRGAN_x4plus_anime_6B |
| Fast low-VRAM model | Fast mode (also for video + browser) | realesr-general-x4v3 (+ wdn variant) |
| Denoise | Denoise (real camera noise) | NAFNet-SIDD |
| Sharpen / motion deblur | Sharpen | NAFNet-GoPro |
| Recover faces | Face recovery | YuNet detector + GFPGAN v1.4 (clean arch) |
| Autopilot | Autopilot: noise estimate, face count, size → suggested settings | heuristics, no model |
| Strength sliders | Strength per step (blends original back) | n/a |
| Video upscale / denoise | Frame-by-frame via ffmpeg pipe, audio kept | same models |
| Frame interpolation (Chronos/Apollo) | fps boost 2x/4x | RIFE (MIT), phase 3 |
| Compare view | Before/after slider with synced zoom and 100% view | n/a |
| Batch | Queue with per-item progress, cancel, output folder | n/a |

## Licenses (checked 3 Nov 2026 against each official repo)
- Real-ESRGAN: BSD-3-Clause. SwinIR: Apache-2.0 (optional later). NAFNet: MIT. RIFE: MIT. YuNet (OpenCV Zoo): MIT.
- GFPGAN: Apache-2.0, but its repo also carries NVIDIA StyleGAN2 code (non-commercial) and DFDNet code (CC BY-NC-SA).
  We use only the clean v1.4 architecture (no NVIDIA CUDA ops). Weights are trained on FFHQ (CC BY-NC-SA dataset):
  a grey area shared by every face model. Face recovery is therefore a separate optional model; Omar decides before
  anything is published.
- CodeFormer: excluded (S-Lab non-commercial).
- Every model in `models/manifest.json` records source URL, SHA-256, size and license. The app shows the list.

## Architecture
```
tools/convert/   (dev only, needs PyTorch) official .pth -> ONNX fp32/fp16, parity checked against PyTorch
models/          ONNX files (git-ignored) + manifest.json (in git)
enhancer/        Python package, runtime needs only onnxruntime-directml, numpy, pillow, opencv
  runtime.py     ONNX Runtime sessions: DirectML GPU first, CPU fallback, cache per model
  tiling.py      overlapping tiles + feathered blending, so any size fits in VRAM with no seams
  ops.py         denoise, sharpen, upscale, faces: each one function image -> image
  faces.py       YuNet detect, align to FFHQ 512 template, GFPGAN, paste back with soft mask
  autopilot.py   noise estimate + face count -> suggested settings
  pipeline.py    runs the chosen steps in order: denoise -> sharpen -> upscale -> faces, strength blend
  jobs.py        batch queue worker, progress events, cancel
  app.py         pywebview window + JS API (same pattern as Downstream)
  web/           UI: drop zone, queue, settings, before/after slider
  cli.py         `python -m enhancer.cli in.jpg -o out.png --upscale 4 --denoise 0.7`
```
The browser port (phase 4) reuses the same ONNX files and ports `tiling.py` and the pipeline to JS with
onnxruntime-web (WebGPU, WASM fallback) as a S3eidat Tools repo. Models download on first use with SHA-256 check.

## Data flow
Image file -> RGB float32 (EXIF/ICC kept aside) -> steps in order, each tiled on the GPU -> strength blend
-> save PNG/JPG/WebP/TIFF with EXIF and ICC restored -> UI gets a preview + output path.

## Errors
GPU session failure falls back to CPU once and says so in the UI. Out of memory on a tile retries with smaller tiles.
A missing model shows which file and how to get it. One bad file in a batch fails only that item.

## Testing
- Converter: ONNX output vs PyTorch output on a random tile (max abs diff reported).
- Unit tests: tiling is exact for an identity model, blending has no seams, pipeline order, strength blend, EXIF kept.
- Real check: upscale + denoise a real photo on Omar's PC through the app and look at it in the before/after view.

## Phases
1. Desktop photo: upscale + denoise + before/after + queue (this session's "done means").
2. Sharpen, faces, Autopilot, anime/fast models.
3. Video: ffmpeg frames in/out, audio kept, RIFE fps boost.
4. Browser tool on S3eidat Tools (only with Omar's OK), then plug into the S3eidat Tools app.
