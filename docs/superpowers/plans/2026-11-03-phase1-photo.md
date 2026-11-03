# Phase 1–2 (desktop photo) Implementation Plan

> Executed inline in the session that wrote it (Omar asked for minimal round trips). Steps use checkboxes.

**Goal:** A Windows desktop app that upscales (2x/4x), denoises, sharpens and recovers faces on real photos
on the GPU, with a batch queue, progress and a before/after slider.

**Architecture:** `tools/convert` turns official weights into ONNX once. The `enhancer` package runs them with
ONNX Runtime (DirectML, CPU fallback) through a tiling engine; a pywebview window talks to it over a JS API and
loads images from a token-guarded localhost server.

**Tech Stack:** Python 3.11, onnxruntime-directml 1.24, numpy, pillow, opencv-python-headless, pywebview, pytest.

## Global Constraints
- Every ONNX model: RGB float [0,1] NCHW in and out; input name `input`, output `output`.
- No torch at runtime. No Topaz code, models, assets or branding.
- Commits as s3eidat <161432703+s3eidat@users.noreply.github.com>. Local git only; nothing published.
- Model files stay out of git; `models/manifest.json` (source, sha256, license) is in git.

---

### Task 1: Converter (done when every model exports and matches PyTorch)
Files: `tools/convert/convert.py`, `tools/convert/vendor/*`
- [x] Vendor official arch files, shim BasicSR imports, ONNX-friendly LayerNorm2d
- [x] Export fp32 + fp16, parity check, dynamic shape check, write manifest

### Task 2: Runtime + tiling
Files: `enhancer/models.py`, `enhancer/runtime.py`, `enhancer/tiling.py`, `tests/test_tiling.py`
Produces: `Runtime.run(model_id, x: np.ndarray[1,3,h,w]) -> np.ndarray`, `run_tiled(fn, img HxWx3, scale, tile,
overlap, multiple, progress=None, cancel=None) -> img`.
- [ ] Test: identity fn and nearest-upscale fn give exact output for odd sizes, small images, big images
- [ ] Test: every tile passed to fn has the same shape (DirectML compiles per shape)
- [ ] Implement, run, commit

### Task 3: Ops + pipeline + image IO
Files: `enhancer/imageio.py`, `enhancer/faces.py`, `enhancer/ops.py`, `enhancer/pipeline.py`, `enhancer/autopilot.py`,
`tests/test_pipeline.py`, `tests/test_imageio.py`
Produces: `Settings` dataclass, `process(img, settings, rt, progress, cancel) -> img`, `load(path) -> Loaded`,
`save(loaded_meta, img, path, fmt)`, `suggest(img) -> dict`.
- [ ] Test: strength 0 returns the input, order is denoise → sharpen → upscale → faces (fake runtime)
- [ ] Test: EXIF and ICC survive load/save; orientation is applied once; alpha is kept
- [ ] Test: autopilot sees noise on a noisy image and none on a clean one
- [ ] Implement, run, commit

### Task 4: Jobs + CLI
Files: `enhancer/jobs.py`, `enhancer/cli.py`, `tests/test_jobs.py`
- [ ] Test: queue runs items in order, one failing item does not stop the rest, cancel stops a running item
- [ ] CLI on a real photo with the real models (GPU) → output file exists, right size

### Task 5: Desktop UI
Files: `enhancer/app.py`, `enhancer/server.py`, `enhancer/web/*`
- [ ] Drop/pick files, settings panel, queue with progress, before/after slider with synced zoom/pan
- [ ] Launch on Omar's PC, enhance a real photo, screenshot the before/after view

### Task 6: Verify + commit
- [ ] Full test run, real-photo run timings (GPU vs CPU), README with honest tested/untested list
