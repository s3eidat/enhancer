# AI Photo Enhancer

Upscale photos 2× or 4×, remove noise and sharpen blur with AI, in your browser. The models run on your own GPU
(WebGPU) or, without one, on the CPU. Nothing is uploaded: the page has no server and makes no requests except
for its own files.

## What it does

- **Upscale 2× / 4×** with three Real-ESRGAN models: Photo, Art & anime, Fast. "AI detail" mixes in a smooth resize
  for a less processed look.
- **Remove noise** (NAFNet, trained on real camera noise) and **sharpen / deblur** (NAFNet, motion blur), each
  with a strength slider.
- **Spot preview**: click a photo to see one spot enhanced in a second or two before running the whole thing.
- **Before / after view** with a draggable line, zoom, pan, and hold B or A to see one side.
- **Batch**: drop many photos, download each or all as a .zip. PNG, JPG or WebP. Transparency is kept for PNG/WebP.

## Honest limits

- The output is capped at 64 MP and 16384 px per side (a browser tab cannot hold more), so very large photos need 2× or Off.
- Without WebGPU (older browsers, some phones) it runs on one CPU core and is slow for big photos.
- Each model downloads from this page the first time you use it: Fast 2.5 MB, Photo 34 MB, Art 9 MB,
  Noise 31 MB, Sharpen 19 MB. They are kept in your browser for next time.

## Models and licenses

| Model | Use | License |
|---|---|---|
| Real-ESRGAN x4plus, x4plus anime 6B, general x4v3 | upscale | BSD-3-Clause, github.com/xinntao/Real-ESRGAN |
| NAFNet SIDD width 32 | remove noise | MIT, github.com/megvii-research/NAFNet |
| NAFNet GoPro width 32 | sharpen | MIT, github.com/megvii-research/NAFNet |
| onnxruntime-web 1.30.0 | runs the models | MIT, github.com/microsoft/onnxruntime |

The model files are the official weights converted to ONNX, with weights stored in 16 bits (8 bits for the two
NAFNets) and the maths in 32-bit, checked against the full model.
