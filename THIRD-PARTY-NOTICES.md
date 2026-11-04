# Third-party notices

S3eidat Enhancer's own code is MIT licensed (see LICENSE). It uses these separate works, under their own licenses.

## FFmpeg (bundled with the Windows app, as separate programs)

`ffmpeg.exe` and `ffprobe.exe` are the unmodified GPL v3 "full" build from www.gyan.dev, version
2025-10-09-git-469aad3897. The app runs them as separate programs. Their license and build notes are next to them
in the app folder, in `_internal/bin/ffmpeg-LICENSE.txt` and `_internal/bin/ffmpeg-README.txt`, which also list
the external libraries in this build and their project pages.

Source code: `ffmpeg-source-469aad3897.tar.gz`, attached to the same GitHub release as the app
(github.com/s3eidat/enhancer/releases), is FFmpeg at the exact commit of this build.

## AI models (downloaded by the app on first use, not part of the app's code)

| Model | Use | License | Project |
|---|---|---|---|
| Real-ESRGAN x4plus, x2plus, x4plus anime 6B, general x4v3 | upscale | BSD-3-Clause | github.com/xinntao/Real-ESRGAN |
| NAFNet SIDD and GoPro (width 32, 64) | remove noise, sharpen | MIT | github.com/megvii-research/NAFNet |
| RIFE v4.25 | video frame rate, slow motion | MIT | github.com/hzwer/Practical-RIFE |
| YuNet | find faces | MIT | github.com/opencv/opencv_zoo |
| GFPGAN v1.4, clean architecture (optional download) | recover faces | Apache-2.0 | github.com/TencentARC/GFPGAN |

GFPGAN's weights were trained on FFHQ, a dataset under CC BY-NC-SA 4.0. CodeFormer is not used. No Topaz code,
models, assets or branding are used. Each model's source, hash and license are in `models/manifest.json`.

## Python libraries in the Windows app

| Library | License |
|---|---|
| Python | PSF License |
| ONNX Runtime with DirectML (onnxruntime-directml) | MIT, plus Microsoft's DirectML redistribution terms |
| NumPy | BSD-3-Clause |
| OpenCV (opencv-python-headless) | Apache-2.0 (the package also bundles LGPL FFmpeg libraries) |
| Pillow | MIT-CMU |
| pywebview | BSD-3-Clause |
| pythonnet, clr_loader | MIT |

## Browser version

onnxruntime-web 1.30.0 (MIT, github.com/microsoft/onnxruntime), plus the Real-ESRGAN and NAFNet models above.
