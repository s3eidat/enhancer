# Vendored architecture code (conversion only)

Used only by `tools/convert` to load the official weights and export ONNX. Not shipped in the app.
Import lines were pointed at `_shim.py`; `NAFNetLocal` (TLC) was removed. Otherwise unchanged.

| File | Source | License |
|---|---|---|
| rrdbnet_arch.py | github.com/XPixelGroup/BasicSR | Apache-2.0 |
| srvgg_arch.py | github.com/xinntao/Real-ESRGAN | BSD-3-Clause |
| NAFNet_arch.py | github.com/megvii-research/NAFNet | MIT |
| gfpganv1_clean_arch.py, stylegan2_clean_arch.py | github.com/TencentARC/GFPGAN (clean arch, no NVIDIA CUDA ops) | Apache-2.0 |
| _shim.py | helpers after BasicSR (Apache-2.0) and NAFNet (MIT) | same |
