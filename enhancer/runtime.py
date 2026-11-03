"""ONNX Runtime sessions: GPU through DirectML when it works, CPU otherwise.

fp32 everywhere: on DirectML (RTX 3050, 3 Nov 2026) fp16 was slower for Real-ESRGAN and wrong for
NAFNet and GFPGAN (reductions overflow), so the converter no longer ships fp16 by default.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

import numpy as np
import onnxruntime as ort

from .models import Catalog, MissingModel

log = logging.getLogger(__name__)

MAX_SESSIONS = 3  # keeps VRAM bounded when a pipeline uses several models


def is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in ("out of memory", "outofmemory", "8007000e", "887a0005", "887a0006",
                                  "bad_alloc", "failed to allocate"))


class Runtime:
    def __init__(self, catalog: Catalog | None = None, device: str = "auto"):
        self.catalog = catalog or Catalog()
        self.gpu_available = "DmlExecutionProvider" in ort.get_available_providers()
        self.use_gpu = self.gpu_available and device != "cpu"
        self.notice = "" if self.use_gpu or device == "cpu" else "No DirectML GPU found, running on the CPU."
        self._sessions: OrderedDict[tuple[str, bool], ort.InferenceSession] = OrderedDict()
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()  # one model call at a time: the queue and live previews share the GPU

    @property
    def device_name(self) -> str:
        return "GPU (DirectML)" if self.use_gpu else "CPU"

    def _create(self, mid: str, gpu: bool) -> ort.InferenceSession:
        info = self.catalog.get(mid)
        path = info.path("fp32", self.catalog.root)
        if not path.exists():
            raise MissingModel(f"Model file missing: {path}. Run tools/convert/convert.py or download the models.")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if gpu:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts = {"performance_preference": "high_performance", "device_filter": "gpu"}
            if os.environ.get("ENHANCER_GPU_ID"):
                opts = {"device_id": int(os.environ["ENHANCER_GPU_ID"])}
            providers = [("DmlExecutionProvider", opts)]
        else:
            so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
            providers = ["CPUExecutionProvider"]
        log.info("loading %s on %s", path.name, "GPU" if gpu else "CPU")
        return ort.InferenceSession(str(path), sess_options=so, providers=providers)

    def session(self, mid: str) -> ort.InferenceSession:
        with self._lock:
            key = (mid, self.use_gpu)
            if key in self._sessions:
                self._sessions.move_to_end(key)
                return self._sessions[key]
            try:
                sess = self._create(mid, self.use_gpu)
            except MissingModel:
                raise
            except Exception as e:
                if not self.use_gpu:
                    raise
                log.warning("GPU session for %s failed (%s); falling back to CPU", mid, e)
                self.use_gpu = False
                self.notice = "The GPU could not run the models, so this ran on the CPU (slower)."
                self._sessions.clear()
                key = (mid, False)
                sess = self._create(mid, False)
            self._sessions[key] = sess
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
            return sess

    def run(self, mid: str, x: np.ndarray) -> np.ndarray:
        """x: (N,3,h,w) float32 in [0,1] -> float32 output."""
        sess = self.session(mid)
        with self._run_lock:
            out = sess.run(None, {"input": np.ascontiguousarray(x, dtype=np.float32)})[0]
        return out.astype(np.float32, copy=False)

    def fn(self, mid: str):
        return lambda x: self.run(mid, x)

    def release(self) -> None:
        with self._lock:
            self._sessions.clear()
