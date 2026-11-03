"""The model manifest: which ONNX files exist, what they do and where they came from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import paths


class MissingModel(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    kind: str          # upscale | denoise | sharpen | face | detect | interp
    scale: int
    multiple: int      # input H and W must be divisible by this
    fixed: int | None  # fixed square input size (faces), else dynamic
    files: dict        # precision -> {"file", "bytes", "sha256", ...}
    license: str
    credit: str
    project: str
    source: str
    note: str = ""

    def path(self, precision: str, root: Path | None = None) -> Path:
        entry = self.files.get(precision) or self.files["fp32"]
        return (root or paths.models_dir()) / entry["file"]


class Catalog:
    def __init__(self, root: Path | None = None):
        self.root = root or paths.models_dir()
        mf = self.root / "manifest.json"
        data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"models": {}}
        self.models = {
            mid: ModelInfo(id=mid, kind=m["kind"], scale=m["scale"], multiple=m["multiple"], fixed=m.get("fixed"),
                           files=m["files"], license=m["license"], credit=m["credit"], project=m["project"],
                           source=m["source"], note=m.get("note", ""))
            for mid, m in data["models"].items()
        }

    def get(self, mid: str) -> ModelInfo:
        if mid not in self.models:
            raise MissingModel(f"Model '{mid}' is not in {self.root / 'manifest.json'}")
        return self.models[mid]

    def available(self, mid: str) -> bool:
        return mid in self.models and self.models[mid].path("fp32", self.root).exists()

    def credits(self) -> list[dict]:
        return [{"id": m.id, "credit": m.credit, "license": m.license, "project": m.project, "note": m.note}
                for m in self.models.values()]
