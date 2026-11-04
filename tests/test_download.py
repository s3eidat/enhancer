import functools
import hashlib
import http.server
import json
import threading

import pytest

from enhancer import download
from enhancer.models import Catalog


@pytest.fixture()
def served(tmp_path):
    """A folder served over HTTP holding one good model file and one whose bytes do not match the manifest."""
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    good, bad = b"onnx-bytes" * 1000, b"tampered" * 10
    (srv_dir / "good.onnx").write_bytes(good)
    (srv_dir / "bad.onnx").write_bytes(bad)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(srv_dir))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    root = tmp_path / "models"
    root.mkdir()

    def entry(name, data, claimed=None):
        return {"kind": "upscale", "scale": 4, "multiple": 1, "license": "MIT", "credit": "", "project": "",
                "source": "", "files": {"fp32": {"file": name, "bytes": len(data),
                                                  "sha256": claimed or hashlib.sha256(data).hexdigest()}}}

    manifest = {"models": {"good": entry("good.onnx", good),
                           "bad": entry("bad.onnx", bad, claimed=hashlib.sha256(b"something else").hexdigest())}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    yield Catalog(root), f"http://127.0.0.1:{httpd.server_port}/", good
    httpd.shutdown()


def test_download_checks_and_saves(served):
    cat, url, good = served
    assert download.missing(cat, ["good", "bad"]) == ["good", "bad"]
    seen = []
    download.download(cat, ["good"], lambda d, t, m: seen.append((d, t, m)), url=url)
    assert (cat.root / "good.onnx").read_bytes() == good
    assert download.missing(cat, ["good"]) == []
    assert seen[-1] == (len(good), len(good), "good")


def test_bad_checksum_is_not_saved(served):
    cat, url, _ = served
    with pytest.raises(ValueError, match="checksum"):
        download.download(cat, ["bad"], url=url)
    assert not (cat.root / "bad.onnx").exists()
    assert not list(cat.root.glob("*.part"))


def test_cancel(served):
    cat, url, _ = served
    with pytest.raises(download.Cancelled):
        download.download(cat, ["good"], cancel=lambda: True, url=url)
    assert not (cat.root / "good.onnx").exists()


def test_real_manifest_lists_every_download_id():
    cat = Catalog()
    for mid in download.ESSENTIAL + download.BEST:
        assert mid in cat.models, mid


def test_model_left_out_of_the_manifest_is_simply_not_offered(served):
    cat, _, _ = served
    assert download.missing(cat, ["gfpgan-1.4"]) == [] and download.size_of(cat, ["gfpgan-1.4"]) == 0
    assert not cat.available("gfpgan-1.4")
