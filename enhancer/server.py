"""A tiny localhost server for the UI and for the images shown in the compare view.

Bound to 127.0.0.1 on a random port; every URL starts with a random token, and only files the app
registered can be read.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from . import paths

log = logging.getLogger(__name__)


class LocalServer:
    def __init__(self, web_root: Path | None = None, port: int = 0):
        self.web_root = (web_root or paths.web_dir()).resolve()
        self.token = secrets.token_urlsafe(16)
        self._files: dict[str, Path] = {}
        self._lock = threading.Lock()
        self.api = None  # set only in --browser dev mode: exposes the JS API over POST
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                server._handle(self)

            def do_POST(self):
                server._handle_api(self)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self.token}"

    def ui_url(self) -> str:
        return f"{self.base}/ui/index.html"

    def file_url(self, path: Path) -> str:
        key = secrets.token_urlsafe(9)
        with self._lock:
            self._files[key] = Path(path)
        return f"{self.base}/f/{key}/{quote(Path(path).name)}"

    def _send(self, h: BaseHTTPRequestHandler, path: Path | None) -> None:
        if path is None or not path.is_file():
            h.send_error(404)
            return
        data = path.read_bytes()
        h.send_response(200)
        h.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        h.send_header("Content-Length", str(len(data)))
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        h.wfile.write(data)

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        parts = unquote(urlparse(h.path).path).strip("/").split("/")
        if parts == [""] and self.api is not None:  # dev mode convenience: / opens the UI
            h.send_response(302)
            h.send_header("Location", self.ui_url() + "?browser=1")
            h.end_headers()
            return
        if len(parts) < 3 or not secrets.compare_digest(parts[0], self.token):
            h.send_error(404)
            return
        if parts[1] == "ui":
            target = (self.web_root / "/".join(parts[2:])).resolve()
            self._send(h, target if target.is_relative_to(self.web_root) else None)
        elif parts[1] == "f":
            with self._lock:
                target = self._files.get(parts[2])
            self._send(h, target)
        else:
            h.send_error(404)

    def _handle_api(self, h: BaseHTTPRequestHandler) -> None:
        parts = urlparse(h.path).path.strip("/").split("/")
        if self.api is None or len(parts) != 3 or not secrets.compare_digest(parts[0], self.token)                 or parts[1] != "api" or parts[2].startswith("_") or not hasattr(self.api, parts[2]):
            h.send_error(404)
            return
        args = json.loads(h.rfile.read(int(h.headers.get("Content-Length") or 0)) or b"[]")
        try:
            result = getattr(self.api, parts[2])(*args)
        except Exception as e:
            log.exception("api %s failed", parts[2])
            result = {"error": str(e)}
        body = json.dumps(result, default=str).encode()
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def close(self) -> None:
        self.httpd.shutdown()
