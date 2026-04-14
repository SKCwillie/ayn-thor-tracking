#!/usr/bin/env python3
"""Start local backend + frontend for UI testing.

- Backend: uvicorn api:app on BACKEND_PORT
- Frontend: lightweight static server on FRONTEND_PORT
  - Serves `ui/index.html` at /
  - Serves files from `ui/` and `assets/`
  - Proxies unknown routes (API calls) to backend
"""

from __future__ import annotations

import argparse
import mimetypes
import shutil
import signal
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
UI_DIR = PROJECT_ROOT / "ui"
ASSETS_DIR = PROJECT_ROOT / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local backend + frontend dev servers")
    parser.add_argument("--host", default="127.0.0.1", help="Host for both servers")
    parser.add_argument("--backend-port", type=int, default=8001, help="Backend port")
    parser.add_argument("--frontend-port", type=int, default=8000, help="Frontend port")
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn --reload")
    parser.add_argument("--startup-timeout", type=int, default=20, help="Seconds to wait for backend health")
    return parser.parse_args()


def wait_for_backend(backend_url: str, timeout_seconds: int) -> None:
    health_url = f"{backend_url}/health"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(f"Backend did not become ready in {timeout_seconds}s: {health_url}")


class FrontendProxyHandler(SimpleHTTPRequestHandler):
    ui_dir: Path = UI_DIR
    assets_dir: Path = ASSETS_DIR
    backend_url: str = "http://127.0.0.1:8001"

    # Keep logs compact but useful.
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stdout.write(f"[frontend] {self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        if self._serve_index_or_static(head_only=False):
            return
        self._proxy_request("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        if self._serve_index_or_static(head_only=True):
            return
        self._proxy_request("HEAD")

    def _serve_index_or_static(self, head_only: bool) -> bool:
        path = unquote(urlsplit(self.path).path)

        if path == "/":
            return self._send_file(self.ui_dir / "index.html", head_only=head_only)

        rel = path.lstrip("/")
        if not rel or rel.endswith("/"):
            return False

        for base in (self.ui_dir, self.assets_dir):
            candidate = (base / rel).resolve()
            if self._is_within(base.resolve(), candidate) and candidate.is_file():
                return self._send_file(candidate, head_only=head_only)

        return False

    @staticmethod
    def _is_within(base: Path, candidate: Path) -> bool:
        return candidate == base or base in candidate.parents

    def _send_file(self, file_path: Path, head_only: bool) -> bool:
        if not file_path.exists() or not file_path.is_file():
            return False

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        file_size = file_path.stat().st_size

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        if not head_only:
            with file_path.open("rb") as f:
                self._stream_to_client(f)
        return True

    def _proxy_request(self, method: str) -> None:
        target_url = f"{self.backend_url}{self.path}"
        req = Request(target_url, method=method)

        for header, value in self.headers.items():
            if header.lower() in {"host", "connection", "content-length"}:
                continue
            req.add_header(header, value)

        try:
            with urlopen(req, timeout=30) as upstream:
                self.send_response(upstream.status)
                self._copy_upstream_headers(upstream.headers.items())
                self.end_headers()
                if method != "HEAD":
                    self._stream_to_client(upstream)
        except HTTPError as http_error:
            self.send_response(http_error.code)
            self._copy_upstream_headers(http_error.headers.items())
            self.end_headers()
            if method != "HEAD" and http_error.fp is not None:
                self._stream_to_client(http_error.fp)
        except URLError as url_error:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Proxy error: {url_error.reason}".encode("utf-8"))

    def _copy_upstream_headers(self, headers: Iterable[tuple[str, str]]) -> None:
        for name, value in headers:
            if name.lower() in {"transfer-encoding", "connection", "content-encoding"}:
                continue
            self.send_header(name, value)

    def _stream_to_client(self, source: BinaryIO) -> None:
        while True:
            chunk = source.read(64 * 1024)
            if not chunk:
                break
            self.wfile.write(chunk)


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    args = parse_args()

    if not UI_DIR.exists() or not (UI_DIR / "index.html").exists():
        raise FileNotFoundError(f"Missing UI file: {UI_DIR / 'index.html'}")

    backend_url = f"http://{args.host}:{args.backend_port}"

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
    ]
    if not args.no_reload:
        backend_cmd.append("--reload")

    print(f"Starting backend: {' '.join(backend_cmd)}")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(PROJECT_ROOT))

    try:
        wait_for_backend(backend_url, timeout_seconds=args.startup_timeout)
    except Exception:
        terminate_process(backend_proc)
        raise

    FrontendProxyHandler.backend_url = backend_url
    server = ThreadingHTTPServer((args.host, args.frontend_port), FrontendProxyHandler)

    def shutdown_handler(_signum: int, _frame: object) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"Frontend URL: http://{args.host}:{args.frontend_port}")
    print(f"Backend URL:  {backend_url}")
    print("Press Ctrl+C to stop both servers.")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        terminate_process(backend_proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

