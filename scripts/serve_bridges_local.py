#!/usr/bin/env python3
"""Serve a directory over HTTP with CORS + HTTP Range support.

Two uses:

1. **Local preview of the published tree.** Point it at a staging tree built by
   ``matcher factory publish --target-dir <dir>`` so the browser data browser can
   query it exactly as it will query R2 (CORS + range reads):

       python scripts/serve_bridges_local.py data/publish_staging_local --port 8000

   then open the site with ``?base=http://localhost:8000``.

2. **Local preview of the site itself** (the static Pages ``site/`` dir):

       python scripts/serve_bridges_local.py site --port 8001

   then open ``http://localhost:8001/index.html?base=http://localhost:8000``.

Python's stock ``http.server`` supports neither CORS nor Range requests; DuckDB-WASM
needs both to range-read Parquet cross-origin, so this small handler adds them. It
is a *local* convenience only — production hosting is Cloudflare R2 (see
``docs/PUBLISHING.md``).
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class CorsRangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + permissive CORS + single-range GET support."""

    def end_headers(self) -> None:  # noqa: D401 - stdlib hook
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges"
        )
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook name
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().do_GET()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()

        size = os.path.getsize(path)
        try:
            start_s, end_s = rng[len("bytes=") :].split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            return super().do_GET()
        start = max(0, start)
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        ctype = self.guess_type(path)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", help="directory to serve")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    handler = partial(CorsRangeHandler, directory=os.path.abspath(args.directory))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"serving {os.path.abspath(args.directory)} at http://{args.host}:{args.port} (CORS + Range)"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
