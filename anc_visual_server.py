#!/usr/bin/env python3
"""
Open the ANC visual dashboard in a local browser.

Run:
    python anc_visual_server.py

This server only serves files from the project directory. It does not provide
real audio input or output; it is a visual companion to anc_research_demo.py.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import time
import webbrowser
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
HTML_FILE = PROJECT_DIR / "anc_visual_dashboard.html"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


def open_browser(url: str) -> None:
    time.sleep(0.4)
    webbrowser.open(url)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the ANC visual dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    if not HTML_FILE.exists():
        raise SystemExit(f"Missing {HTML_FILE.name}; run this script from the project with the HTML file present.")

    handler = functools.partial(QuietHandler, directory=str(PROJECT_DIR))
    with socketserver.TCPServer((args.host, args.port), handler) as server:
        url = f"http://{args.host}:{args.port}/{HTML_FILE.name}"
        print(f"Serving ANC visual dashboard at {url}")
        print("Press Ctrl+C to stop.")
        if not args.no_browser:
            threading.Thread(target=open_browser, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
