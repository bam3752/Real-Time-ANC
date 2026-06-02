from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from urllib.parse import parse_qs, urlparse

import sounddevice as sd

from anc_audio_core import AudioSettings, RealtimeANCProcessor


ROOT = Path(__file__).resolve().parent


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class AudioScope:
    def __init__(self) -> None:
        self.settings = AudioSettings(controller="inversion", output_enabled=True, gain=0.35, lowpass_enabled=False)
        self.processor = RealtimeANCProcessor(self.settings)
        self.stream: sd.InputStream | None = None
        self.running = False
        self.status = "off"

    def start(
        self,
        input_device: int | None = None,
        sample_rate: int = 48_000,
        block_size: int = 256,
        gain: float = 0.35,
    ) -> None:
        self.stop()
        self.settings = AudioSettings(
            controller="inversion",
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            channels=1,
            output_enabled=True,
            muted=False,
            gain=clamp(gain),
            delay_samples=0,
            lowpass_enabled=False,
        )
        self.processor.configure(self.settings)
        self.processor.reset()
        self.stream = sd.InputStream(
            device=input_device,
            samplerate=self.settings.sample_rate,
            blocksize=self.settings.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )
        self.stream.start()
        self.running = True
        self.status = "on"

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.running = False
        self.status = "off"

    def configure(self, gain: float | None = None) -> None:
        if gain is not None:
            self.settings.gain = clamp(gain)
        self.processor.configure(self.settings)

    def frame(self) -> dict[str, object]:
        ref = self.processor.ref_ring.snapshot()[-2048:]
        anti = self.processor.anti_ring.snapshot()[-2048:]
        metrics = self.processor.metrics.as_dict()
        return {
            "running": self.running,
            "status": self.status,
            "sample_rate": self.settings.sample_rate,
            "input": ref.astype(float).round(6).tolist(),
            "anti": anti.astype(float).round(6).tolist(),
            "metrics": metrics,
        }

    def _callback(self, indata, frames, _time_info, status) -> None:
        if status:
            self.status = str(status)
        try:
            self.processor.process(indata.copy(), frames)
        except Exception as exc:
            self.status = f"audio error: {exc}"


SCOPE = AudioScope()


class ReusableThreadingTCPServer(ThreadingTCPServer):
    allow_reuse_address = True


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.redirect("/anc_visual_dashboard.html")
            return
        if parsed.path == "/api/start":
            params = parse_qs(parsed.query)
            input_device = self.optional_int(params.get("input_device", [None])[0])
            sample_rate = self.optional_int(params.get("sample_rate", ["48000"])[0]) or 48_000
            block_size = self.optional_int(params.get("block_size", ["256"])[0]) or 256
            gain = self.optional_float(params.get("gain", ["0.35"])[0]) or 0.35
            try:
                SCOPE.start(input_device=input_device, sample_rate=sample_rate, block_size=block_size, gain=gain)
                self.json_response({"ok": True, "running": True, "status": SCOPE.status})
            except Exception as exc:
                SCOPE.status = str(exc)
                self.json_response({"ok": False, "running": False, "status": str(exc)}, status=500)
            return
        if parsed.path == "/api/stop":
            SCOPE.stop()
            self.json_response({"ok": True, "running": False, "status": SCOPE.status})
            return
        if parsed.path == "/api/config":
            params = parse_qs(parsed.query)
            gain = self.optional_float(params.get("gain", [None])[0])
            SCOPE.configure(gain=gain)
            self.json_response({"ok": True, "settings": {"gain": SCOPE.settings.gain}})
            return
        if parsed.path == "/api/frame":
            self.json_response(SCOPE.frame())
            return
        super().do_GET()

    def redirect(self, target: str) -> None:
        self.send_response(302)
        self.send_header("Location", target)
        self.end_headers()

    def json_response(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def optional_int(value: object) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def optional_float(value: object) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    for port in range(args.port, args.port + 20):
        try:
            with ReusableThreadingTCPServer((args.host, port), Handler) as server:
                print(f"http://{args.host}:{port}/anc_visual_dashboard.html")
                server.serve_forever()
        except OSError:
            continue
    raise SystemExit(f"No open port found from {args.port} to {args.port + 19}")


if __name__ == "__main__":
    raise SystemExit(main())
