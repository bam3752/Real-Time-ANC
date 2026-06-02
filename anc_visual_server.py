from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from anc_audio_core import AudioSettings, RealtimeANCProcessor, SynchronizedVisualFrame, sd


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_BLOCK_SIZE = 0
DEFAULT_DISPLAY_SAMPLES = 4096
CLOUD_FRAME = SynchronizedVisualFrame(DEFAULT_DISPLAY_SAMPLES, DEFAULT_SAMPLE_RATE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class AudioScope:
    def __init__(self) -> None:
        self.settings = AudioSettings(controller="inversion", output_enabled=True, gain=0.35, lowpass_enabled=False)
        self.processor = RealtimeANCProcessor(self.settings)
        self.stream = None
        self.running = False
        self.status = "off"
        self.display_samples = DEFAULT_DISPLAY_SAMPLES

    def start(
        self,
        input_device: int | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        gain: float = 0.35,
        display_samples: int = DEFAULT_DISPLAY_SAMPLES,
    ) -> None:
        if sd is None:
            raise RuntimeError("Local sounddevice capture is unavailable on this server; use browser capture instead.")
        self.stop()
        self.display_samples = int(display_samples)
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
        self.processor.visual_frame.configure(display_samples=self.display_samples, sample_rate=self.settings.sample_rate)
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
        frame = self.processor.visual_snapshot()
        return {
            "running": self.running,
            "status": self.status,
            "sequence": frame["sequence"],
            "sample_rate": frame["sample_rate"],
            "gain": frame["gain"],
            "input": frame["input"].astype(float).round(6).tolist(),
            "inverse": frame["inverse"].astype(float).round(6).tolist(),
            "input_rms": frame["input_rms"],
            "input_dbfs": frame["input_dbfs"],
            "inverse_rms": frame["inverse_rms"],
            "inverse_dbfs": frame["inverse_dbfs"],
            "inverse_correlation": frame["inverse_correlation"],
        }

    def _callback(self, indata, frames, _time_info, status) -> None:
        if status:
            self.status = str(status)
        try:
            ref = indata[:, 0].copy()
            self.processor.process(indata.copy(), frames)
            self.processor.update_visual_frame(ref, self.settings.gain)
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
        if parsed.path == "/api/capabilities":
            self.json_response({"browser_audio": True, "local_audio": sd is not None})
            return
        if parsed.path == "/api/start":
            params = parse_qs(parsed.query)
            input_device = self.optional_int(params.get("input_device", [None])[0])
            sample_rate = self.optional_int(params.get("sample_rate", [str(DEFAULT_SAMPLE_RATE)])[0]) or DEFAULT_SAMPLE_RATE
            block_size = self.optional_int(params.get("block_size", [str(DEFAULT_BLOCK_SIZE)])[0])
            if block_size is None:
                block_size = DEFAULT_BLOCK_SIZE
            display_samples = self.optional_int(params.get("display_samples", [str(DEFAULT_DISPLAY_SAMPLES)])[0]) or DEFAULT_DISPLAY_SAMPLES
            gain = self.optional_float(params.get("gain", ["0.35"])[0]) or 0.35
            try:
                SCOPE.start(
                    input_device=input_device,
                    sample_rate=sample_rate,
                    block_size=block_size,
                    gain=gain,
                    display_samples=display_samples,
                )
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

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/invert":
            self.send_error(404)
            return
        try:
            payload = self.read_json()
            samples = np.asarray(payload.get("samples", []), dtype=np.float32)
            gain = clamp(float(payload.get("gain", 0.35)))
            sample_rate = int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE))
            display_samples = int(payload.get("display_samples", DEFAULT_DISPLAY_SAMPLES))
            CLOUD_FRAME.configure(display_samples=display_samples, sample_rate=sample_rate)
            frame = CLOUD_FRAME.update(samples, gain)
            self.json_response(
                {
                    "ok": True,
                    "running": True,
                    "status": "browser",
                    "sequence": frame["sequence"],
                    "sample_rate": frame["sample_rate"],
                    "gain": frame["gain"],
                    "input": frame["input"].astype(float).round(6).tolist(),
                    "inverse": frame["inverse"].astype(float).round(6).tolist(),
                    "input_rms": frame["input_rms"],
                    "input_dbfs": frame["input_dbfs"],
                    "inverse_rms": frame["inverse_rms"],
                    "inverse_dbfs": frame["inverse_dbfs"],
                    "inverse_correlation": frame["inverse_correlation"],
                }
            )
        except Exception as exc:
            self.json_response({"ok": False, "status": str(exc)}, status=500)

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

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

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
    global DEFAULT_SAMPLE_RATE, DEFAULT_BLOCK_SIZE, DEFAULT_DISPLAY_SAMPLES
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--display-samples", type=int, default=DEFAULT_DISPLAY_SAMPLES)
    args = parser.parse_args()
    DEFAULT_SAMPLE_RATE = args.sample_rate
    DEFAULT_BLOCK_SIZE = args.block_size
    DEFAULT_DISPLAY_SAMPLES = args.display_samples
    SCOPE.display_samples = DEFAULT_DISPLAY_SAMPLES
    for port in range(args.port, args.port + 20):
        try:
            with ReusableThreadingTCPServer((args.host, port), Handler) as server:
                display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
                print(f"http://{display_host}:{port}/anc_visual_dashboard.html")
                server.serve_forever()
        except OSError:
            continue
    raise SystemExit(f"No open port found from {args.port} to {args.port + 19}")


if __name__ == "__main__":
    raise SystemExit(main())
