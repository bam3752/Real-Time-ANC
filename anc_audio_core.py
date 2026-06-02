from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from scipy import signal


SECONDARY_PATH_FILE = "secondary_path_estimate.npz"


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1.0e-12))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def hann_spectrum(x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 8:
        return np.zeros(1), np.zeros(1)
    n = min(4096, int(x.size))
    frame = np.asarray(x[-n:], dtype=np.float64)
    frame = frame - np.mean(frame)
    window = np.hanning(n)
    mag = np.abs(np.fft.rfft(frame * window))
    mag_db = 20.0 * np.log10(mag / max(n / 2.0, 1.0) + 1.0e-12)
    freq = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
    return freq, mag_db


def clip_audio(x: np.ndarray, limit: float = 0.8) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), -float(limit), float(limit)).astype(np.float32)


def estimate_delay_samples(
    reference: np.ndarray,
    measured: np.ndarray,
    sample_rate: int,
    max_delay_ms: float = 100.0,
) -> tuple[int, float]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    measured = np.asarray(measured, dtype=np.float32).reshape(-1)
    n = min(reference.size, measured.size)
    if n < 16:
        return 0, 0.0
    reference = reference[-n:] - float(np.mean(reference[-n:]))
    measured = measured[-n:] - float(np.mean(measured[-n:]))
    corr = signal.correlate(measured, reference, mode="full", method="fft")
    lags = signal.correlation_lags(measured.size, reference.size, mode="full")
    max_lag = int(max(1, sample_rate * max_delay_ms / 1000.0))
    mask = (lags >= 0) & (lags <= max_lag)
    if not np.any(mask):
        return 0, 0.0
    sub_corr = np.abs(corr[mask])
    sub_lags = lags[mask]
    index = int(np.argmax(sub_corr))
    confidence = float(sub_corr[index] / (np.mean(sub_corr) + 1.0e-12))
    return int(sub_lags[index]), confidence


def list_audio_devices() -> list[dict[str, object]]:
    devices = []
    for index, device in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": index,
                "name": str(device["name"]),
                "max_input_channels": int(device["max_input_channels"]),
                "max_output_channels": int(device["max_output_channels"]),
                "default_samplerate": float(device["default_samplerate"]),
            }
        )
    return devices


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data = np.zeros(self.capacity, dtype=np.float32)
        self.index = 0
        self.full = False
        self.lock = threading.Lock()

    def push(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return
        with self.lock:
            if values.size >= self.capacity:
                self.data[:] = values[-self.capacity :]
                self.index = 0
                self.full = True
                return
            end = self.index + values.size
            if end <= self.capacity:
                self.data[self.index : end] = values
            else:
                first = self.capacity - self.index
                self.data[self.index :] = values[:first]
                self.data[: end % self.capacity] = values[first:]
                self.full = True
            self.index = end % self.capacity
            if self.index == 0:
                self.full = True

    def snapshot(self) -> np.ndarray:
        with self.lock:
            if self.full:
                return np.concatenate((self.data[self.index :], self.data[: self.index])).copy()
            return self.data[: self.index].copy()

    def clear(self) -> None:
        with self.lock:
            self.data.fill(0.0)
            self.index = 0
            self.full = False


class DelayLine:
    def __init__(self, max_delay_samples: int = 96_000) -> None:
        self.buffer = np.zeros(int(max_delay_samples) + 1, dtype=np.float32)
        self.write_index = 0

    def process(self, x: np.ndarray, delay_samples: int) -> np.ndarray:
        delay = int(np.clip(delay_samples, 0, self.buffer.size - 1))
        y = np.empty_like(x, dtype=np.float32)
        for i, sample in enumerate(np.asarray(x, dtype=np.float32)):
            read_index = (self.write_index - delay) % self.buffer.size
            y[i] = self.buffer[read_index]
            self.buffer[self.write_index] = sample
            self.write_index = (self.write_index + 1) % self.buffer.size
        return y

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.write_index = 0


class SmoothGain:
    def __init__(self, value: float = 0.0, max_step: float = 0.002) -> None:
        self.current = float(value)
        self.target = float(value)
        self.max_step = float(max_step)

    def set_target(self, target: float) -> None:
        self.target = float(target)

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float32)
        for i, sample in enumerate(np.asarray(x, dtype=np.float32)):
            delta = np.clip(self.target - self.current, -self.max_step, self.max_step)
            self.current += float(delta)
            y[i] = sample * self.current
        return y

    def force_zero(self) -> None:
        self.current = 0.0
        self.target = 0.0


class BlockLowPass:
    def __init__(self, sample_rate: int, cutoff_hz: float = 800.0) -> None:
        self.sample_rate = int(sample_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.zi = np.zeros((1, 2), dtype=np.float64)
        self.configure(sample_rate, cutoff_hz)

    def configure(self, sample_rate: int, cutoff_hz: float) -> None:
        self.sample_rate = int(sample_rate)
        nyquist = max(1.0, self.sample_rate / 2.0)
        cutoff = float(np.clip(cutoff_hz, 20.0, nyquist * 0.9))
        self.cutoff_hz = cutoff
        self.sos = signal.butter(4, cutoff / nyquist, btype="lowpass", output="sos")
        self.zi = signal.sosfilt_zi(self.sos) * 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, np.asarray(x, dtype=np.float64), zi=self.zi)
        return y.astype(np.float32)


class AnalysisFilterBank:
    def __init__(
        self,
        sample_rate: int,
        edges_hz: tuple[float, ...] = (80.0, 160.0, 320.0, 640.0, 1200.0),
        order: int = 2,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.edges_hz = tuple(float(x) for x in edges_hz)
        self.order = int(order)
        self.filters: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.configure(sample_rate, self.edges_hz, order)

    def configure(
        self,
        sample_rate: int,
        edges_hz: tuple[float, ...] | None = None,
        order: int | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        if edges_hz is not None:
            self.edges_hz = tuple(float(x) for x in edges_hz)
        if order is not None:
            self.order = int(order)
        nyquist = max(1.0, self.sample_rate / 2.0)
        edges = [float(np.clip(x, 20.0, nyquist * 0.92)) for x in self.edges_hz]
        edges = sorted(set(edges))
        bands: list[tuple[float, float | None]] = []
        previous = 20.0
        for edge in edges:
            if edge > previous:
                bands.append((previous, edge))
            previous = edge
        if previous < nyquist * 0.92:
            bands.append((previous, None))
        self.filters = []
        self.states = []
        for low, high in bands:
            if high is None:
                sos = signal.butter(self.order, low / nyquist, btype="highpass", output="sos")
            elif low <= 20.0:
                sos = signal.butter(self.order, high / nyquist, btype="lowpass", output="sos")
            else:
                sos = signal.butter(self.order, (low / nyquist, high / nyquist), btype="bandpass", output="sos")
            self.filters.append(sos)
            self.states.append(signal.sosfilt_zi(sos) * 0.0)

    def process(self, x: np.ndarray) -> list[np.ndarray]:
        out = []
        for i, sos in enumerate(self.filters):
            y, self.states[i] = signal.sosfilt(sos, np.asarray(x, dtype=np.float64), zi=self.states[i])
            out.append(y.astype(np.float32))
        return out

    def reset(self) -> None:
        for i, sos in enumerate(self.filters):
            self.states[i] = signal.sosfilt_zi(sos) * 0.0


class SafetyLimiter:
    def __init__(self, ceiling: float = 0.8, release: float = 0.9995, eps: float = 1.0e-9) -> None:
        self.ceiling = float(ceiling)
        self.release = float(release)
        self.eps = float(eps)
        self.gain = 1.0
        self.last_peak = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        self.last_peak = peak
        target = min(1.0, self.ceiling / (peak + self.eps))
        if target < self.gain:
            self.gain = target
        else:
            self.gain = min(1.0, self.gain / max(self.release, 1.0e-6))
        return clip_audio(x * self.gain, self.ceiling)

    def reset(self) -> None:
        self.gain = 1.0
        self.last_peak = 0.0


class FxNLMSFilter:
    def __init__(
        self,
        filter_len: int = 128,
        secondary_path: np.ndarray | None = None,
        mu: float = 0.05,
        leak: float = 0.0,
        eps: float = 1.0e-6,
    ) -> None:
        self.filter_len = int(filter_len)
        self.secondary_path = np.asarray(
            secondary_path if secondary_path is not None else np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        ).reshape(-1)
        self.mu = float(mu)
        self.leak = float(leak)
        self.eps = float(eps)
        self.freeze = False
        self.w = np.zeros(self.filter_len, dtype=np.float32)
        self.x_buf = np.zeros(self.filter_len, dtype=np.float32)
        self.x_secondary_buf = np.zeros(self.secondary_path.size, dtype=np.float32)
        self.filtered_x_buf = np.zeros(self.filter_len, dtype=np.float32)

    def set_secondary_path(self, secondary_path: np.ndarray) -> None:
        secondary_path = np.asarray(secondary_path, dtype=np.float32).reshape(-1)
        if secondary_path.size == 0:
            secondary_path = np.array([1.0], dtype=np.float32)
        self.secondary_path = secondary_path
        self.x_secondary_buf = np.zeros(self.secondary_path.size, dtype=np.float32)
        self.filtered_x_buf.fill(0.0)

    def reset(self) -> None:
        self.w.fill(0.0)
        self.x_buf.fill(0.0)
        self.x_secondary_buf.fill(0.0)
        self.filtered_x_buf.fill(0.0)

    def predict_sample(self, x: float) -> float:
        self.x_buf[1:] = self.x_buf[:-1]
        self.x_buf[0] = float(x)
        self.x_secondary_buf[1:] = self.x_secondary_buf[:-1]
        self.x_secondary_buf[0] = float(x)
        filtered_x = float(np.dot(self.secondary_path, self.x_secondary_buf))
        self.filtered_x_buf[1:] = self.filtered_x_buf[:-1]
        self.filtered_x_buf[0] = filtered_x
        return float(np.dot(self.w, self.x_buf))

    def adapt(self, error: float) -> None:
        if self.freeze:
            return
        power = float(np.dot(self.filtered_x_buf, self.filtered_x_buf)) + self.eps
        step = self.mu * float(error) / power
        self.w *= 1.0 - self.leak
        self.w -= step * self.filtered_x_buf
        np.clip(self.w, -4.0, 4.0, out=self.w)

    def process_with_error(self, x: np.ndarray, error: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.empty_like(x, dtype=np.float32)
        if error is None:
            for i, sample in enumerate(x):
                y[i] = self.predict_sample(float(sample))
            return y
        error = np.asarray(error, dtype=np.float32)
        for i, sample in enumerate(x):
            y[i] = self.predict_sample(float(sample))
            self.adapt(float(error[i]))
        return y


class SubbandFxNLMSFilter:
    def __init__(
        self,
        sample_rate: int = 48_000,
        bands_hz: tuple[float, ...] = (80.0, 160.0, 320.0, 640.0, 1200.0),
        filter_len: int = 64,
        secondary_path: np.ndarray | None = None,
        mu: float = 0.02,
        leak: float = 1.0e-5,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.bands_hz = tuple(float(x) for x in bands_hz)
        self.filter_len = int(filter_len)
        self.secondary_path = np.asarray(
            secondary_path if secondary_path is not None else SecondaryPathEstimator.default_path(),
            dtype=np.float32,
        ).reshape(-1)
        self.mu = float(mu)
        self.freeze = False
        self.ref_bank = AnalysisFilterBank(self.sample_rate, self.bands_hz)
        self.err_bank = AnalysisFilterBank(self.sample_rate, self.bands_hz)
        self.controllers = [
            FxNLMSFilter(self.filter_len, self.secondary_path, mu=self.mu, leak=leak)
            for _ in self.ref_bank.filters
        ]

    def configure(
        self,
        sample_rate: int | None = None,
        bands_hz: tuple[float, ...] | None = None,
        secondary_path: np.ndarray | None = None,
        mu: float | None = None,
    ) -> None:
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            self.sample_rate = int(sample_rate)
            self.ref_bank.configure(self.sample_rate, bands_hz or self.bands_hz)
            self.err_bank.configure(self.sample_rate, bands_hz or self.bands_hz)
        elif bands_hz is not None and tuple(float(x) for x in bands_hz) != self.bands_hz:
            self.ref_bank.configure(self.sample_rate, bands_hz)
            self.err_bank.configure(self.sample_rate, bands_hz)
        if bands_hz is not None:
            self.bands_hz = tuple(float(x) for x in bands_hz)
        if secondary_path is not None:
            self.secondary_path = np.asarray(secondary_path, dtype=np.float32).reshape(-1)
        if mu is not None:
            self.mu = float(mu)
        if len(self.controllers) != len(self.ref_bank.filters):
            self.controllers = [
                FxNLMSFilter(self.filter_len, self.secondary_path, mu=self.mu, leak=1.0e-5)
                for _ in self.ref_bank.filters
            ]
        for controller in self.controllers:
            controller.set_secondary_path(self.secondary_path)
            controller.mu = self.mu

    def reset(self) -> None:
        self.ref_bank.reset()
        self.err_bank.reset()
        for controller in self.controllers:
            controller.reset()

    def process_with_error(self, x: np.ndarray, error: np.ndarray | None = None) -> np.ndarray:
        x_bands = self.ref_bank.process(x)
        e_bands = self.err_bank.process(error) if error is not None else [None] * len(x_bands)
        y = np.zeros_like(np.asarray(x, dtype=np.float32), dtype=np.float32)
        for controller, xb, eb in zip(self.controllers, x_bands, e_bands):
            controller.freeze = self.freeze
            controller.mu = self.mu
            y += controller.process_with_error(xb, eb)
        return y.astype(np.float32)


class SecondaryPathEstimator:
    @staticmethod
    def identify_nlms(excitation: np.ndarray, measured: np.ndarray, taps: int = 128, mu: float = 0.3) -> np.ndarray:
        excitation = np.asarray(excitation, dtype=np.float32).reshape(-1)
        measured = np.asarray(measured, dtype=np.float32).reshape(-1)
        n = min(excitation.size, measured.size)
        excitation = excitation[:n]
        measured = measured[:n]
        w = np.zeros(int(taps), dtype=np.float32)
        xbuf = np.zeros(int(taps), dtype=np.float32)
        for x, d in zip(excitation, measured):
            xbuf[1:] = xbuf[:-1]
            xbuf[0] = x
            y = float(np.dot(w, xbuf))
            e = float(d) - y
            norm = float(np.dot(xbuf, xbuf)) + 1.0e-6
            w += (float(mu) * e / norm) * xbuf
        return w

    @staticmethod
    def default_path() -> np.ndarray:
        path = np.zeros(96, dtype=np.float32)
        path[12] = 0.8
        path[18] = -0.22
        path[31] = 0.12
        path[47] = -0.06
        win = signal.windows.hann(15, sym=True).astype(np.float32)
        path[12:27] += 0.04 * win
        return path


class OnlineSecondaryPathTracker:
    def __init__(
        self,
        sample_rate: int = 48_000,
        taps: int = 128,
        mu: float = 0.05,
        probe_gain: float = 0.002,
        probe_cutoff_hz: float = 1000.0,
        eps: float = 1.0e-6,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.taps = int(taps)
        self.mu = float(mu)
        self.probe_gain = float(probe_gain)
        self.eps = float(eps)
        self.enabled = False
        self.freeze = False
        self.w = SecondaryPathEstimator.default_path()
        if self.w.size != self.taps:
            self.w = np.pad(self.w, (0, max(0, self.taps - self.w.size)))[: self.taps].astype(np.float32)
        self.xbuf = np.zeros(self.taps, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.probe_filter = BlockLowPass(self.sample_rate, probe_cutoff_hz)
        self.last_error_rms = 0.0
        self.last_probe_rms = 0.0

    def configure(
        self,
        sample_rate: int | None = None,
        taps: int | None = None,
        mu: float | None = None,
        probe_gain: float | None = None,
        probe_cutoff_hz: float | None = None,
    ) -> None:
        if sample_rate is not None:
            self.sample_rate = int(sample_rate)
        if taps is not None and int(taps) != self.taps:
            self.taps = int(taps)
            self.w = np.pad(self.w, (0, max(0, self.taps - self.w.size)))[: self.taps].astype(np.float32)
            self.xbuf = np.zeros(self.taps, dtype=np.float32)
        if mu is not None:
            self.mu = float(mu)
        if probe_gain is not None:
            self.probe_gain = float(probe_gain)
        if sample_rate is not None or probe_cutoff_hz is not None:
            cutoff = self.probe_filter.cutoff_hz if probe_cutoff_hz is None else float(probe_cutoff_hz)
            self.probe_filter.configure(self.sample_rate, cutoff)

    def make_probe(self, n: int, gain: float | None = None) -> np.ndarray:
        level = self.probe_gain if gain is None else float(gain)
        if not self.enabled or level <= 0.0:
            return np.zeros(int(n), dtype=np.float32)
        raw = self.rng.standard_normal(int(n)).astype(np.float32)
        probe = self.probe_filter.process(raw)
        probe = probe / max(float(np.max(np.abs(probe))), 1.0e-6)
        probe = probe * level
        self.last_probe_rms = rms(probe)
        return probe.astype(np.float32)

    def adapt(
        self,
        probe: np.ndarray,
        measured_error: np.ndarray,
        disturbance_estimate: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.freeze:
            return self.w.copy()
        probe = np.asarray(probe, dtype=np.float32).reshape(-1)
        measured_error = np.asarray(measured_error, dtype=np.float32).reshape(-1)
        if disturbance_estimate is not None:
            measured_error = measured_error - np.asarray(disturbance_estimate, dtype=np.float32).reshape(-1)[: measured_error.size]
        n = min(probe.size, measured_error.size)
        for x, d in zip(probe[:n], measured_error[:n]):
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = x
            y = float(np.dot(self.w, self.xbuf))
            e = float(d) - y
            norm = float(np.dot(self.xbuf, self.xbuf)) + self.eps
            self.w += (self.mu * e / norm) * self.xbuf
        np.clip(self.w, -2.0, 2.0, out=self.w)
        self.last_error_rms = rms(measured_error[:n])
        return self.w.copy()

    def path(self) -> np.ndarray:
        return self.w.copy()

    def reset(self, path: np.ndarray | None = None) -> None:
        if path is None:
            self.w = SecondaryPathEstimator.default_path()
            if self.w.size != self.taps:
                self.w = np.pad(self.w, (0, max(0, self.taps - self.w.size)))[: self.taps].astype(np.float32)
        else:
            self.w = np.asarray(path, dtype=np.float32).reshape(-1)
            self.taps = self.w.size
        self.xbuf = np.zeros(self.taps, dtype=np.float32)
        self.last_error_rms = 0.0
        self.last_probe_rms = 0.0


@dataclass
class AudioMetrics:
    input_rms: float = 0.0
    anti_rms: float = 0.0
    error_rms: float = 0.0
    probe_rms: float = 0.0
    reduction_db: float = 0.0
    limiter_gain: float = 1.0
    limiter_peak: float = 0.0
    delay_samples: int = 0
    delay_confidence: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "input_rms": self.input_rms,
            "input_dbfs": dbfs(self.input_rms),
            "anti_rms": self.anti_rms,
            "anti_dbfs": dbfs(self.anti_rms),
            "error_rms": self.error_rms,
            "error_dbfs": dbfs(self.error_rms),
            "probe_rms": self.probe_rms,
            "probe_dbfs": dbfs(self.probe_rms),
            "reduction_db": self.reduction_db,
            "limiter_gain": self.limiter_gain,
            "limiter_peak": self.limiter_peak,
            "delay_samples": float(self.delay_samples),
            "delay_confidence": self.delay_confidence,
        }


@dataclass
class AudioSettings:
    controller: str = "inversion"
    sample_rate: int = 48_000
    block_size: int = 256
    channels: int = 1
    output_enabled: bool = False
    muted: bool = False
    gain: float = 0.02
    delay_samples: int = 0
    lowpass_enabled: bool = True
    lowpass_hz: float = 800.0
    fx_mu: float = 0.03
    fx_filter_len: int = 128
    fx_freeze: bool = False
    subband_edges_hz: tuple[float, ...] = (80.0, 160.0, 320.0, 640.0, 1200.0)
    limiter_ceiling: float = 0.8
    online_secondary_enabled: bool = False
    secondary_mu: float = 0.05
    probe_gain: float = 0.002
    probe_cutoff_hz: float = 1000.0


class AudioEngine:
    def __init__(self, process_callback) -> None:
        self.process_callback = process_callback
        self.stream: sd.Stream | None = None
        self.running = False
        self.status_queue: queue.Queue[str] = queue.Queue()

    def start(
        self,
        input_device: int | None,
        output_device: int | None,
        sample_rate: int,
        block_size: int,
        input_channels: int,
        output_channels: int,
    ) -> None:
        self.stop()
        self.stream = sd.Stream(
            device=(input_device, output_device),
            samplerate=sample_rate,
            blocksize=block_size,
            channels=(input_channels, output_channels),
            dtype="float32",
            callback=self._callback,
            latency="low",
        )
        self.stream.start()
        self.running = True

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.running = False

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            self.status_queue.put(str(status))
        try:
            anti = self.process_callback(indata.copy(), frames)
            anti = np.asarray(anti, dtype=np.float32).reshape(-1)
            if anti.size != frames:
                anti = np.zeros(frames, dtype=np.float32)
            outdata[:] = 0.0
            outdata[:, :] = anti[:, None]
        except Exception as exc:
            self.status_queue.put(f"Audio callback muted after error: {exc}")
            outdata[:] = 0.0


class RealtimeInversionProcessor:
    def __init__(self, settings: AudioSettings | None = None) -> None:
        self.settings = settings or AudioSettings()
        self.delay_line = DelayLine(96_000)
        self.gain = SmoothGain(0.0)
        self.lowpass = BlockLowPass(self.settings.sample_rate, self.settings.lowpass_hz)
        self.ref_ring = RingBuffer(self.settings.sample_rate * 5)
        self.anti_ring = RingBuffer(self.settings.sample_rate * 5)
        self.last_input_rms = 0.0
        self.last_anti_rms = 0.0
        self.lock = threading.Lock()

    def configure(self, settings: AudioSettings) -> None:
        with self.lock:
            rebuild_filter = (
                self.settings.sample_rate != settings.sample_rate
                or self.settings.lowpass_hz != settings.lowpass_hz
            )
            self.settings = settings
            if rebuild_filter:
                self.lowpass.configure(settings.sample_rate, settings.lowpass_hz)

    def process(self, indata: np.ndarray, frames: int) -> np.ndarray:
        with self.lock:
            settings = AudioSettings(**self.settings.__dict__)
        ref = indata[:, 0].astype(np.float32) if indata.ndim == 2 and indata.shape[1] >= 1 else np.zeros(frames, dtype=np.float32)
        anti = -self.delay_line.process(ref, settings.delay_samples)
        if settings.lowpass_enabled:
            anti = self.lowpass.process(anti)
        target_gain = settings.gain if settings.output_enabled and not settings.muted else 0.0
        self.gain.set_target(target_gain)
        anti = clip_audio(self.gain.apply(anti))
        self.ref_ring.push(ref)
        self.anti_ring.push(anti)
        self.last_input_rms = rms(ref)
        self.last_anti_rms = rms(anti)
        return anti


class RealtimeANCProcessor:
    def __init__(self, settings: AudioSettings | None = None) -> None:
        self.settings = settings or AudioSettings()
        self.secondary_path = SecondaryPathEstimator.default_path()
        self.delay_line = DelayLine(96_000)
        self.gain = SmoothGain(0.0)
        self.lowpass = BlockLowPass(self.settings.sample_rate, self.settings.lowpass_hz)
        self.limiter = SafetyLimiter(self.settings.limiter_ceiling)
        self.fx_filter = FxNLMSFilter(
            self.settings.fx_filter_len,
            self.secondary_path,
            mu=self.settings.fx_mu,
            leak=1.0e-5,
        )
        self.subband_filter = SubbandFxNLMSFilter(
            self.settings.sample_rate,
            self.settings.subband_edges_hz,
            max(16, self.settings.fx_filter_len // 2),
            self.secondary_path,
            mu=self.settings.fx_mu,
        )
        self.secondary_tracker = OnlineSecondaryPathTracker(
            self.settings.sample_rate,
            taps=self.secondary_path.size,
            mu=self.settings.secondary_mu,
            probe_gain=self.settings.probe_gain,
            probe_cutoff_hz=self.settings.probe_cutoff_hz,
        )
        self.ref_ring = RingBuffer(self.settings.sample_rate * 5)
        self.anti_ring = RingBuffer(self.settings.sample_rate * 5)
        self.err_ring = RingBuffer(self.settings.sample_rate * 5)
        self.probe_ring = RingBuffer(self.settings.sample_rate * 5)
        self.metrics = AudioMetrics()
        self.lock = threading.Lock()

    def configure(self, settings: AudioSettings) -> None:
        with self.lock:
            old = self.settings
            self.settings = settings
            if old.sample_rate != settings.sample_rate or old.lowpass_hz != settings.lowpass_hz:
                self.lowpass.configure(settings.sample_rate, settings.lowpass_hz)
            self.limiter.ceiling = float(settings.limiter_ceiling)
            self.fx_filter.mu = float(settings.fx_mu)
            self.fx_filter.freeze = bool(settings.fx_freeze)
            self.fx_filter.set_secondary_path(self.secondary_path)
            self.subband_filter.configure(
                sample_rate=settings.sample_rate,
                bands_hz=settings.subband_edges_hz,
                secondary_path=self.secondary_path,
                mu=settings.fx_mu,
            )
            self.subband_filter.freeze = bool(settings.fx_freeze)
            self.secondary_tracker.enabled = bool(settings.online_secondary_enabled)
            self.secondary_tracker.configure(
                sample_rate=settings.sample_rate,
                taps=self.secondary_path.size,
                mu=settings.secondary_mu,
                probe_gain=settings.probe_gain,
                probe_cutoff_hz=settings.probe_cutoff_hz,
            )
            if old.sample_rate != settings.sample_rate:
                self.ref_ring = RingBuffer(settings.sample_rate * 5)
                self.anti_ring = RingBuffer(settings.sample_rate * 5)
                self.err_ring = RingBuffer(settings.sample_rate * 5)
                self.probe_ring = RingBuffer(settings.sample_rate * 5)

    def set_secondary_path(self, path: np.ndarray) -> None:
        with self.lock:
            path = np.asarray(path, dtype=np.float32).reshape(-1)
            if path.size == 0:
                path = np.array([1.0], dtype=np.float32)
            self.secondary_path = path
            self.fx_filter.set_secondary_path(path)
            self.subband_filter.configure(secondary_path=path)
            self.secondary_tracker.reset(path)

    def reset(self) -> None:
        with self.lock:
            self.delay_line.reset()
            self.gain.force_zero()
            self.limiter.reset()
            self.fx_filter.reset()
            self.subband_filter.reset()
            self.secondary_tracker.reset(self.secondary_path)
            self.ref_ring.clear()
            self.anti_ring.clear()
            self.err_ring.clear()
            self.probe_ring.clear()
            self.metrics = AudioMetrics()

    def estimate_recent_delay(self, max_delay_ms: float = 100.0) -> tuple[int, float]:
        delay, confidence = estimate_delay_samples(
            self.ref_ring.snapshot(),
            self.err_ring.snapshot(),
            self.settings.sample_rate,
            max_delay_ms,
        )
        self.metrics.delay_samples = delay
        self.metrics.delay_confidence = confidence
        return delay, confidence

    def process(self, indata: np.ndarray, frames: int) -> np.ndarray:
        with self.lock:
            settings = AudioSettings(**self.settings.__dict__)
        ref = indata[:, 0].astype(np.float32) if indata.ndim == 2 and indata.shape[1] >= 1 else np.zeros(frames, dtype=np.float32)
        err = indata[:, 1].astype(np.float32) if indata.ndim == 2 and indata.shape[1] >= 2 else None
        controller = settings.controller.lower().strip()
        if controller in {"fxnlms", "fxlms"} and err is not None:
            anti = self.fx_filter.process_with_error(ref, err)
        elif controller in {"subband", "subband_fxnlms", "subband_fxlms"} and err is not None:
            anti = self.subband_filter.process_with_error(ref, err)
        else:
            anti = -self.delay_line.process(ref, settings.delay_samples)
        if settings.lowpass_enabled:
            anti = self.lowpass.process(anti)
        target_gain = settings.gain if settings.output_enabled and not settings.muted else 0.0
        self.gain.set_target(target_gain)
        anti = self.gain.apply(anti)
        if settings.output_enabled and not settings.muted and settings.online_secondary_enabled:
            probe = self.secondary_tracker.make_probe(frames, settings.probe_gain)
            anti = anti + probe
            if err is not None and rms(probe) > 0.0:
                self.secondary_path = self.secondary_tracker.adapt(probe, err)
                self.fx_filter.set_secondary_path(self.secondary_path)
                self.subband_filter.configure(secondary_path=self.secondary_path)
        else:
            probe = np.zeros(frames, dtype=np.float32)
        anti = self.limiter.process(anti)
        err_data = err if err is not None else np.zeros(frames, dtype=np.float32)
        self.ref_ring.push(ref)
        self.anti_ring.push(anti)
        self.err_ring.push(err_data)
        self.probe_ring.push(probe)
        input_rms = rms(ref)
        anti_rms = rms(anti)
        error_rms = rms(err_data)
        probe_rms = rms(probe)
        reduction = dbfs(input_rms) - dbfs(error_rms) if err is not None and error_rms > 0.0 else 0.0
        self.metrics = AudioMetrics(
            input_rms=input_rms,
            anti_rms=anti_rms,
            error_rms=error_rms,
            probe_rms=probe_rms,
            reduction_db=reduction,
            limiter_gain=self.limiter.gain,
            limiter_peak=self.limiter.last_peak,
            delay_samples=self.metrics.delay_samples,
            delay_confidence=self.metrics.delay_confidence,
        )
        return anti.astype(np.float32)


class FxLMSSimulation:
    def __init__(self, sample_rate: int = 48_000) -> None:
        self.sample_rate = int(sample_rate)
        self.primary = np.array([0.0] * 24 + [0.9, 0.25, -0.12, 0.06], dtype=np.float32)
        self.secondary = SecondaryPathEstimator.default_path()
        self.controller = FxNLMSFilter(160, self.secondary, mu=0.04, leak=1.0e-5)
        self.pbuf = np.zeros(self.primary.size, dtype=np.float32)
        self.ybuf = np.zeros(self.secondary.size, dtype=np.float32)
        self.t = 0
        self.error_history = RingBuffer(48_000)
        self.ref_history = RingBuffer(48_000)
        self.anti_history = RingBuffer(48_000)

    def reset(self) -> None:
        self.controller.reset()
        self.pbuf.fill(0.0)
        self.ybuf.fill(0.0)
        self.t = 0
        self.error_history.clear()
        self.ref_history.clear()
        self.anti_history.clear()

    def step(self, n: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = np.empty(n, dtype=np.float32)
        anti = np.empty(n, dtype=np.float32)
        err = np.empty(n, dtype=np.float32)
        for i in range(n):
            tt = self.t / float(self.sample_rate)
            fan = (
                0.45 * math.sin(2.0 * math.pi * 90.0 * tt)
                + 0.22 * math.sin(2.0 * math.pi * 180.0 * tt + 0.3)
                + 0.14 * math.sin(2.0 * math.pi * 270.0 * tt + 1.1)
                + 0.04 * np.random.randn()
            )
            self.pbuf[1:] = self.pbuf[:-1]
            self.pbuf[0] = fan
            primary_at_error = float(np.dot(self.primary, self.pbuf))
            y = self.controller.predict_sample(fan)
            self.ybuf[1:] = self.ybuf[:-1]
            self.ybuf[0] = y
            secondary_at_error = float(np.dot(self.secondary, self.ybuf))
            e = primary_at_error + secondary_at_error
            self.controller.adapt(e)
            ref[i] = fan
            anti[i] = y
            err[i] = e
            self.t += 1
        self.ref_history.push(ref)
        self.anti_history.push(anti)
        self.error_history.push(err)
        return ref, anti, err
