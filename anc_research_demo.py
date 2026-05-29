#!/usr/bin/env python3
"""
Real-Time ANC Research Demo

Install:
    pip install sounddevice numpy scipy PyQt6 pyqtgraph

Run:
    python anc_research_demo.py

This is a school-project prototype for active noise control concepts. It is
not a production ANC system and it defaults to muted output.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg


APP_TITLE = "Real-Time Active Noise Control Research Demo"
SECONDARY_PATH_FILE = "secondary_path_estimate.npz"


REPORT_MD = r"""
# Active Noise Control Research Notes

## What ANC really is

Active noise control (ANC) uses a loudspeaker to produce a pressure wave that
destructively interferes with an unwanted pressure wave at a target location,
usually near an ear or an error microphone. That target location matters. A
waveform that is inverted electrically is not automatically opposite in acoustic
phase after it travels through a DAC, amplifier, loudspeaker, air, housing,
ear cup, microphone, and ADC.

Paul Lueg's 1936 patent, US2043416A, described the core idea: receive sound and
reproduce a related sound in opposite phase. The patent also depends on geometry
and propagation timing: the microphone and loudspeaker positions determine
whether the reproduced wave arrives at the cancellation point in phase
opposition. Source: https://patents.google.com/patent/US2043416A/en

## Why passive isolation still matters

Passive isolation reduces the amount of sound that reaches the target before
the controller has to do anything. Foam, seals, ear cups, dense panels, and
mechanical fit are especially important for high-frequency and impulsive noise,
where wavelengths are short and small placement errors ruin cancellation. ANC is
most useful as a low-frequency addition to passive isolation, not a replacement
for it.

## Feedforward, feedback, and hybrid ANC

Feedforward ANC places a reference microphone closer to the noise source or
outside the ear cup. It measures the disturbance before it reaches the protected
point, then the controller predicts the anti-noise. This can work very well when
the reference signal is coherent with the later disturbance and the processing
delay is smaller than the acoustic delay.

Feedback ANC places an error microphone near the ear or protected point. It
measures the residual noise and corrects the loudspeaker output. Feedback can
adapt to leakage and fit changes, but it must be carefully stabilized because it
forms an acoustic feedback loop.

Hybrid ANC combines both: a feedforward path predicts incoming noise and a
feedback path corrects what remains. Modern headphones often use hybrid systems
because no single microphone location is robust across fit, wind, speech,
movement, and changing noise fields.

Apple US20140086425A1 describes multiple reference microphone signals, adaptive
filters, an error microphone, and coherence-weighted combining of anti-noise
components. Bose US9445184B2 describes an active noise reduction headphone with
multiple feedback microphones and combining of their signals. Apple
US20220084494A1 / US11335316 describes multiple reference microphones in an ear
cup used for ANC and transparency, with oversight for wind and scratch events.
Recent hybrid patent work, such as US12548544, explicitly combines adaptive
feedforward ANC with feedback filtering in leaky headphones.

Sources:
- https://patents.google.com/patent/US20140086425A1/en
- https://patents.google.com/patent/US9445184B2/en
- https://patents.google.com/patent/US20220084494A1/en
- https://patents.justia.com/patent/11335316
- https://patents.justia.com/patent/12548544

## LMS, NLMS, FxLMS, and secondary path modeling

LMS adjusts FIR filter coefficients in the direction that reduces measured
error. NLMS improves stability by normalizing the step size by input power.

Ordinary LMS is not enough for real ANC because the controller output does not
appear instantly at the error microphone. The loudspeaker-to-error-microphone
path is called the secondary path. It includes the DAC, amplifier, speaker,
acoustic path, microphone, ADC, and buffering. FxLMS handles this by filtering
the reference signal through an estimate of the secondary path before updating
the adaptive filter. This filtered reference gives the adaptive update the
correct phase and magnitude information.

US20160300563A1 describes secondary path estimation in ANC systems. Reviews of
FxLMS systems treat secondary-path compensation as a central requirement.
Sources:
- https://patents.google.com/patent/US20160300563A1/en
- https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-023-01088-x

## Multi-microphone and subband ANC

Multiple reference microphones improve robustness because different microphones
can be more coherent with the disturbance under different wind, fit, direction,
or handling conditions. Multiple error microphones can enlarge or stabilize the
quiet zone by measuring residual sound at more than one point.

Subband and frequency-domain ANC split the problem into frequency regions. This
can reduce computation for long filters and allow different adaptation rates per
band. The tradeoff is block-processing delay. A frequency-domain FxLMS review
notes that delay has a major effect on broadband noise reduction, which is why
true ANC systems are designed around very low-latency signal paths.
Source: https://www.mdpi.com/2076-3417/8/11/2313

## Why ANC focuses on low frequencies

Low-frequency noise has long wavelengths and is often steady: engines, fans, air
conditioners, transformer hum, and road rumble. These are easier to predict and
cancel across a small region. High-frequency sound has short wavelengths, so a
few centimeters of microphone or ear movement can change the phase dramatically.
Sudden sounds are also difficult because the anti-noise must arrive before or at
the same time as the disturbance.

## Latency warning

Feedforward ANC has a causality constraint: the anti-noise must reach the target
no later than the primary noise. Bluetooth microphones, AirPods output,
including AirPods Pro models, and normal computer audio stacks add codec,
buffering, scheduling, and device latency. Using one physical device for input
and a different physical device for output can add still more delay because the
devices have independent audio clocks and the operating system has to resample
and buffer between them. They are useful for visualization and classroom
demonstrations, but they are not suitable for true open-air cancellation
controlled by this Python app. Wired low-latency audio interfaces and physically
close microphone/speaker geometry are recommended.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10516317/

## What this prototype demonstrates

This app demonstrates the difference between:
- a simple waveform inversion visualization,
- manual and estimated delay compensation,
- a controlled FxLMS simulation where convergence can be measured, and
- an experimental real-time FxLMS mode that requires an error microphone and a
  secondary-path estimate.

It also measures RMS, dBFS, spectrum, and simulation error reduction. It does
not claim true open-air cancellation unless an error microphone actually
measures reduced sound at the target point.

## Demonstration presets

The app includes presets for MacBook speakers, wired headphones, and Bluetooth
AirPods/EarPods. These presets configure sensible sample rate, block size,
channels, gain, and mode values, but they do not override the physics.
Bluetooth presets are visualization-only, and true ANC still needs low latency
plus an error microphone at the cancellation point.

## Safe school demonstration setup

Use wired audio. Start with output disabled. Use low volume. Put a small speaker
and microphones on a desk, not near anyone's ear. A stereo USB interface or
stereo USB microphone is best because channel 0 can be the reference microphone
and channel 1 can be the error microphone. Demonstrate Mode C first because the
simulation reliably shows FxLMS convergence without acoustic feedback risk.

## Presentation explanation

Start by saying: "Inverting a waveform is not the same thing as acoustic ANC."
Then show Mode A to make that limitation visible. Next show delay adjustment and
explain phase alignment. Then show the FxLMS simulation and explain that the
adaptive filter is learning how to drive the speaker through the secondary path.
Finally, show the real-time mode with output muted or very low, emphasizing that
real ANC performance depends on latency, physical placement, path modeling, and
measurement at the cancellation point.

## Limitations and future improvements

Limitations:
- Python and desktop OS audio are not hard real-time.
- Separate USB devices may drift because their audio clocks are not synchronized.
- Open-air cancellation creates a small quiet zone and can increase sound
  elsewhere.
- The real-time FxLMS mode is experimental and conservative.
- The built-in secondary-path estimator is a teaching tool, not a calibrated
  laboratory measurement.

Future improvements:
- JACK/CoreAudio low-latency routing support.
- True multichannel ANC with several reference and error microphones.
- Frequency-domain or subband FxLMS.
- Online secondary-path tracking with injected low-level probe noise.
- Better wind-noise and coherence detection.
- Exportable plots and report PDF generation.
"""


def dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(float(rms), 1.0e-12))


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
        for i, sample in enumerate(x):
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
        y, self.zi = signal.sosfilt(self.sos, x.astype(np.float64), zi=self.zi)
        return y.astype(np.float32)


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

    def process_with_error(self, x: np.ndarray, error: np.ndarray | None) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float32)
        if error is None:
            for i, sample in enumerate(x):
                y[i] = self.predict_sample(float(sample))
            return y
        for i, sample in enumerate(x):
            y[i] = self.predict_sample(float(sample))
            self.adapt(float(error[i]))
        return y


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


@dataclass
class AudioSettings:
    mode: str = "Simple inversion"
    sample_rate: int = 48_000
    block_size: int = 256
    channels: int = 1
    output_enabled: bool = False
    muted: bool = False
    gain: float = 0.02
    delay_samples: int = 0
    auto_delay_enabled: bool = False
    lowpass_enabled: bool = True
    lowpass_hz: float = 800.0
    fx_mu: float = 0.03
    fx_filter_len: int = 128
    fx_freeze: bool = False


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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        pg.setConfigOptions(antialias=True)

        self.settings = AudioSettings()
        self.settings_lock = threading.Lock()
        self.secondary_path = SecondaryPathEstimator.default_path()
        self.fx_filter = FxNLMSFilter(self.settings.fx_filter_len, self.secondary_path, mu=self.settings.fx_mu)
        self.delay_line = DelayLine(96_000)
        self.gain = SmoothGain(0.0)
        self.lowpass = BlockLowPass(self.settings.sample_rate, self.settings.lowpass_hz)
        self.engine = AudioEngine(self.process_audio)
        self.simulation = FxLMSSimulation(self.settings.sample_rate)
        self.output_warning_seen = False
        self.active_input_channels = 0
        self.active_output_channels = 0
        self.active_input_name = ""
        self.active_output_name = ""

        seconds = 5
        capacity = self.settings.sample_rate * seconds
        self.ref_ring = RingBuffer(capacity)
        self.anti_ring = RingBuffer(capacity)
        self.err_ring = RingBuffer(capacity)
        self.recent_ref = RingBuffer(self.settings.sample_rate)
        self.recent_err = RingBuffer(self.settings.sample_rate)

        self.last_metrics = {
            "input_rms": 0.0,
            "anti_rms": 0.0,
            "error_rms": 0.0,
            "reduction_db": 0.0,
            "delay": 0,
            "confidence": 0.0,
            "mode_note": "Stream is not running yet.",
        }
        self.metrics_lock = threading.Lock()

        self._build_ui()
        self.refresh_devices()

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(50)

        self.sim_timer = QtCore.QTimer(self)
        self.sim_timer.timeout.connect(self.run_simulation_tick)
        self.sim_timer.start(40)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.drain_status)
        self.status_timer.start(250)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)

        toolbar = QtWidgets.QGridLayout()
        layout.addLayout(toolbar)

        self.input_combo = QtWidgets.QComboBox()
        self.output_combo = QtWidgets.QComboBox()
        self.sample_rate_combo = QtWidgets.QComboBox()
        for rate in (8000, 16000, 24000, 44100, 48000, 96000):
            self.sample_rate_combo.addItem(str(rate), rate)
        self.sample_rate_combo.setCurrentText("48000")

        self.block_combo = QtWidgets.QComboBox()
        for block in (64, 128, 256, 512, 1024):
            self.block_combo.addItem(str(block), block)
        self.block_combo.setCurrentText("256")

        self.channels_combo = QtWidgets.QComboBox()
        self.channels_combo.addItem("1 channel", 1)
        self.channels_combo.addItem("2 channels: ref + error", 2)

        self.refresh_button = QtWidgets.QPushButton("Refresh Devices")
        self.start_button = QtWidgets.QPushButton("Start Stream")
        self.stop_button = QtWidgets.QPushButton("Stop Stream")
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("MacBook speakers demo", "macbook")
        self.preset_combo.addItem("Wired headphones demo", "wired")
        self.preset_combo.addItem("Bluetooth AirPods/EarPods demo", "bluetooth")
        self.apply_preset_button = QtWidgets.QPushButton("Apply Preset")
        self.output_enable = QtWidgets.QCheckBox("Output enabled")
        self.mute_button = QtWidgets.QCheckBox("Mute")
        self.emergency_button = QtWidgets.QPushButton("EMERGENCY STOP")
        self.emergency_button.setStyleSheet("font-weight: 700; color: white; background: #9d1c1c;")

        toolbar.addWidget(QtWidgets.QLabel("Input"), 0, 0)
        toolbar.addWidget(self.input_combo, 0, 1)
        toolbar.addWidget(QtWidgets.QLabel("Output"), 0, 2)
        toolbar.addWidget(self.output_combo, 0, 3)
        toolbar.addWidget(QtWidgets.QLabel("Sample rate"), 0, 4)
        toolbar.addWidget(self.sample_rate_combo, 0, 5)
        toolbar.addWidget(QtWidgets.QLabel("Block"), 0, 6)
        toolbar.addWidget(self.block_combo, 0, 7)
        toolbar.addWidget(QtWidgets.QLabel("Channels"), 0, 8)
        toolbar.addWidget(self.channels_combo, 0, 9)
        toolbar.addWidget(self.refresh_button, 1, 0)
        toolbar.addWidget(self.start_button, 1, 1)
        toolbar.addWidget(self.stop_button, 1, 2)
        toolbar.addWidget(self.output_enable, 1, 3)
        toolbar.addWidget(self.mute_button, 1, 4)
        toolbar.addWidget(self.emergency_button, 1, 5, 1, 2)
        toolbar.addWidget(QtWidgets.QLabel("Preset"), 2, 0)
        toolbar.addWidget(self.preset_combo, 2, 1, 1, 2)
        toolbar.addWidget(self.apply_preset_button, 2, 3)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_controls_tab()
        self._build_simulation_tab()
        self._build_report_tab()

        self.status_label = QtWidgets.QLabel("Output is OFF by default. Use wired low-latency devices for real experiments.")
        layout.addWidget(self.status_label)

        self.refresh_button.clicked.connect(self.refresh_devices)
        self.apply_preset_button.clicked.connect(self.apply_selected_preset)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.output_enable.toggled.connect(self.on_output_toggled)
        self.mute_button.toggled.connect(self.sync_settings)
        self.emergency_button.clicked.connect(self.emergency_stop)
        self.sample_rate_combo.currentIndexChanged.connect(self.sync_settings)
        self.block_combo.currentIndexChanged.connect(self.sync_settings)
        self.channels_combo.currentIndexChanged.connect(self.sync_settings)

    def _build_controls_tab(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        self.tabs.addTab(page, "Real-Time Modes")

        controls = QtWidgets.QVBoxLayout()
        layout.addLayout(controls, 0)

        self.mode_combo = QtWidgets.QComboBox()
        for mode in (
            "Simple inversion",
            "Delay compensated inversion",
            "Experimental FxLMS",
        ):
            self.mode_combo.addItem(mode)
        controls.addWidget(QtWidgets.QLabel("Mode"))
        controls.addWidget(self.mode_combo)

        self.config_label = QtWidgets.QLabel("")
        self.config_label.setWordWrap(True)
        self.config_label.setStyleSheet(
            "QLabel { background: #242424; border: 1px solid #666; padding: 8px; }"
        )
        controls.addWidget(self.config_label)

        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 500)
        self.gain_slider.setValue(20)
        self.gain_label = QtWidgets.QLabel("Gain: 0.020")
        controls.addWidget(self.gain_label)
        controls.addWidget(self.gain_slider)

        self.delay_spin = QtWidgets.QSpinBox()
        self.delay_spin.setRange(0, 96_000)
        self.delay_spin.setSuffix(" samples")
        self.delay_ms_label = QtWidgets.QLabel("Delay: 0.00 ms")
        controls.addWidget(QtWidgets.QLabel("Manual delay"))
        controls.addWidget(self.delay_spin)
        controls.addWidget(self.delay_ms_label)

        self.auto_delay_button = QtWidgets.QPushButton("Estimate Delay From Ref/Error")
        controls.addWidget(self.auto_delay_button)

        self.lowpass_check = QtWidgets.QCheckBox("Low-pass anti-noise")
        self.lowpass_check.setChecked(True)
        self.lowpass_spin = QtWidgets.QSpinBox()
        self.lowpass_spin.setRange(50, 3000)
        self.lowpass_spin.setValue(800)
        self.lowpass_spin.setSuffix(" Hz")
        controls.addWidget(self.lowpass_check)
        controls.addWidget(self.lowpass_spin)

        self.mu_spin = QtWidgets.QDoubleSpinBox()
        self.mu_spin.setRange(0.0001, 0.5)
        self.mu_spin.setSingleStep(0.005)
        self.mu_spin.setDecimals(4)
        self.mu_spin.setValue(0.03)
        self.freeze_check = QtWidgets.QCheckBox("Freeze FxLMS adaptation")
        self.reset_fx_button = QtWidgets.QPushButton("Reset FxLMS Filter")
        controls.addWidget(QtWidgets.QLabel("FxLMS/NLMS step size"))
        controls.addWidget(self.mu_spin)
        controls.addWidget(self.freeze_check)
        controls.addWidget(self.reset_fx_button)

        self.load_secondary_button = QtWidgets.QPushButton("Load Secondary Path")
        self.save_secondary_button = QtWidgets.QPushButton("Save Current Secondary Path")
        self.estimate_secondary_button = QtWidgets.QPushButton("Estimate Secondary Path From Recent Output/Error")
        controls.addWidget(self.load_secondary_button)
        controls.addWidget(self.save_secondary_button)
        controls.addWidget(self.estimate_secondary_button)

        self.metrics_label = QtWidgets.QLabel("Meters will appear after audio starts.")
        self.metrics_label.setWordWrap(True)
        controls.addWidget(self.metrics_label)
        controls.addStretch(1)

        plots = QtWidgets.QGridLayout()
        layout.addLayout(plots, 1)
        self.ref_plot = pg.PlotWidget(title="Live microphone waveform")
        self.anti_plot = pg.PlotWidget(title="Generated anti-noise waveform")
        self.err_plot = pg.PlotWidget(title="Error microphone waveform, if channel 2 exists")
        self.spec_in_plot = pg.PlotWidget(title="Incoming noise spectrum")
        self.spec_err_plot = pg.PlotWidget(title="Error/after-cancellation spectrum, if measured")
        self.ref_curve = self.ref_plot.plot(pen="#2b6cb0")
        self.anti_curve = self.anti_plot.plot(pen="#c05621")
        self.err_curve = self.err_plot.plot(pen="#2f855a")
        self.spec_in_curve = self.spec_in_plot.plot(pen="#2b6cb0")
        self.spec_err_curve = self.spec_err_plot.plot(pen="#2f855a")
        for plot in (self.ref_plot, self.anti_plot, self.err_plot):
            plot.setYRange(-1.0, 1.0)
            plot.showGrid(x=True, y=True, alpha=0.25)
        for plot in (self.spec_in_plot, self.spec_err_plot):
            plot.setYRange(-120, 0)
            plot.setXRange(0, 3000)
            plot.showGrid(x=True, y=True, alpha=0.25)
        plots.addWidget(self.ref_plot, 0, 0)
        plots.addWidget(self.anti_plot, 0, 1)
        plots.addWidget(self.err_plot, 1, 0)
        plots.addWidget(self.spec_in_plot, 1, 1)
        plots.addWidget(self.spec_err_plot, 2, 0, 1, 2)

        for widget in (
            self.mode_combo,
            self.gain_slider,
            self.delay_spin,
            self.lowpass_check,
            self.lowpass_spin,
            self.mu_spin,
            self.freeze_check,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.sync_settings)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.sync_settings)
            if hasattr(widget, "toggled"):
                widget.toggled.connect(self.sync_settings)
        self.auto_delay_button.clicked.connect(self.estimate_delay_once)
        self.reset_fx_button.clicked.connect(self.reset_fx)
        self.load_secondary_button.clicked.connect(self.load_secondary_path)
        self.save_secondary_button.clicked.connect(self.save_secondary_path)
        self.estimate_secondary_button.clicked.connect(self.estimate_secondary_from_recent)
        self.input_combo.currentIndexChanged.connect(self.update_config_guidance)
        self.output_combo.currentIndexChanged.connect(self.update_config_guidance)
        self.update_config_guidance()

    def _build_simulation_tab(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.tabs.addTab(page, "FxLMS Simulation")

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.sim_reset_button = QtWidgets.QPushButton("Reset Simulation")
        controls.addWidget(self.sim_reset_button)
        self.sim_status = QtWidgets.QLabel("Controlled fan/motor simulation is running.")
        controls.addWidget(self.sim_status)
        controls.addStretch(1)
        self.sim_reset_button.clicked.connect(self.simulation.reset)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)
        self.sim_ref_plot = pg.PlotWidget(title="Synthetic reference noise")
        self.sim_anti_plot = pg.PlotWidget(title="Adaptive anti-noise output")
        self.sim_err_plot = pg.PlotWidget(title="Error at simulated error microphone")
        self.sim_learning_plot = pg.PlotWidget(title="Error level over time")
        self.sim_weight_plot = pg.PlotWidget(title="FxLMS adaptive filter coefficients")
        self.sim_ref_curve = self.sim_ref_plot.plot(pen="#2b6cb0")
        self.sim_anti_curve = self.sim_anti_plot.plot(pen="#c05621")
        self.sim_err_curve = self.sim_err_plot.plot(pen="#2f855a")
        self.sim_learning_curve = self.sim_learning_plot.plot(pen="#805ad5")
        self.sim_weight_curve = self.sim_weight_plot.plot(pen="#dd6b20")
        for plot in (self.sim_ref_plot, self.sim_anti_plot, self.sim_err_plot):
            plot.setYRange(-1.2, 1.2)
            plot.showGrid(x=True, y=True, alpha=0.25)
        self.sim_learning_plot.setYRange(-80, 0)
        self.sim_learning_plot.showGrid(x=True, y=True, alpha=0.25)
        self.sim_weight_plot.showGrid(x=True, y=True, alpha=0.25)
        grid.addWidget(self.sim_ref_plot, 0, 0)
        grid.addWidget(self.sim_anti_plot, 0, 1)
        grid.addWidget(self.sim_err_plot, 1, 0)
        grid.addWidget(self.sim_learning_plot, 1, 1)
        grid.addWidget(self.sim_weight_plot, 2, 0, 1, 2)

    def _build_report_tab(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.tabs.addTab(page, "Report")
        report = QtWidgets.QTextEdit()
        report.setReadOnly(True)
        report.setMarkdown(REPORT_MD)
        layout.addWidget(report)

    def refresh_devices(self) -> None:
        self.input_combo.clear()
        self.output_combo.clear()
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
        for index, device in enumerate(devices):
            name = f"{index}: {device['name']}"
            if device["max_input_channels"] > 0:
                label = f"{name} ({device['max_input_channels']} in)"
                self.input_combo.addItem(label, index)
                if index == default_in:
                    self.input_combo.setCurrentIndex(self.input_combo.count() - 1)
            if device["max_output_channels"] > 0:
                label = f"{name} ({device['max_output_channels']} out)"
                self.output_combo.addItem(label, index)
                if index == default_out:
                    self.output_combo.setCurrentIndex(self.output_combo.count() - 1)
        self.status_label.setText(f"Found {self.input_combo.count()} input devices and {self.output_combo.count()} output devices.")

    def sync_settings(self) -> None:
        sample_rate = int(self.sample_rate_combo.currentData())
        block_size = int(self.block_combo.currentData())
        channels = int(self.channels_combo.currentData())
        gain = self.gain_slider.value() / 1000.0
        delay = int(self.delay_spin.value())
        cutoff = float(self.lowpass_spin.value())
        self.gain_label.setText(f"Gain: {gain:.3f}")
        self.delay_ms_label.setText(f"Delay: {1000.0 * delay / max(sample_rate, 1):.2f} ms")
        with self.settings_lock:
            rebuild_lowpass = self.settings.sample_rate != sample_rate or self.settings.lowpass_hz != cutoff
            self.settings.mode = self.mode_combo.currentText()
            self.settings.sample_rate = sample_rate
            self.settings.block_size = block_size
            self.settings.channels = channels
            self.settings.output_enabled = self.output_enable.isChecked()
            self.settings.muted = self.mute_button.isChecked()
            self.settings.gain = gain
            self.settings.delay_samples = delay
            self.settings.lowpass_enabled = self.lowpass_check.isChecked()
            self.settings.lowpass_hz = cutoff
            self.settings.fx_mu = float(self.mu_spin.value())
            self.settings.fx_freeze = self.freeze_check.isChecked()
            self.fx_filter.mu = self.settings.fx_mu
            self.fx_filter.freeze = self.settings.fx_freeze
            if rebuild_lowpass:
                self.lowpass.configure(sample_rate, cutoff)
        self.update_config_guidance()

    def selected_device_name(self, combo: QtWidgets.QComboBox) -> str:
        text = combo.currentText()
        if ": " in text:
            return text.split(": ", 1)[1]
        return text

    def set_combo_to_data(self, combo: QtWidgets.QComboBox, value: int) -> bool:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
            return True
        return False

    def set_combo_by_keywords(
        self,
        combo: QtWidgets.QComboBox,
        include_any: tuple[str, ...],
        exclude_any: tuple[str, ...] = (),
    ) -> bool:
        for index in range(combo.count()):
            text = combo.itemText(index).lower()
            if include_any and not any(word in text for word in include_any):
                continue
            if exclude_any and any(word in text for word in exclude_any):
                continue
            combo.setCurrentIndex(index)
            return True
        return False

    def selected_input_channel_count(self) -> int:
        device_id = self.input_combo.currentData()
        if device_id is None:
            return 0
        try:
            return int(sd.query_devices(device_id)["max_input_channels"])
        except Exception:
            return 0

    def apply_selected_preset(self) -> None:
        preset = self.preset_combo.currentData()
        self.output_enable.setChecked(False)
        self.mute_button.setChecked(False)
        self.delay_spin.setValue(0)
        self.lowpass_check.setChecked(True)
        self.lowpass_spin.setValue(800)

        if preset == "macbook":
            self.set_combo_by_keywords(self.input_combo, ("macbook", "built-in", "internal"), ("speaker", "output"))
            self.set_combo_by_keywords(self.output_combo, ("macbook", "speaker", "built-in"), ("airpods", "bluetooth"))
            self.sample_rate_combo.setCurrentText("48000")
            self.block_combo.setCurrentText("256")
            self.channels_combo.setCurrentIndex(self.channels_combo.findData(1))
            self.mode_combo.setCurrentText("Simple inversion")
            self.gain_slider.setValue(20)
            message = "Applied MacBook speakers preset: visualization/demo only, output disabled, low gain."
        elif preset == "wired":
            self.set_combo_by_keywords(
                self.input_combo,
                ("usb", "interface", "external", "microphone", "macbook"),
                ("airpods", "bluetooth"),
            )
            output_found = self.set_combo_by_keywords(
                self.output_combo,
                ("headphones", "external", "usb", "interface", "line out"),
                ("airpods", "bluetooth"),
            )
            if not output_found:
                self.set_combo_by_keywords(self.output_combo, ("built-in", "macbook"), ("airpods", "bluetooth"))
            self.sample_rate_combo.setCurrentText("48000")
            self.block_combo.setCurrentText("128")
            self.channels_combo.setCurrentIndex(self.channels_combo.findData(2 if self.selected_input_channel_count() >= 2 else 1))
            self.mode_combo.setCurrentText("Delay compensated inversion")
            self.gain_slider.setValue(20)
            message = "Applied wired headphones preset: best for low-latency experiments if the input has an error mic channel."
        else:
            self.set_combo_by_keywords(self.input_combo, ("airpods", "earpods", "bluetooth"))
            self.set_combo_by_keywords(self.output_combo, ("airpods", "earpods", "bluetooth"))
            self.sample_rate_combo.setCurrentText("48000")
            self.block_combo.setCurrentText("1024")
            self.channels_combo.setCurrentIndex(self.channels_combo.findData(1))
            self.mode_combo.setCurrentText("Simple inversion")
            self.gain_slider.setValue(10)
            message = "Applied Bluetooth AirPods/EarPods preset: visualization only because Bluetooth latency is too high for ANC."

        self.sync_settings()
        self.status_label.setText(message)

    def update_config_guidance(self) -> None:
        if not hasattr(self, "config_label"):
            return
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""
        input_label = self.selected_device_name(self.input_combo) if self.input_combo.count() else ""
        output_label = self.selected_device_name(self.output_combo) if self.output_combo.count() else ""
        input_name = input_label.lower()
        output_name = output_label.lower()
        requested_channels = int(self.channels_combo.currentData()) if self.channels_combo.count() else 1
        notes = []
        if mode == "Experimental FxLMS":
            if requested_channels < 2:
                notes.append("Experimental FxLMS needs 2 input channels: channel 0 reference mic, channel 1 error mic.")
            if self.engine.running and self.active_input_channels < 2:
                notes.append("Current stream has no error mic channel, so the adaptive filter cannot learn.")
            notes.append("For reliable proof, use the FxLMS Simulation tab first.")
        elif mode == "Simple inversion":
            notes.append("This mode can output an inverted, delayed copy, but it is only a phase/delay demonstration.")
        elif mode == "Delay compensated inversion":
            notes.append("Delay estimation needs a second input channel measuring sound at the target point.")
        if "airpods" in input_name or "airpods" in output_name or "bluetooth" in input_name or "bluetooth" in output_name:
            notes.append(
                "AirPods, including AirPods Pro 3, have their own internal ANC, but Bluetooth routing adds too much latency for this Python app to do true ANC."
            )
        if input_label and output_label and input_label != output_label:
            notes.append(
                "Input and output are different physical devices; their clocks are not synchronized, so macOS may add large buffering delay or underflows."
            )
        if "macbook" in output_name and mode != "FxLMS Simulation":
            notes.append("MacBook speakers are not positioned at the cancellation point, so open-air cancellation will be weak or absent.")
        if not notes:
            notes.append("Best real-time setup: wired multichannel input, channel 0 reference mic, channel 1 error mic, low gain, close speaker/mic geometry.")
        self.config_label.setText("Configuration diagnosis:\n" + "\n".join(f"- {note}" for note in notes))

    def on_output_toggled(self, checked: bool) -> None:
        if checked and not self.output_warning_seen:
            QtWidgets.QMessageBox.warning(
                self,
                "Feedback safety warning",
                "Output can create loud feedback if a speaker is near a microphone. "
                "Start at very low gain, use wired devices, and keep speakers away from ears.",
            )
            self.output_warning_seen = True
        self.sync_settings()

    def start_stream(self) -> None:
        self.sync_settings()
        input_device = self.input_combo.currentData()
        output_device = self.output_combo.currentData()
        with self.settings_lock:
            sample_rate = self.settings.sample_rate
            block_size = self.settings.block_size
            channels = self.settings.channels
        try:
            input_info = sd.query_devices(input_device) if input_device is not None else None
            output_info = sd.query_devices(output_device) if output_device is not None else None
            max_in = int(input_info["max_input_channels"]) if input_info is not None else channels
            max_out = int(output_info["max_output_channels"]) if output_info is not None else 1
            input_channels = min(channels, max_in)
            output_channels = max(1, min(channels, max_out))
            self.active_input_channels = input_channels
            self.active_output_channels = output_channels
            self.active_input_name = str(input_info["name"]) if input_info is not None else ""
            self.active_output_name = str(output_info["name"]) if output_info is not None else ""
            if input_channels < channels:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Input channel limit",
                    "The selected input device does not expose two channels, so real-time error-mic "
                    "measurement and experimental FxLMS adaptation will use the available channel only.",
                )
            if input_info is not None and output_info is not None and str(input_info["name"]) != str(output_info["name"]):
                QtWidgets.QMessageBox.warning(
                    self,
                    "High-latency device combination",
                    "You selected different physical devices for input and output. On macOS this can create "
                    "large buffering delay because the devices have independent audio clocks. For real-time "
                    "ANC experiments, use one wired audio interface for both input and output.",
                )
            self.ref_ring = RingBuffer(sample_rate * 5)
            self.anti_ring = RingBuffer(sample_rate * 5)
            self.err_ring = RingBuffer(sample_rate * 5)
            self.recent_ref = RingBuffer(sample_rate)
            self.recent_err = RingBuffer(sample_rate)
            self.ref_ring.clear()
            self.anti_ring.clear()
            self.err_ring.clear()
            self.recent_ref.clear()
            self.recent_err.clear()
            self.delay_line.reset()
            self.engine.start(input_device, output_device, sample_rate, block_size, input_channels, output_channels)
            self.status_label.setText("Audio stream running. Output remains controlled by Output enabled and Mute.")
            self.update_config_guidance()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not start audio stream", str(exc))
            self.status_label.setText(f"Audio start failed: {exc}")

    def stop_stream(self) -> None:
        self.engine.stop()
        self.active_input_channels = 0
        self.active_output_channels = 0
        self.status_label.setText("Audio stream stopped.")
        self.update_config_guidance()

    def emergency_stop(self) -> None:
        self.output_enable.setChecked(False)
        self.mute_button.setChecked(True)
        self.gain.force_zero()
        with self.settings_lock:
            self.settings.output_enabled = False
            self.settings.muted = True
        self.engine.stop()
        self.status_label.setText("Emergency stop: output muted and stream stopped.")

    def process_audio(self, indata: np.ndarray, frames: int) -> np.ndarray:
        with self.settings_lock:
            settings = AudioSettings(**self.settings.__dict__)
        ref = indata[:, 0].astype(np.float32) if indata.shape[1] >= 1 else np.zeros(frames, dtype=np.float32)
        err = indata[:, 1].astype(np.float32) if indata.shape[1] >= 2 else np.zeros(frames, dtype=np.float32)
        has_error_mic = indata.shape[1] >= 2
        mode_note = ""

        if settings.mode == "Simple inversion":
            delayed = self.delay_line.process(ref, settings.delay_samples)
            anti = -delayed
            mode_note = "Simple inversion is producing a delayed inverted copy; acoustic cancellation still depends on placement and latency."
        elif settings.mode == "Delay compensated inversion":
            delayed = self.delay_line.process(ref, settings.delay_samples)
            anti = -delayed
            mode_note = "Delay-compensated inversion is active; use the error mic plot/RMS to judge whether it helps."
        else:
            self.fx_filter.mu = settings.fx_mu
            self.fx_filter.freeze = settings.fx_freeze
            if has_error_mic:
                anti = self.fx_filter.process_with_error(ref, err)
                mode_note = "Experimental FxLMS is adapting from channel 1 error mic."
            else:
                anti = np.zeros(frames, dtype=np.float32)
                mode_note = "Experimental FxLMS is silent because no channel 1 error mic is available; select 2 input channels or use Simulation."

        if settings.lowpass_enabled:
            anti = self.lowpass.process(anti)

        target_gain = settings.gain if settings.output_enabled and not settings.muted else 0.0
        self.gain.set_target(target_gain)
        anti = self.gain.apply(anti)
        anti = np.clip(anti, -0.8, 0.8).astype(np.float32)

        self.ref_ring.push(ref)
        self.anti_ring.push(anti)
        self.err_ring.push(err)
        self.recent_ref.push(ref)
        self.recent_err.push(err)

        input_rms = rms(ref)
        anti_rms = rms(anti)
        error_rms = rms(err)
        reduction = dbfs(input_rms) - dbfs(error_rms) if error_rms > 0 else 0.0
        with self.metrics_lock:
            self.last_metrics.update(
                input_rms=input_rms,
                anti_rms=anti_rms,
                error_rms=error_rms,
                reduction_db=reduction,
                delay=settings.delay_samples,
                mode_note=mode_note,
            )
        return anti

    def estimate_delay_once(self) -> None:
        ref = self.recent_ref.snapshot()
        err = self.recent_err.snapshot()
        if ref.size < 512 or err.size < 512 or rms(err) < 1.0e-5:
            self.status_label.setText("Delay estimate needs a second input channel with measurable error-mic audio.")
            return
        n = min(ref.size, err.size, self.settings.sample_rate // 2)
        ref = ref[-n:] - np.mean(ref[-n:])
        err = err[-n:] - np.mean(err[-n:])
        corr = signal.correlate(err, ref, mode="full", method="fft")
        lags = signal.correlation_lags(err.size, ref.size, mode="full")
        max_lag = min(self.settings.sample_rate // 10, n // 2)
        mask = (lags >= 0) & (lags <= max_lag)
        if not np.any(mask):
            return
        sub_corr = np.abs(corr[mask])
        sub_lags = lags[mask]
        best_index = int(np.argmax(sub_corr))
        best_lag = int(sub_lags[best_index])
        confidence = float(sub_corr[best_index] / (np.mean(sub_corr) + 1.0e-12))
        self.delay_spin.setValue(best_lag)
        with self.metrics_lock:
            self.last_metrics["delay"] = best_lag
            self.last_metrics["confidence"] = confidence
        self.status_label.setText(f"Estimated delay: {best_lag} samples ({confidence:.1f}x peak/mean confidence).")

    def reset_fx(self) -> None:
        self.fx_filter.reset()
        self.status_label.setText("FxLMS adaptive filter reset.")

    def load_secondary_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load secondary path", str(Path.cwd()), "NumPy files (*.npz *.npy)"
        )
        if not path:
            return
        try:
            if path.endswith(".npz"):
                data = np.load(path)
                coeffs = data["secondary_path"]
            else:
                coeffs = np.load(path)
            self.secondary_path = np.asarray(coeffs, dtype=np.float32).reshape(-1)
            self.fx_filter.set_secondary_path(self.secondary_path)
            self.status_label.setText(f"Loaded secondary path with {self.secondary_path.size} taps.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load secondary path", str(exc))

    def save_secondary_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save secondary path", str(Path.cwd() / SECONDARY_PATH_FILE), "NumPy archive (*.npz)"
        )
        if not path:
            return
        np.savez(
            path,
            secondary_path=self.secondary_path.astype(np.float32),
            sample_rate=self.settings.sample_rate,
            block_size=self.settings.block_size,
            timestamp=time.time(),
            note="Teaching estimate of speaker-to-error-microphone secondary path.",
        )
        self.status_label.setText(f"Saved secondary path to {path}.")

    def estimate_secondary_from_recent(self) -> None:
        excitation = self.anti_ring.snapshot()
        measured = self.err_ring.snapshot()
        if excitation.size < 2048 or measured.size < 2048 or rms(excitation) < 1.0e-5 or rms(measured) < 1.0e-5:
            self.status_label.setText("Secondary-path estimate needs recent anti-noise output and error-mic input.")
            return
        n = min(excitation.size, measured.size, self.settings.sample_rate)
        coeffs = SecondaryPathEstimator.identify_nlms(excitation[-n:], measured[-n:], taps=128, mu=0.2)
        self.secondary_path = coeffs
        self.fx_filter.set_secondary_path(coeffs)
        self.status_label.setText("Estimated secondary path from recent output/error buffers. Save it if it looks useful.")

    def run_simulation_tick(self) -> None:
        self.simulation.step(512)
        err = self.simulation.error_history.snapshot()
        if err.size >= 2048:
            recent = rms(err[-2048:])
            early = rms(err[:2048]) if err.size > 4096 else recent
            reduction = dbfs(early) - dbfs(recent)
            self.sim_status.setText(f"Simulation recent error: {dbfs(recent):.1f} dBFS, reduction vs start: {reduction:.1f} dB")

    def update_plots(self) -> None:
        sample_rate = self.settings.sample_rate
        ref = self.ref_ring.snapshot()
        anti = self.anti_ring.snapshot()
        err = self.err_ring.snapshot()
        plot_n = min(2048, ref.size)
        if plot_n > 0:
            t = np.arange(plot_n) / float(sample_rate)
            self.ref_curve.setData(t, ref[-plot_n:])
        plot_n = min(2048, anti.size)
        if plot_n > 0:
            t = np.arange(plot_n) / float(sample_rate)
            self.anti_curve.setData(t, anti[-plot_n:])
        plot_n = min(2048, err.size)
        if plot_n > 0:
            t = np.arange(plot_n) / float(sample_rate)
            self.err_curve.setData(t, err[-plot_n:])

        if ref.size > 64:
            f, m = hann_spectrum(ref, sample_rate)
            self.spec_in_curve.setData(f, m)
        if err.size > 64:
            f, m = hann_spectrum(err, sample_rate)
            self.spec_err_curve.setData(f, m)

        sim_ref = self.simulation.ref_history.snapshot()
        sim_anti = self.simulation.anti_history.snapshot()
        sim_err = self.simulation.error_history.snapshot()
        n = min(2048, sim_ref.size)
        if n > 0:
            x = np.arange(n) / float(self.simulation.sample_rate)
            self.sim_ref_curve.setData(x, sim_ref[-n:])
            self.sim_anti_curve.setData(x, sim_anti[-n:])
            self.sim_err_curve.setData(x, sim_err[-n:])
        if sim_err.size >= 1024:
            window = 1024
            hop = 512
            values = []
            for start in range(0, sim_err.size - window + 1, hop):
                values.append(dbfs(rms(sim_err[start : start + window])))
            self.sim_learning_curve.setData(np.arange(len(values)), np.asarray(values))
        self.sim_weight_curve.setData(self.simulation.controller.w)

        with self.metrics_lock:
            metrics = dict(self.last_metrics)
        self.metrics_label.setText(
            "Input: {0:.4f} RMS / {1:.1f} dBFS\n"
            "Anti-noise: {2:.4f} RMS / {3:.1f} dBFS\n"
            "Error mic: {4:.4f} RMS / {5:.1f} dBFS\n"
            "Estimated input-minus-error difference: {6:.1f} dB\n"
            "Delay: {7} samples, confidence: {8:.1f}x\n"
            "{9}".format(
                metrics["input_rms"],
                dbfs(metrics["input_rms"]),
                metrics["anti_rms"],
                dbfs(metrics["anti_rms"]),
                metrics["error_rms"],
                dbfs(metrics["error_rms"]),
                metrics["reduction_db"],
                metrics["delay"],
                metrics["confidence"],
                metrics["mode_note"],
            )
        )
        self.update_config_guidance()

    def drain_status(self) -> None:
        messages = []
        while True:
            try:
                messages.append(self.engine.status_queue.get_nowait())
            except queue.Empty:
                break
        if messages:
            self.status_label.setText(messages[-1])

    def closeEvent(self, event) -> None:
        self.engine.stop()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
