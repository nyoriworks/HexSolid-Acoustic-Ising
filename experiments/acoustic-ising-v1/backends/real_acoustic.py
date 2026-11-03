"""Phase 3 backend: physical CoreAudio (sounddevice) speaker-to-microphone loop with pilot-chirp sync and H(f) calibration."""
from __future__ import annotations

import numpy as np
from .base import PhaseTransportBackend
from frequency_map import FrequencyMap, Musical12ToneMap, TopOctaveCombMap


class RealAcousticBackend(PhaseTransportBackend):
    """
    Physical acoustic OIM transport over macOS CoreAudio (via PortAudio / `sounddevice`).
    Safety guard: refuses to emit speaker audio unless `allow_speaker_output=True` is explicitly passed.
    Uses a linear pilot chirp for round-trip sample latency synchronization followed by cyclic-prefix
    reference multitone frames to calibrate the room transfer function H(f_i) and Schroeder phase offsets.
    """

    def __init__(
        self,
        frequency_map: FrequencyMap | None = None,
        allow_speaker_output: bool = False,
        output_gain: float = 0.20,
        cyclic_prefix_samples: int = 960,
        record_margin_samples: int = 4800,
        cal_averages: int = 4,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        if not allow_speaker_output:
            raise PermissionError(
                "RealAcousticBackend speaker output is disabled by safety guard. "
                "Pass allow_speaker_output=True (--enable-speaker) only when ready to play physical audio."
            )
        if not (0.0 < output_gain <= 0.85):
            raise ValueError("output_gain must be in (0.0, 0.85] to prevent speaker clipping")
        self._explicit_fmap = frequency_map
        self.fmap: FrequencyMap | None = frequency_map
        self.output_gain = float(output_gain)
        self.cyclic_prefix_samples = int(cyclic_prefix_samples)
        self.record_margin_samples = int(record_margin_samples)
        self.cal_averages = int(cal_averages)
        self.input_device = input_device
        self.output_device = output_device
        self._sd = None
        self._pilot: np.ndarray | None = None
        self._latency_samples: int = 0
        self._h_cal: np.ndarray | None = None
        self.calls: int = 0

    @property
    def name(self) -> str:
        fmap_name = self.fmap.name if self.fmap is not None else "AutoMap"
        return f"RealAcousticBackend({fmap_name})"

    def _build_pilot_chirp(self) -> np.ndarray:
        """20 ms Hann-windowed linear chirp (2 kHz -> 9 kHz) outside the 10-20 kHz high-density spin band."""
        assert self.fmap is not None
        fs = self.fmap.sample_rate
        n_pilot = int(0.020 * fs)
        t = np.arange(n_pilot, dtype=np.float64) / fs
        f0, f1 = 2000.0, 9000.0
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t * t) / (n_pilot / fs))
        window = np.hanning(n_pilot)
        return self.output_gain * window * np.cos(phase)

    def _synthesize_single(self, phi_1d: np.ndarray) -> np.ndarray:
        assert self.fmap is not None
        m = self.fmap.frame_samples
        spec = np.zeros(m // 2 + 1, dtype=np.complex128)
        theta = phi_1d + self.fmap.schroeder_phases
        amp = (0.5 * m) / np.sqrt(max(self.fmap.n_spins, 1))
        spec[self.fmap.bins] = amp * (np.cos(theta) + 1j * np.sin(theta))
        raw = np.fft.irfft(spec, n=m)
        peak = max(float(np.max(np.abs(raw))), 1e-12)
        scaled = (self.output_gain / peak) * raw
        cp = self.cyclic_prefix_samples
        return np.concatenate([scaled[-cp:], scaled]) if cp > 0 else scaled

    def _playrec_packet(self, payload: np.ndarray) -> np.ndarray:
        """Transmit [pilot | gap | payload | margin] and align received payload via pilot cross-correlation."""
        assert self.fmap is not None and self._pilot is not None and self._sd is not None
        gap = np.zeros(240, dtype=np.float64)
        tail = np.zeros(self.record_margin_samples, dtype=np.float64)
        tx = np.concatenate([self._pilot, gap, payload, tail]).astype(np.float32)
        rx = self._sd.playrec(
            tx[:, None],
            samplerate=self.fmap.sample_rate,
            channels=1,
            dtype="float32",
            device=(self.input_device, self.output_device),
            blocking=True,
        ).ravel().astype(np.float64)
        # FFT cross-correlation to locate pilot onset accurately against clock/buffer jitter
        n_fft = 1 << int(np.ceil(np.log2(len(rx) + len(self._pilot))))
        corr = np.fft.irfft(np.fft.rfft(rx, n=n_fft) * np.conj(np.fft.rfft(self._pilot, n=n_fft)), n=n_fft)
        onset = int(np.argmax(np.abs(corr[: self.record_margin_samples + len(self._pilot)])))
        self._latency_samples = onset
        start = onset + len(self._pilot) + len(gap) + self.cyclic_prefix_samples
        end = start + self.fmap.frame_samples
        if end > len(rx):
            raise RuntimeError(f"Acoustic round-trip latency ({onset} samples) exceeded record_margin_samples")
        return rx[start:end]

    def calibrate(self) -> np.ndarray:
        assert self.fmap is not None
        zero_payload = self._synthesize_single(np.zeros(self.fmap.n_spins, dtype=np.float64))
        acc = np.zeros(self.fmap.n_spins, dtype=np.complex128)
        for _ in range(self.cal_averages):
            window = self._playrec_packet(zero_payload)
            acc += np.fft.rfft(window)[self.fmap.bins]
        h_cal = acc / float(self.cal_averages)
        if np.any(np.abs(h_cal) < 1e-10):
            raise RuntimeError("Acoustic calibration failed: insufficient SNR on at least one frequency bin")
        self._h_cal = h_cal
        return h_cal

    def reset(self, n_spins: int, rng: np.random.Generator) -> None:
        del rng
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "RealAcousticBackend requires the `sounddevice` package (PortAudio wrapper). "
                "Install `sounddevice` when connecting physical speaker/microphone hardware."
            ) from exc
        self._sd = sd
        if self._explicit_fmap is not None:
            if self._explicit_fmap.n_spins != n_spins:
                raise ValueError(f"FrequencyMap has {self._explicit_fmap.n_spins} spins, solver requested {n_spins}")
            self.fmap = self._explicit_fmap
        else:
            self.fmap = Musical12ToneMap(n_spins) if n_spins <= 12 else TopOctaveCombMap(n_spins)
        self._pilot = self._build_pilot_chirp()
        self.calls = 0
        self.calibrate()

    def measure_phasors(self, phi: np.ndarray) -> np.ndarray:
        if self.fmap is None or self._h_cal is None:
            raise RuntimeError("RealAcousticBackend.reset() must be called before measure_phasors()")
        if phi.shape[-1] != self.fmap.n_spins:
            raise ValueError(f"Expected last axis {self.fmap.n_spins}, got {phi.shape}")
        self.calls += 1
        if phi.ndim == 1:
            rx = self._playrec_packet(self._synthesize_single(phi))
            z_eq = np.fft.rfft(rx)[self.fmap.bins] / self._h_cal
            return z_eq / np.maximum(np.abs(z_eq), 1e-15)
        rows = []
        for b in range(phi.shape[0]):
            rx = self._playrec_packet(self._synthesize_single(phi[b]))
            z_eq = np.fft.rfft(rx)[self.fmap.bins] / self._h_cal
            rows.append(z_eq / np.maximum(np.abs(z_eq), 1e-15))
        return np.stack(rows, axis=0)

