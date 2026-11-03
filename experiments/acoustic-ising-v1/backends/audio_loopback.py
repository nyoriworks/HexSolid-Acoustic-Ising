"""Phase 2 backend: silent in-memory 48 kHz multitone synthesis (iRFFT), virtual room multipath + AWGN, and FFT demodulation."""
from __future__ import annotations

from pathlib import Path
import wave
import numpy as np

from .base import PhaseTransportBackend
from frequency_map import FrequencyMap, Musical12ToneMap, TopOctaveCombMap


class AudioLoopbackBackend(PhaseTransportBackend):
    """
    Synthesizes real 48 kHz time-domain multitone waveforms with Schroeder anti-clipping phases
    and cyclic prefix, propagates them through a simulated multipath room impulse response + AWGN,
    and demodulates unit complex phasors via RFFT and single-pass H(f) calibration.
    Produces ZERO speaker output.
    """

    def __init__(
        self,
        frequency_map: FrequencyMap | None = None,
        cyclic_prefix_samples: int = 480,
        snr_db: float | None = 42.0,
        cal_averages: int = 4,
        use_schroeder_offset: bool = True,
    ) -> None:
        self._explicit_fmap = frequency_map
        self.fmap: FrequencyMap | None = frequency_map
        self.cyclic_prefix_samples = int(cyclic_prefix_samples)
        self.snr_db = snr_db
        self.cal_averages = int(cal_averages)
        self.use_schroeder_offset = bool(use_schroeder_offset)
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._h_room: np.ndarray | None = None
        self._h_room_rfft: np.ndarray | None = None
        self._l_conv: int = 0
        self._h_cal: np.ndarray | None = None
        self.calls: int = 0
        self.last_crest_factor: float = 0.0

    @property
    def name(self) -> str:
        fmap_name = self.fmap.name if self.fmap is not None else "AutoMap"
        return f"AudioLoopbackBackend({fmap_name})"

    def _build_room_impulse(self) -> np.ndarray:
        """Deterministic virtual room impulse response (direct path + 3 delayed reflections < cyclic prefix)."""
        length = min(max(self.cyclic_prefix_samples, 64), 480)
        h = np.zeros(length, dtype=np.float64)
        taps = [(17, 0.88), (93, -0.31), (211, 0.16), (347, -0.08)]
        for delay, gain in taps:
            if delay < length:
                h[delay] += gain
        # Mild low-pass coloration [0.85, 0.15] typical of speaker-mic HF rolloff
        h_filt = np.convolve(h, np.array([0.82, 0.18], dtype=np.float64))[:length]
        return h_filt

    def synthesize_frame(self, phi: np.ndarray) -> np.ndarray:
        """Synthesize M-sample multitone waveform(s) from phase array (..., N) via single iRFFT."""
        assert self.fmap is not None
        m = self.fmap.frame_samples
        spec = np.zeros(phi.shape[:-1] + (m // 2 + 1,), dtype=np.complex128)
        theta = phi + self.fmap.schroeder_phases if self.use_schroeder_offset else phi
        # Scale by (M / 2) / sqrt(N) so each tone has amplitude 1/sqrt(N) and total RMS ~ 1/sqrt(2)
        amp = (0.5 * m) / np.sqrt(max(self.fmap.n_spins, 1))
        spec[..., self.fmap.bins] = amp * (np.cos(theta) + 1j * np.sin(theta))
        waveform = np.fft.irfft(spec, n=m, axis=-1)
        rms = float(np.sqrt(np.mean(waveform * waveform)))
        peak = float(np.max(np.abs(waveform)))
        if rms > 0.0:
            self.last_crest_factor = peak / rms
        return waveform

    def _propagate_channel(self, waveform: np.ndarray) -> np.ndarray:
        """Prepend cyclic prefix, apply linear convolution with room impulse response, add AWGN, strip CP."""
        assert self.fmap is not None and self._h_room_rfft is not None
        m = self.fmap.frame_samples
        cp = self.cyclic_prefix_samples
        tx = np.concatenate([waveform[..., -cp:], waveform], axis=-1) if cp > 0 else waveform
        rx_full = np.fft.irfft(np.fft.rfft(tx, n=self._l_conv, axis=-1) * self._h_room_rfft, n=self._l_conv, axis=-1)
        rx = rx_full[..., cp : cp + m]
        if self.snr_db is not None and np.isfinite(self.snr_db):
            sig_power = float(np.mean(rx * rx))
            noise_std = np.sqrt(max(sig_power, 1e-24) * (10.0 ** (-float(self.snr_db) / 10.0)))
            rx = rx + noise_std * self._rng.standard_normal(rx.shape)
        return rx

    def calibrate(self) -> np.ndarray:
        """Measure combined room + Schroeder reference transfer H_cal(f_i) using phi = 0 pilot frames."""
        assert self.fmap is not None
        zero_phi = np.zeros((self.cal_averages, self.fmap.n_spins), dtype=np.float64)
        tx = self.synthesize_frame(zero_phi)
        rx = self._propagate_channel(tx)
        spec_rx = np.fft.rfft(rx, axis=-1)[..., self.fmap.bins]
        h_cal = np.mean(spec_rx, axis=0)
        if np.any(np.abs(h_cal) < 1e-12):
            raise RuntimeError("Calibration failed: near-zero transfer gain on at least one spin bin")
        self._h_cal = h_cal
        return h_cal

    def reset(self, n_spins: int, rng: np.random.Generator) -> None:
        self._rng = rng
        if self._explicit_fmap is not None:
            if self._explicit_fmap.n_spins != n_spins:
                raise ValueError(f"FrequencyMap has {self._explicit_fmap.n_spins} spins, solver requested {n_spins}")
            self.fmap = self._explicit_fmap
        else:
            self.fmap = Musical12ToneMap(n_spins) if n_spins <= 12 else TopOctaveCombMap(n_spins)
        self._h_room = self._build_room_impulse()
        total_len = self.fmap.frame_samples + self.cyclic_prefix_samples + len(self._h_room) - 1
        self._l_conv = 1 << int(np.ceil(np.log2(max(total_len, 2))))
        self._h_room_rfft = np.fft.rfft(self._h_room, n=self._l_conv)
        self.calls = 0
        self.calibrate()

    def measure_phasors(self, phi: np.ndarray) -> np.ndarray:
        if self.fmap is None or self._h_cal is None:
            raise RuntimeError("AudioLoopbackBackend.reset() must be called before measure_phasors()")
        if phi.shape[-1] != self.fmap.n_spins:
            raise ValueError(f"Expected last axis {self.fmap.n_spins}, got {phi.shape}")
        self.calls += 1
        tx = self.synthesize_frame(phi)
        rx = self._propagate_channel(tx)
        y_bins = np.fft.rfft(rx, axis=-1)[..., self.fmap.bins]
        z_eq = y_bins / self._h_cal
        mag = np.maximum(np.abs(z_eq), 1e-15)
        return z_eq / mag

    def export_trajectory_wav(
        self,
        phi_history: np.ndarray,
        path: Path,
        audible_chord: bool = True,
    ) -> dict[str, object]:
        """
        Render a 1D phase trajectory history (steps, N) to a 16-bit PCM mono WAV file without speaker playback.
        When audible_chord=True and use_schroeder_offset=False, phases {0, pi} map directly to audible chord polarity.
        """
        assert self.fmap is not None
        if phi_history.ndim != 2 or phi_history.shape[1] != self.fmap.n_spins:
            raise ValueError(f"Expected (T, {self.fmap.n_spins}) phi_history, got {phi_history.shape}")
        m = self.fmap.frame_samples
        spec = np.zeros((len(phi_history), m // 2 + 1), dtype=np.complex128)
        theta = phi_history if audible_chord else (phi_history + self.fmap.schroeder_phases)
        amp = (0.5 * m) / np.sqrt(max(self.fmap.n_spins, 1))
        spec[:, self.fmap.bins] = amp * (np.cos(theta) + 1j * np.sin(theta))
        frames = np.fft.irfft(spec, n=m, axis=-1).ravel()
        peak = float(np.max(np.abs(frames)))
        pcm = np.clip((frames / max(peak, 1e-12)) * 0.85 * 32767.0, -32768, 32767).astype("<i2")
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.fmap.sample_rate)
            wf.writeframes(pcm.tobytes())
        return {
            "wav_path": str(path),
            "sample_rate": self.fmap.sample_rate,
            "total_samples": int(len(pcm)),
            "duration_seconds": float(len(pcm)) / float(self.fmap.sample_rate),
        }

