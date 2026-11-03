"""Frequency-bin mappings for acoustic OIM: 12-tone audible preset and 10-20 kHz harmonic-free comb."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NOTE_NAMES_12 = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def schroeder_phase_offsets(n_spins: int) -> np.ndarray:
    """Quadratic Schroeder phases theta_0[k] = pi * k * (k + 1) / N (mod 2*pi) to minimize crest factor."""
    idx = np.arange(n_spins, dtype=np.float64)
    return np.mod(np.pi * idx * (idx + 1.0) / max(n_spins, 1), 2.0 * np.pi)


@dataclass(frozen=True)
class FrequencyMap:
    """Exact FFT-bin-centered multitone frequency allocation."""

    name: str
    n_spins: int
    sample_rate: int
    frame_samples: int
    bins: np.ndarray
    frequencies_hz: np.ndarray
    schroeder_phases: np.ndarray
    harmonic_free: bool
    labels: tuple[str, ...]

    @property
    def bin_hz(self) -> float:
        return float(self.sample_rate) / float(self.frame_samples)

    @property
    def frame_duration_seconds(self) -> float:
        return float(self.frame_samples) / float(self.sample_rate)

    def validate(self) -> dict[str, object]:
        nyquist_bin = self.frame_samples // 2
        strictly_increasing = bool(np.all(np.diff(self.bins) > 0)) if self.n_spins > 1 else True
        in_range = bool(np.all((self.bins > 0) & (self.bins < nyquist_bin)))
        exact_bin_centers = bool(
            np.max(np.abs(self.frequencies_hz - self.bins * self.bin_hz)) < 1e-12
        )
        second_harmonic_clear = bool(2.0 * float(self.frequencies_hz[0]) > float(self.frequencies_hz[-1]))
        min_bin_gap = int(np.min(np.diff(self.bins))) if self.n_spins > 1 else nyquist_bin
        ok = strictly_increasing and in_range and exact_bin_centers and (
            second_harmonic_clear or not self.harmonic_free
        )
        if not ok:
            raise ValueError(f"Invalid FrequencyMap {self.name}: bins/harmonics check failed")
        return {
            "name": self.name,
            "n_spins": self.n_spins,
            "sample_rate": self.sample_rate,
            "frame_samples": self.frame_samples,
            "bin_hz": self.bin_hz,
            "f_min_hz": float(self.frequencies_hz[0]),
            "f_max_hz": float(self.frequencies_hz[-1]),
            "min_bin_gap": min_bin_gap,
            "strictly_increasing_bins": strictly_increasing,
            "exact_bin_centers": exact_bin_centers,
            "second_harmonic_above_band": second_harmonic_clear,
            "passed": ok,
        }


def Musical12ToneMap(
    n_spins: int = 12,
    sample_rate: int = 48000,
    frame_samples: int = 4800,
    base_freq_hz: float = 523.2511306011972,  # C5
    base_octave: int = 5,
    allow_multi_octave: bool = False,
) -> FrequencyMap:
    """Map spins to 12-TET chromatic notes (default C5..B5 in one harmonic-free octave), snapped to FFT bins."""
    if n_spins < 1:
        raise ValueError("n_spins must be >= 1")
    if n_spins > 12 and not allow_multi_octave:
        raise ValueError("Musical12ToneMap strict harmonic-free mode supports n_spins <= 12; use TopOctaveCombMap for N > 12")
    semitones = np.arange(n_spins, dtype=np.float64)
    targets = base_freq_hz * (2.0 ** (semitones / 12.0))
    df = sample_rate / frame_samples
    bins = np.rint(targets / df).astype(np.int64)
    freqs = bins.astype(np.float64) * df
    labels = tuple(f"{NOTE_NAMES_12[i % 12]}{base_octave + (i // 12)}" for i in range(n_spins))
    fmap = FrequencyMap(
        name="Musical12ToneMap",
        n_spins=n_spins,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        bins=bins,
        frequencies_hz=freqs,
        schroeder_phases=schroeder_phase_offsets(n_spins),
        harmonic_free=(n_spins <= 12),
        labels=labels,
    )
    fmap.validate()
    return fmap


def TopOctaveCombMap(
    n_spins: int,
    sample_rate: int = 48000,
    frame_samples: int | None = None,
    f_min_hz: float = 10000.0,
    f_max_hz: float = 20000.0,
) -> FrequencyMap:
    """
    Map up to 2,000+ spins strictly onto integer FFT bins inside [10 kHz, 20 kHz).
    Because 2 * 10 kHz = 20 kHz >= f_max_hz, all 2nd and higher speaker harmonics fall outside the band.
    """
    if n_spins < 1:
        raise ValueError("n_spins must be >= 1")
    if frame_samples is None:
        if n_spins <= 500:
            frame_samples = 4800   # 0.1 s, df = 10 Hz, 1000 bins in [10k, 20k) -> >= 2-bin spacing
        elif n_spins <= 1000:
            frame_samples = 9600   # 0.2 s, df = 5 Hz, 2000 bins in [10k, 20k) -> >= 2-bin spacing
        else:
            frame_samples = 19200  # 0.4 s, df = 2.5 Hz, 4000 bins in [10k, 20k)

    df = sample_rate / frame_samples
    bin_start = int(np.ceil(f_min_hz / df))
    bin_stop = int(np.ceil(f_max_hz / df)) - 1  # strictly < f_max_hz
    available = bin_stop - bin_start + 1
    if available < n_spins:
        raise ValueError(
            f"Band [{f_min_hz}, {f_max_hz}) Hz at M={frame_samples} has {available} bins < n_spins={n_spins}"
        )
    stride = available // n_spins
    bins = bin_start + np.arange(n_spins, dtype=np.int64) * stride
    # Ensure strictly < f_max_hz
    assert int(bins[-1]) * df < f_max_hz
    freqs = bins.astype(np.float64) * df
    labels = tuple(f"bin{int(b)}_{freqs[i]:.1f}Hz" for i, b in enumerate(bins))
    fmap = FrequencyMap(
        name="TopOctaveCombMap",
        n_spins=n_spins,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        bins=bins,
        frequencies_hz=freqs,
        schroeder_phases=schroeder_phase_offsets(n_spins),
        harmonic_free=True,
        labels=labels,
    )
    fmap.validate()
    return fmap

