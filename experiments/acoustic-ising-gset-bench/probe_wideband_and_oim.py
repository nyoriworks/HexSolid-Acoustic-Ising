"""Quick verification of:
1. Wideband odd-bin + raised-cosine band-edge skirt on live L-only audio (10,000 spins).
2. Classical 1984 Hopfield-gain + degree-normalized 1-matvec Phase OIM on G-set (PhaseODE + VirtualAudio).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
import sounddevice as sd

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
LIVE_DIR = THIS_DIR.parent / "acoustic-ising-live-test"
for p in (THIS_DIR, V1_DIR, LIVE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backends.audio_loopback import AudioLoopbackBackend  # noqa: E402
from backends.base import PhaseTransportBackend  # noqa: E402
from backends.ode_backend import PhaseODEBackend  # noqa: E402
from frequency_map import FrequencyMap, schroeder_phase_offsets  # noqa: E402
from gset_loader import load_gset_problem  # noqa: E402
from live_scale10k_test import DualChirp10kAcousticBackend, fixed_reference_phases  # noqa: E402


def build_odd_bin_wideband_map(
    n_spins: int = 10000,
    sample_rate: int = 48000,
    frame_samples: int = 76800,
    f_min_hz: float = 2200.0,
) -> FrequencyMap:
    """Place all N spins strictly on ODD FFT bins (2m+1) so all x^2 2nd-harmonic & IMD products land on EVEN bins."""
    df = sample_rate / frame_samples
    first_bin = int(math.ceil(f_min_hz / df))
    if first_bin % 2 == 0:
        first_bin += 1
    bins = first_bin + 2 * np.arange(n_spins, dtype=np.int64)
    freqs = bins.astype(np.float64) * df
    return FrequencyMap(
        name="OddBinWidebandMap",
        n_spins=n_spins,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        bins=bins,
        frequencies_hz=freqs,
        schroeder_phases=schroeder_phase_offsets(n_spins),
        harmonic_free=False,  # Multi-octave via odd-bin x^2 orthogonality instead of single-octave bound
        labels=tuple(f"odd{int(b)}" for b in bins),
    )


class SkirtedWidebandAcousticBackend(DualChirp10kAcousticBackend):
    """Adds smooth raised-cosine guard tones before & after the spin band to remove sharp spectral corners."""

    def __init__(self, *args, n_guard_each_side: int = 160, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_guard = int(n_guard_each_side)
        step = int(self.fmap.bins[1] - self.fmap.bins[0]) if self.fmap.n_spins > 1 else 2
        lo_g = self.fmap.bins[0] - step * np.arange(self.n_guard, 0, -1, dtype=np.int64)
        hi_g = self.fmap.bins[-1] + step * np.arange(1, self.n_guard + 1, dtype=np.int64)
        lo_g = lo_g[lo_g > 120]
        hi_g = hi_g[hi_g < (self.fmap.frame_samples // 2 - 120)]
        # Exclude pilot bins
        pset = set(int(b) for b in self._pilot_bins)
        lo_g = np.array([b for b in lo_g if int(b) not in pset], dtype=np.int64)
        hi_g = np.array([b for b in hi_g if int(b) not in pset], dtype=np.int64)
        self._guard_lo = lo_g
        self._guard_hi = hi_g
        # Smooth cosine-squared roll-off weights (0 -> 1 on low side, 1 -> 0 on high side)
        self._w_lo = np.sin(0.5 * np.pi * np.linspace(0.05, 0.95, len(lo_g))) ** 2 if len(lo_g) else np.empty(0)
        self._w_hi = np.cos(0.5 * np.pi * np.linspace(0.05, 0.95, len(hi_g))) ** 2 if len(hi_g) else np.empty(0)
        self._guard_phases_lo = fixed_reference_phases(len(lo_g))
        self._guard_phases_hi = fixed_reference_phases(len(hi_g))

    def _synthesize_symbol(self, phi_1d: np.ndarray) -> np.ndarray:
        m = self.fmap.frame_samples
        spec = np.zeros(m // 2 + 1, dtype=np.complex128)
        theta = phi_1d + self._ref_phases
        n_eff = self.fmap.n_spins + 0.5 * (len(self._guard_lo) + len(self._guard_hi)) + len(self._pilot_bins)
        amp = (0.5 * m) / np.sqrt(max(n_eff, 1.0))
        spec[self.fmap.bins] = amp * (np.cos(theta) + 1j * np.sin(theta))
        if len(self._guard_lo):
            spec[self._guard_lo] = (amp * self._w_lo) * np.exp(1j * self._guard_phases_lo)
        if len(self._guard_hi):
            spec[self._guard_hi] = (amp * self._w_hi) * np.exp(1j * self._guard_phases_hi)
        from live_scale10k_test import schroeder_pilot_phases

        p_phases = schroeder_pilot_phases(len(self._pilot_bins))
        spec[self._pilot_bins] = (5.0 * amp) * np.exp(1j * p_phases)
        raw = np.fft.irfft(spec, n=m)
        scaled = np.clip((self.output_gain / 3.8) * raw, -0.85, 0.85)
        cp = self.cyclic_prefix_samples
        return np.concatenate([scaled[-cp:], scaled]) if cp > 0 else scaled


def solve_hopfield_degree_phase_oim(
    problem,
    backend: PhaseTransportBackend,
    rng: np.random.Generator,
    n_steps: int = 400,
    batch_size: int = 32,
    dt: float = 0.22,
):
    """
    100% Classical Public-Domain (Hopfield 1984 tanh saturation + 1980s degree-normalized local field + 1-variable phase OIM).
    Uses ONLY 1 sparse matvec per step (2x-4x faster than 2-matvec Heun OIM) and 0 extra matrix memory.
    """
    t0 = time.perf_counter()
    n = problem.n_spins
    backend.reset(n, rng)
    if sparse.issparse(problem.couplings):
        row_sq = np.asarray(problem.couplings.multiply(problem.couplings).sum(axis=1)).ravel()
    else:
        row_sq = np.sum(problem.couplings * problem.couplings, axis=1)
    deg_norm = np.sqrt(np.maximum(row_sq + problem.field * problem.field, 1e-12))
    inv_deg = 1.0 / deg_norm

    shape = (batch_size, n) if batch_size > 1 else (n,)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    best_e = float("inf")
    best_cut = -float("inf")
    has_field = bool(np.any(problem.field != 0.0))

    for step in range(n_steps):
        u = float(step) / float(max(n_steps - 1, 1))
        z = backend.measure_phasors(phi)
        c = z.real
        s = z.imag
        # 1984 Hopfield continuous neuron high-gain annealing tanh(gamma * cos(phi))
        gamma = 1.2 + 8.0 * (u ** 1.8)
        m_soft = np.tanh(gamma * c)
        # ONLY ONE sparse matvec!
        h_loc = problem.matvec(m_soft)
        if has_field:
            h_loc = h_loc + problem.field
        g_norm = h_loc * inv_deg  # O(1) normalized local field for every vertex regardless of degree!

        kc = 0.9 + 0.9 * u
        osc = math.sin(math.pi * 5.0 * min(u / 0.92, 1.0)) ** 2
        ks = (0.04 + 0.55 * (u ** 1.6)) + 0.55 * math.sin(math.pi * min(u / 0.96, 1.0)) * osc
        # Smooth PLL phase-detector window so locked spins near 0/pi can still flip when g_norm opposes cos(phi)
        pd_sin = s / np.sqrt(s * s + 0.12 * 0.12)
        sigma = 0.22 * ((1.0 - u) ** 1.6) if u < 0.96 else 0.0
        dw = (sigma * math.sqrt(dt)) * rng.standard_normal(shape) if sigma > 0.0 else 0.0

        dphi = kc * (pd_sin * g_norm) - (2.0 * ks) * (s * c)
        phi = np.mod(phi + dt * dphi + dw, 2.0 * math.pi)

        if step >= int(0.75 * n_steps) and (step % 25 == 0 or step == n_steps - 1):
            spins = np.where(np.cos(phi) >= 0.0, 1.0, -1.0)
            es = np.atleast_1d(np.asarray(problem.energy(spins), dtype=np.float64))
            idx = int(np.argmin(es))
            if float(es[idx]) < best_e:
                best_e = float(es[idx])
                best_cut = float(0.5 * (problem.total_edge_weight - best_e))

    return best_cut, time.perf_counter() - t0


def main() -> None:
    # 1. Test acoustic band tapering & wideband odd-bin on live L-only audio!
    rng_bits = np.random.default_rng(42)
    true_bits = rng_bits.integers(0, 2, size=10000)
    phi_tx = np.where(true_bits == 0, 0.0, np.pi)

    # (A) [6.0, 11.0) kHz + smooth guard skirts on 5.84-6.0 kHz and 11.0-11.16 kHz (M=96000, 2.0s)
    from frequency_map import TopOctaveCombMap

    fmap_6_12 = TopOctaveCombMap(10000, 48000, 96000, 6000.0, 12000.0)
    b_skirt = SkirtedWidebandAcousticBackend(
        fmap_6_12, allow_speaker_output=True, output_gain=0.20, cal_averages=3,
        input_device=15, output_device=14, same_clock=True, out_channel="l", n_guard_each_side=200,
    )
    b_skirt.reset(10000, np.random.default_rng(0))
    z_s = b_skirt.measure_phasors(phi_tx)
    acc_s = float(np.mean((z_s.real < 0).astype(int) == true_bits))
    std_s = float(np.std(np.rad2deg(np.angle(z_s * np.exp(-1j * phi_tx)))))
    print(f"[Live L-only Skirted 6-11kHz M=96000 (2.0s)] BPSK={acc_s*100:.2f}%, phase_std={std_s:.2f} deg, cal_std={b_skirt.cal_phase_std_deg:.2f} deg")

    # (B) Wideband Odd-Bin [2.4, 14.9) kHz + guard skirts (M=76800, 1.60s -> 20% faster!)
    fmap_wb = build_odd_bin_wideband_map(10000, 48000, 76800, 2400.0)
    b_wb = SkirtedWidebandAcousticBackend(
        fmap_wb, allow_speaker_output=True, output_gain=0.20, cal_averages=3,
        input_device=15, output_device=14, same_clock=True, out_channel="l", n_guard_each_side=120,
    )
    b_wb.reset(10000, np.random.default_rng(0))
    z_w = b_wb.measure_phasors(phi_tx)
    acc_w = float(np.mean((z_w.real < 0).astype(int) == true_bits))
    std_w = float(np.std(np.rad2deg(np.angle(z_w * np.exp(-1j * phi_tx)))))
    print(f"[Live L-only OddBin Wideband 2.4-14.9kHz M=76800 (1.6s)] BPSK={acc_w*100:.2f}%, phase_std={std_w:.2f} deg, cal_std={b_wb.cal_phase_std_deg:.2f} deg")

    # 2. Test Hopfield+Degree 1-matvec Phase OIM on G1..G70 (B=32 and B=1)!
    manifest = json.loads((THIS_DIR / "data" / "manifest.json").read_text())
    seed = hashlib.sha256(b"HexSolid-Gset-Public-Benchmark-v1").digest()
    from solver import derive_problem_rng

    print("\n=== Classical Hopfield(1984)+Degree 1-Matvec Phase OIM on G-set ===")
    for g in ("G1", "G11", "G14", "G22", "G55", "G67", "G70"):
        prob = load_gset_problem(THIS_DIR / "data" / f"{g}.txt", g)
        bks = float(manifest[g]["bks"])
        cut1, t1 = solve_hopfield_degree_phase_oim(prob, PhaseODEBackend(), derive_problem_rng(seed, g, "hop1"), n_steps=110, batch_size=1)
        cut32, t32 = solve_hopfield_degree_phase_oim(prob, PhaseODEBackend(), derive_problem_rng(seed, g, "hop32"), n_steps=500, batch_size=32)
        print(f"  {g:3s} (BKS={bks:6.0f}): 1-traj(110st)={cut1:7.0f} ({100*cut1/bks:6.2f}%, {t1:.3f}s) | 32-traj(500st)={cut32:7.0f} ({100*cut32/bks:6.2f}%, {t32:.3f}s)")


if __name__ == "__main__":
    main()

