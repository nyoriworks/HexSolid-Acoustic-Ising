"""
Stanford G-set benchmark using:
  - 100% Classical 1918 Duffing / 1954 Parametron Non-linear Dynamics (NO walls, NO resets, 1 matvec/step)
  - Front/Back Cosine-Skirted L-only Acoustic Link (6.0 - 11.0 kHz band + 5.9-6.0 kHz & 11.0-11.1 kHz skirts)
  - Virtual 48 kHz Acoustic Loopback and Live Physical Audio
Evaluated on B=1 and B=8 across all 7 G-set instances (G1, G11, G14, G22, G55, G67, G70).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
LIVE_DIR = THIS_DIR.parent / "acoustic-ising-live-test"
for p in (THIS_DIR, V1_DIR, LIVE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backends.audio_loopback import AudioLoopbackBackend  # noqa: E402
from backends.base import PhaseTransportBackend  # noqa: E402
from backends.ode_backend import PhaseODEBackend  # noqa: E402
from frequency_map import FrequencyMap, TopOctaveCombMap  # noqa: E402
from gset_loader import get_peak_rss_bytes, load_gset_problem, verify_gset_cut_from_raw_file  # noqa: E402
from live_scale10k_test import DualChirp10kAcousticBackend, fixed_reference_phases, schroeder_pilot_phases  # noqa: E402
from problem import save_json  # noqa: E402

DATA_DIR = THIS_DIR / "data"
RESULTS_DIR = THIS_DIR / "results_duffing"
BENCH_SEED = hashlib.sha256(b"HexSolid-Gset-Duffing-Parametron-v1").digest()


# Remove unused dummy class


class SkirtedAcousticBackend(DualChirp10kAcousticBackend):
    """Adds smooth raised-cosine skirt guard tones before & after the spin band to remove sharp spectral corners."""

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

        p_phases = schroeder_pilot_phases(len(self._pilot_bins))
        spec[self._pilot_bins] = (5.0 * amp) * np.exp(1j * p_phases)
        raw = np.fft.irfft(spec, n=m)
        scaled = np.clip((self.output_gain / 3.8) * raw, -0.85, 0.85)
        cp = self.cyclic_prefix_samples
        return np.concatenate([scaled[-cp:], scaled]) if cp > 0 else scaled


def frame_samples_for_n(n_spins: int) -> int:
    """Scale frame duration linearly with N so bin spacing df stays clean in [6.0, 11.0] kHz."""
    if n_spins <= 800:
        return 12000   # 0.25 s, df = 4.0 Hz
    if n_spins <= 2000:
        return 24000   # 0.50 s, df = 2.0 Hz
    if n_spins <= 5000:
        return 48000   # 1.00 s, df = 1.0 Hz
    return 96000       # 2.00 s, df = 0.5 Hz


def solve_duffing_param_oim(
    problem,
    backend: PhaseTransportBackend,
    rng: np.random.Generator,
    n_steps: int = 100,
    batch_size: int = 1,
    dt: float = 0.35,
    c0_mult: float = 0.65,
):
    """
    1918 Duffing / 1954 Parametron Acoustic-in-the-Loop solver:
      dv/dt = -gamma(t)*v + (p(t) - 0.8)*u - u^3 - c0 * K @ m_hat
      du/dt = v
    where m_hat = Re(z) is measured from the physical / virtual acoustic channel!
    Uses ONLY ONE sparse matvec per step (O(1) extra memory, zero wall resets).
    """
    t0 = time.perf_counter()
    n = problem.n_spins
    backend.reset(n, rng)

    if sparse.issparse(problem.couplings):
        rsq = np.asarray(problem.couplings.multiply(problem.couplings).sum(axis=1)).ravel()
    else:
        rsq = np.sum(problem.couplings * problem.couplings, axis=1)
    denom = float(np.sqrt(np.mean(rsq + problem.field * problem.field)))
    c0 = float(c0_mult / max(denom, 1e-12))
    has_field = bool(np.any(problem.field != 0.0))

    shape = (batch_size, n) if batch_size > 1 else (n,)
    u = rng.normal(0.0, 0.05, size=shape)
    v = np.zeros_like(u)

    for step in range(n_steps):
        pr = float(step) / float(max(n_steps - 1, 1))
        pump = 1.8 * pr  # Parametric excitation pump (Goto 1954)
        damp = 0.45 * (1.0 - 0.75 * pr)
        beta = 1.0 + 6.0 * (pr ** 2)

        # Map state u to acoustic phase angle phi in [0, pi]
        m_ideal = np.tanh(beta * u)
        # Note: arccos maps [-1, 1] to [pi, 0] smoothly
        phi = np.arccos(np.clip(m_ideal, -0.999999, 0.999999))

        # Propagate through physical / virtual acoustic channel!
        z_rx = backend.measure_phasors(phi)
        m_hat = z_rx.real  # In-phase component cos(phi_hat) directly reflects state!

        # Single sparse matvec per step!
        force = -c0 * problem.matvec(m_hat)
        if has_field:
            force = force - c0 * problem.field

        # Classical 2nd-order Duffing-Parametron update
        v = v + (-damp * v + (pump - 0.8) * u - (u * u * u) + force) * dt
        u = u + v * dt

    # Final spins determined by sign of state u
    spins = np.where(u >= 0.0, 1.0, -1.0)
    energies = np.atleast_1d(np.asarray(problem.energy(spins), dtype=np.float64))
    cuts = np.atleast_1d(np.asarray(problem.cut_value(spins), dtype=np.float64))
    best_idx = int(np.argmin(energies))
    best_cut = float(cuts[best_idx])
    mean_cut = float(np.mean(cuts))
    best_e = float(energies[best_idx])
    elapsed = time.perf_counter() - t0
    best_s = spins[best_idx].copy() if batch_size > 1 else spins.copy()
    return {
        "best_cut": best_cut,
        "mean_cut": mean_cut,
        "best_energy": best_e,
        "seconds": elapsed,
        "best_spins": best_s,
    }


def run_worker(
    instance: str,
    solver_type: str,  # "real_audio", "virtual_audio", "ode"
    batch_size: int,
    out_json: Path,
    in_dev: int = 15,
    out_dev: int = 14,
    gain: float = 0.20,
    n_steps: int = 100,
    dt: float = 0.35,
) -> None:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        print("Stanford G-set data not found locally. Fetching from official Stanford repository...")
        try:
            import download_gset
            download_gset.main()
        except Exception as ex:
            print(f"Auto-download failed: {ex}")
    manifest = json.loads(manifest_path.read_text())
    meta = manifest[instance]
    gpath = DATA_DIR / f"{instance}.txt"

    rss_start = get_peak_rss_bytes()
    problem = load_gset_problem(gpath, name=instance)
    n = problem.n_spins
    m_frame = frame_samples_for_n(n)
    bks = float(meta["bks"])

    rng_seed = hashlib.sha256(BENCH_SEED + f":{instance}:{solver_type}:b{batch_size}".encode()).digest()
    words = np.frombuffer(rng_seed, dtype="<u4")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))

    fmap = TopOctaveCombMap(
        n_spins=n,
        sample_rate=48000,
        frame_samples=m_frame,
        f_min_hz=6000.0,
        f_max_hz=11000.0,  # 6.0 - 11.0 kHz band (leaves room for 5.9-6.0 kHz & 11.0-11.1 kHz skirts)
    )

    extra_info: dict[str, object] = {
        "frame_samples": m_frame,
        "n_guard_each_side": 160 if n <= 5000 else 240,
    }

    if solver_type == "real_audio":
        backend = SkirtedAcousticBackend(
            frequency_map=fmap,
            allow_speaker_output=True,
            output_gain=gain,
            cal_averages=2,
            input_device=in_dev,
            output_device=out_dev,
            same_clock=True,
            out_channel="l",
            n_guard_each_side=int(extra_info["n_guard_each_side"]),
        )
    elif solver_type == "virtual_audio":
        backend = AudioLoopbackBackend(
            frequency_map=fmap,
            cyclic_prefix_samples=480,
            snr_db=42.0,
            cal_averages=2,
        )
    elif solver_type == "ode":
        backend = PhaseODEBackend()
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    res = solve_duffing_param_oim(
        problem=problem,
        backend=backend,
        rng=rng,
        n_steps=n_steps,
        batch_size=batch_size,
        dt=dt,
    )

    peak_rss = get_peak_rss_bytes()
    raw_cut = verify_gset_cut_from_raw_file(gpath, res["best_spins"])
    audit_ok = abs(raw_cut - res["best_cut"]) < 1e-8

    bks_ratio = res["best_cut"] / bks
    bks_pct = 100.0 * bks_ratio

    if solver_type == "real_audio":
        extra_info.update({
            "rx_peak": backend.last_rx_peak,
            "cal_phase_std_deg": backend.cal_phase_std_deg,
            "cal_snr_median_db": backend.cal_snr_median_db,
            "out_channel": "l",
        })

    record = {
        "instance": instance,
        "n_spins": n,
        "n_edges": problem.n_edges,
        "bks": bks,
        "solver": solver_type,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "best_cut": res["best_cut"],
        "mean_cut": res["mean_cut"],
        "best_energy": res["best_energy"],
        "bks_percent": bks_pct,
        "seconds": res["seconds"],
        "peak_rss_mib": peak_rss / (1024.0 * 1024.0),
        "audit_passed": audit_ok,
        "extra_info": extra_info,
    }
    save_json(out_json, record)
    print(
        f"[{instance:3s} | {solver_type:13s} B={batch_size}] Cut={res['best_cut']:8.1f} / {bks:.0f} "
        f"({bks_pct:6.2f}% BKS) | time={res['seconds']:7.3f}s | PeakRSS={record['peak_rss_mib']:5.1f} MiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default="G1")
    parser.add_argument("--solver", type=str, default="virtual_audio", choices=("real_audio", "virtual_audio", "ode"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.35)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--in-dev", type=int, default=15)
    parser.add_argument("--out-dev", type=int, default=14)
    parser.add_argument("--gain", type=float, default=0.20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_json) if args.out_json else RESULTS_DIR / f"{args.instance}__{args.solver}__b{args.batch}.json"
    run_worker(
        instance=args.instance,
        solver_type=args.solver,
        batch_size=args.batch,
        out_json=out_path,
        in_dev=args.in_dev,
        out_dev=args.out_dev,
        gain=args.gain,
        n_steps=args.n_steps,
        dt=args.dt,
    )


if __name__ == "__main__":
    main()
