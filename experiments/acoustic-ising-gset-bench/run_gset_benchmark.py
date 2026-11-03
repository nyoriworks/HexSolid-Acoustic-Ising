"""
Stanford G-set Benchmark Suite comparing:
  1. Real Physical Audio OIM (Headphone L-only -> Air -> Mic, shared USB clock)
  2. Virtual 48 kHz Acoustic OIM (AudioLoopbackBackend: multitone iRFFT + room multipath + AWGN + RFFT)
  3. Digital Phase OIM (PhaseODEBackend)
  4. Discrete Simulated Bifurcation (dSB, Goto et al. 2021)
  5. Exact Metropolis Simulated Annealing (SA, chromatic independent-set sweeps)
  6. Classical Steepest 1-Flip Greedy
across Accuracy (% of Best Known Solution), Wall-Clock Time (s), and Standalone Process Peak RSS (MiB).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
LIVE_DIR = THIS_DIR.parent / "acoustic-ising-live-test"
for p in (THIS_DIR, V1_DIR, LIVE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backends.audio_loopback import AudioLoopbackBackend  # noqa: E402
from backends.ode_backend import PhaseODEBackend  # noqa: E402
from frequency_map import TopOctaveCombMap  # noqa: E402
from gset_loader import get_peak_rss_bytes, load_gset_problem, verify_gset_cut_from_raw_file  # noqa: E402
from live_scale10k_test import DualChirp10kAcousticBackend  # noqa: E402
from other_annealers import solve_dsb, solve_sa  # noqa: E402
from problem import save_json, steepest_greedy_baseline  # noqa: E402
from schedule import AnnealingSchedule  # noqa: E402
from solver import IsingAnnealer, derive_problem_rng  # noqa: E402

DATA_DIR = THIS_DIR / "data"
RESULTS_DIR = THIS_DIR / "results"
BENCH_SEED = hashlib.sha256(b"HexSolid-Gset-Public-Benchmark-v1").digest()


def frame_samples_for_n(n_spins: int) -> int:
    """Harmonic-free [6000, 12000) Hz 48 kHz OFDM frame length scaling linearly with N."""
    if n_spins <= 800:
        return 12000   # 0.25 s, df = 4.0 Hz (1,500 bins in 6-12 kHz)
    if n_spins <= 2000:
        return 24000   # 0.50 s, df = 2.0 Hz (3,000 bins in 6-12 kHz)
    if n_spins <= 5000:
        return 48000   # 1.00 s, df = 1.0 Hz (6,000 bins in 6-12 kHz)
    return 96000       # 2.00 s, df = 0.5 Hz (12,000 bins in 6-12 kHz)


def cfl_safe_euler_dt(problem, dt_max: float) -> float:
    """Ensure explicit Euler step size satisfies linear stability on high-degree graphs."""
    v = np.ones(problem.n_spins, dtype=np.float64) / np.sqrt(max(problem.n_spins, 1))
    for _ in range(24):
        v = np.asarray(problem.couplings @ v, dtype=np.float64)
        nrm = float(np.linalg.norm(v))
        if nrm <= 1e-15:
            break
        v /= nrm
    spec_ratio = max(float(np.linalg.norm(np.asarray(problem.couplings @ v, dtype=np.float64))) / problem.j_scale, 2.0)
    return round(min(dt_max, 1.35 / (1.5 * spec_ratio)), 4)


ALL_SOLVERS = (
    # Single-trajectory 90-step matched comparison (direct 1-to-1 match with physical audio!)
    "real_acoustic_1",
    "virtual_acoustic_1",
    "phase_ode_1",
    "dsb_1",
    "sa_1",
    # Multi-trajectory standard digital/virtual comparison (32 starts / 8 starts for virtual audio)
    "virtual_acoustic_b8",
    "phase_ode_b32",
    "dsb_b32",
    "sa_b32",
    "greedy_b32",
)


def run_single_worker(
    gname: str,
    solver_key: str,
    out_json: Path,
    in_dev: int,
    out_dev: int,
    gain: float,
    steps_1: int,
    dt_1: float,
) -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    meta = manifest[gname]
    gpath = DATA_DIR / f"{gname}.txt"

    rss_before_load = get_peak_rss_bytes()
    problem = load_gset_problem(gpath, name=gname)
    rss_after_load = get_peak_rss_bytes()

    n = problem.n_spins
    m_frame = frame_samples_for_n(n)
    dt_eff_1 = cfl_safe_euler_dt(problem, dt_1)
    steps_eff_1 = 110 if n <= 800 else (95 if n <= 2000 else steps_1)
    rng = derive_problem_rng(BENCH_SEED, gname, extra_tag="gset_bench_v1")

    extra_hw: dict[str, object] = {}

    if solver_key == "real_acoustic_1":
        fmap = TopOctaveCombMap(n_spins=n, sample_rate=48000, frame_samples=m_frame, f_min_hz=6000.0, f_max_hz=12000.0)
        backend = DualChirp10kAcousticBackend(
            frequency_map=fmap,
            allow_speaker_output=True,
            output_gain=gain,
            cal_averages=2,
            input_device=in_dev,
            output_device=out_dev,
            same_clock=True,
            out_channel="l",
        )
        sched = AnnealingSchedule(n_steps=steps_eff_1, dt=dt_eff_1)
        res = IsingAnnealer(sched, backend, "euler").solve(problem, rng, batch_size=1, prefix_sizes=(1,))
        best_cut = float(res.best_cut)
        mean_cut = float(res.mean_cut)
        best_energy = float(res.best_energy)
        seconds = float(res.seconds)
        batch_size = 1
        n_steps = steps_eff_1
        best_spins = res.best_spins
        extra_hw = {
            "frame_samples": m_frame,
            "frame_duration_s": fmap.frame_duration_seconds,
            "dt": dt_eff_1,
            "rx_peak": backend.last_rx_peak,
            "cal_phase_std_deg": backend.cal_phase_std_deg,
            "cal_snr_median_db": backend.cal_snr_median_db,
            "out_channel": "l",
        }

    elif solver_key == "virtual_acoustic_1":
        fmap = TopOctaveCombMap(n_spins=n, sample_rate=48000, frame_samples=m_frame, f_min_hz=6000.0, f_max_hz=12000.0)
        backend_lb = AudioLoopbackBackend(frequency_map=fmap, cyclic_prefix_samples=480, snr_db=42.0, cal_averages=2)
        sched = AnnealingSchedule(n_steps=steps_eff_1, dt=dt_eff_1)
        res = IsingAnnealer(sched, backend_lb, "euler").solve(problem, rng, batch_size=1, prefix_sizes=(1,))
        best_cut = float(res.best_cut)
        mean_cut = float(res.mean_cut)
        best_energy = float(res.best_energy)
        seconds = float(res.seconds)
        batch_size = 1
        n_steps = steps_eff_1
        best_spins = res.best_spins
        extra_hw = {"frame_samples": m_frame, "snr_db": 42.0, "dt": dt_eff_1}

    elif solver_key == "phase_ode_1":
        sched = AnnealingSchedule(n_steps=steps_eff_1, dt=dt_eff_1)
        res = IsingAnnealer(sched, PhaseODEBackend(), "euler").solve(problem, rng, batch_size=1, prefix_sizes=(1,))
        best_cut = float(res.best_cut)
        mean_cut = float(res.mean_cut)
        best_energy = float(res.best_energy)
        seconds = float(res.seconds)
        batch_size = 1
        n_steps = steps_eff_1
        best_spins = res.best_spins

    elif solver_key == "dsb_1":
        r = solve_dsb(problem, rng, n_steps=steps_eff_1, batch_size=1, dt=0.50)
        best_cut, mean_cut, best_energy, seconds = r.best_cut, r.mean_cut, r.best_energy, r.seconds
        batch_size, n_steps, best_spins = 1, steps_eff_1, r.best_spins

    elif solver_key == "sa_1":
        r = solve_sa(problem, rng, n_sweeps=steps_eff_1, batch_size=1)
        best_cut, mean_cut, best_energy, seconds = r.best_cut, r.mean_cut, r.best_energy, r.seconds
        batch_size, n_steps, best_spins = 1, steps_eff_1, r.best_spins

    elif solver_key == "virtual_acoustic_b8":
        fmap = TopOctaveCombMap(n_spins=n, sample_rate=48000, frame_samples=m_frame, f_min_hz=6000.0, f_max_hz=12000.0)
        backend_lb = AudioLoopbackBackend(frequency_map=fmap, cyclic_prefix_samples=480, snr_db=42.0, cal_averages=2)
        sched = AnnealingSchedule(n_steps=400, dt=cfl_safe_euler_dt(problem, 0.20))
        b_sz = 8 if n <= 5000 else 4
        res = IsingAnnealer(sched, backend_lb, "euler").solve(problem, rng, batch_size=b_sz, prefix_sizes=(b_sz,))
        best_cut = float(res.best_cut)
        mean_cut = float(res.mean_cut)
        best_energy = float(res.best_energy)
        seconds = float(res.seconds)
        batch_size = b_sz
        n_steps = 400
        best_spins = res.best_spins
        extra_hw = {"frame_samples": m_frame, "snr_db": 42.0}

    elif solver_key == "phase_ode_b32":
        sched = AnnealingSchedule(n_steps=800, dt=0.10)
        res = IsingAnnealer(sched, PhaseODEBackend(), "heun").solve(problem, rng, batch_size=32, prefix_sizes=(32,))
        best_cut = float(res.best_cut)
        mean_cut = float(res.mean_cut)
        best_energy = float(res.best_energy)
        seconds = float(res.seconds)
        batch_size = 32
        n_steps = 800
        best_spins = res.best_spins

    elif solver_key == "dsb_b32":
        r = solve_dsb(problem, rng, n_steps=800, batch_size=32, dt=0.50)
        best_cut, mean_cut, best_energy, seconds = r.best_cut, r.mean_cut, r.best_energy, r.seconds
        batch_size, n_steps, best_spins = 32, 800, r.best_spins

    elif solver_key == "sa_b32":
        r = solve_sa(problem, rng, n_sweeps=800, batch_size=32)
        best_cut, mean_cut, best_energy, seconds = r.best_cut, r.mean_cut, r.best_energy, r.seconds
        batch_size, n_steps, best_spins = 32, 800, r.best_spins

    elif solver_key == "greedy_b32":
        g_res = steepest_greedy_baseline(problem, n_starts=32)
        best_cut = float(g_res["best_cut"])
        mean_cut = float(g_res["mean_cut"])
        best_energy = float(g_res["best_energy"])
        seconds = float(g_res["seconds"])
        batch_size = 32
        n_steps = int(round(float(g_res["mean_steps"])))
        from problem import bitstring_to_spins

        best_spins = bitstring_to_spins(str(g_res["best_bitstring"]))
    else:
        raise ValueError(f"Unknown solver_key: {solver_key}")

    peak_rss_bytes = get_peak_rss_bytes()
    raw_verified_cut = verify_gset_cut_from_raw_file(gpath, best_spins)
    audit_passed = abs(raw_verified_cut - best_cut) < 1e-8
    if not audit_passed:
        raise RuntimeError(f"Audit mismatch on {gname}/{solver_key}: solver={best_cut}, raw={raw_verified_cut}")

    bks = float(meta["bks"])
    bks_ratio = best_cut / bks

    payload = {
        "instance": gname,
        "n_spins": n,
        "n_edges": problem.n_edges,
        "weights": meta["weights"],
        "topology": meta["topology"],
        "bks": bks,
        "solver": solver_key,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "best_cut": best_cut,
        "mean_cut": mean_cut,
        "best_energy": best_energy,
        "bks_ratio": bks_ratio,
        "bks_percent": 100.0 * bks_ratio,
        "seconds": seconds,
        "peak_rss_mib": peak_rss_bytes / (1024.0 * 1024.0),
        "solver_rss_delta_mib": max(0, peak_rss_bytes - rss_after_load) / (1024.0 * 1024.0),
        "matrix_storage_kib": problem.coupling_storage_bytes / 1024.0,
        "raw_file_audit_passed": audit_passed,
        "extra_hw": extra_hw,
    }
    save_json(out_json, payload)
    print(
        f"[{gname:3s} | {solver_key:19s}] Cut={best_cut:8.1f} / {bks:.0f} ({100.0*bks_ratio:6.2f}% BKS) | "
        f"time={seconds:7.3f}s | PeakRSS={payload['peak_rss_mib']:6.1f} MiB (delta={payload['solver_rss_delta_mib']:5.1f} MiB)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=str, default="G1")
    parser.add_argument("--solver", type=str, default="phase_ode_b32")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--instances", nargs="+", default=["G1", "G11", "G14", "G22", "G55", "G67", "G70"])
    parser.add_argument("--solvers", nargs="+", default=[s for s in ALL_SOLVERS if s != "real_acoustic_1"])
    parser.add_argument("--in-dev", type=int, default=15)
    parser.add_argument("--out-dev", type=int, default=14)
    parser.add_argument("--gain", type=float, default=0.20)
    parser.add_argument("--steps-1", type=int, default=90)
    parser.add_argument("--dt-1", type=float, default=0.46)
    args = parser.parse_args()

    if args.worker:
        run_single_worker(
            gname=args.instance,
            solver_key=args.solver,
            out_json=Path(args.out_json),
            in_dev=args.in_dev,
            out_dev=args.out_dev,
            gain=args.gain,
            steps_1=args.steps_1,
            dt_1=args.dt_1,
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
        env[k] = "1"

    all_records: list[dict[str, object]] = []
    for gname in args.instances:
        for skey in args.solvers:
            out_file = RESULTS_DIR / f"{gname}__{skey}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--instance",
                gname,
                "--solver",
                skey,
                "--out-json",
                str(out_file),
                "--in-dev",
                str(args.in_dev),
                "--out-dev",
                str(args.out_dev),
                "--gain",
                str(args.gain),
                "--steps-1",
                str(args.steps_1),
                "--dt-1",
                str(args.dt_1),
            ]
            subprocess.run(cmd, env=env, check=True)
            all_records.append(json.loads(out_file.read_text()))

    # Merge any existing results in RESULTS_DIR into summary.json
    merged: dict[str, dict[str, object]] = {}
    for p in sorted(RESULTS_DIR.glob("*__*.json")):
        rec = json.loads(p.read_text())
        merged[f"{rec['instance']}__{rec['solver']}"] = rec
    save_json(RESULTS_DIR / "summary.json", list(merged.values()))
    print(f"\nSaved {len(merged)} total benchmark records to {RESULTS_DIR / 'summary.json'}.")


if __name__ == "__main__":
    main()
