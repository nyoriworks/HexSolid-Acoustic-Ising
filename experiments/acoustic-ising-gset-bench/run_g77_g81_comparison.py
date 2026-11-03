"""
Direct Comparison for G77 (N=14,000) and G81 (N=20,000):
  1. New Duffing-Parametron Real Audio (L-only, front/back cosine skirts)
  2. New Duffing-Parametron Virtual Audio (48 kHz OFDM Loopback)
  3. Digital Phase OIM (PhaseODEBackend)
  4. Simulated Bifurcation (dSB)
  5. Simulated Annealing (SA)
Single trajectory (batch=1) comparison across Accuracy (% BKS), Wall-Clock Seconds, and Peak RSS (MiB).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
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
from backends.ode_backend import PhaseODEBackend  # noqa: E402
from duffing_bench import SkirtedAcousticBackend, solve_duffing_param_oim  # noqa: E402
from frequency_map import TopOctaveCombMap  # noqa: E402
from gset_loader import get_peak_rss_bytes, load_gset_problem, verify_gset_cut_from_raw_file  # noqa: E402
from other_annealers import solve_dsb, solve_sa  # noqa: E402
from problem import save_json  # noqa: E402
from schedule import AnnealingSchedule  # noqa: E402
from solver import IsingAnnealer, derive_problem_rng  # noqa: E402

DATA_DIR = THIS_DIR / "data"
RESULTS_DIR = THIS_DIR / "results_g77_g81"
BENCH_SEED = hashlib.sha256(b"HexSolid-Gset-G77-G81-Comparison-v1").digest()


def frame_samples_for_n(n_spins: int) -> int:
    if n_spins <= 800:
        return 12000
    if n_spins <= 2000:
        return 24000
    if n_spins <= 5000:
        return 48000
    if n_spins <= 10000:
        return 96000
    if n_spins <= 14000:
        return 144000  # 3.00s, df = 0.333 Hz, 4667 Hz band in [6.0, 10.67) kHz
    return 192000      # 4.00s, df = 0.250 Hz, 5000 Hz band in [6.0, 11.00) kHz


def cfl_safe_euler_dt(problem, dt_max: float = 0.46) -> float:
    v = np.ones(problem.n_spins, dtype=np.float64) / np.sqrt(max(problem.n_spins, 1))
    for _ in range(24):
        v = np.asarray(problem.couplings @ v, dtype=np.float64)
        nrm = float(np.linalg.norm(v))
        if nrm <= 1e-15:
            break
        v /= nrm
    spec_ratio = max(float(np.linalg.norm(np.asarray(problem.couplings @ v, dtype=np.float64))) / problem.j_scale, 2.0)
    return round(min(dt_max, 1.35 / (1.5 * spec_ratio)), 4)


def run_one_worker(
    instance: str,
    solver_key: str,
    out_json: Path,
    in_dev: int = 15,
    out_dev: int = 14,
    gain: float = 0.20,
    n_steps: int = 90,
) -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    meta = manifest[instance]
    gpath = DATA_DIR / f"{instance}.txt"
    bks = float(meta["bks"])

    rss_start = get_peak_rss_bytes()
    problem = load_gset_problem(gpath, name=instance)
    n = problem.n_spins
    m_frame = frame_samples_for_n(n)

    rng = derive_problem_rng(BENCH_SEED, instance, extra_tag=f"{solver_key}_b1")
    dt_cfl = cfl_safe_euler_dt(problem, 0.46)

    extra_info: dict[str, object] = {
        "frame_samples": m_frame,
        "n_steps": n_steps,
    }

    t0 = time.perf_counter()

    if solver_key == "duffing_real_audio":
        fmap = TopOctaveCombMap(n_spins=n, sample_rate=48000, frame_samples=m_frame, f_min_hz=6000.0, f_max_hz=11010.0)
        n_guard = 240 if n >= 14000 else 160
        backend = SkirtedAcousticBackend(
            frequency_map=fmap,
            allow_speaker_output=True,
            output_gain=gain,
            cal_averages=2,
            input_device=in_dev,
            output_device=out_dev,
            same_clock=True,
            out_channel="l",
            n_guard_each_side=n_guard,
        )
        res = solve_duffing_param_oim(
            problem=problem,
            backend=backend,
            rng=rng,
            n_steps=n_steps,
            batch_size=1,
            dt=0.35,
        )
        best_cut = res["best_cut"]
        mean_cut = res["mean_cut"]
        best_e = res["best_energy"]
        best_spins = res["best_spins"]
        elapsed = res["seconds"]
        extra_info.update({
            "rx_peak": backend.last_rx_peak,
            "cal_phase_std_deg": backend.cal_phase_std_deg,
            "cal_snr_median_db": backend.cal_snr_median_db,
            "out_channel": "l",
        })

    elif solver_key == "duffing_virtual_audio":
        fmap = TopOctaveCombMap(n_spins=n, sample_rate=48000, frame_samples=m_frame, f_min_hz=6000.0, f_max_hz=11010.0)
        backend = AudioLoopbackBackend(frequency_map=fmap, cyclic_prefix_samples=480, snr_db=42.0, cal_averages=2)
        res = solve_duffing_param_oim(
            problem=problem,
            backend=backend,
            rng=rng,
            n_steps=n_steps,
            batch_size=1,
            dt=0.35,
        )
        best_cut = res["best_cut"]
        mean_cut = res["mean_cut"]
        best_e = res["best_energy"]
        best_spins = res["best_spins"]
        elapsed = res["seconds"]

    elif solver_key == "phase_ode":
        sched = AnnealingSchedule(n_steps=n_steps, dt=dt_cfl)
        r = IsingAnnealer(sched, PhaseODEBackend(), "euler").solve(problem, rng, batch_size=1, prefix_sizes=(1,))
        best_cut = float(r.best_cut)
        mean_cut = float(r.mean_cut)
        best_e = float(r.best_energy)
        best_spins = r.best_spins
        elapsed = float(r.seconds)

    elif solver_key == "dsb":
        r = solve_dsb(problem, rng, n_steps=n_steps, batch_size=1, dt=0.50)
        best_cut = float(r.best_cut)
        mean_cut = float(r.mean_cut)
        best_e = float(r.best_energy)
        best_spins = r.best_spins
        elapsed = float(r.seconds)

    elif solver_key == "sa":
        r = solve_sa(problem, rng, n_sweeps=n_steps, batch_size=1)
        best_cut = float(r.best_cut)
        mean_cut = float(r.mean_cut)
        best_e = float(r.best_energy)
        best_spins = r.best_spins
        elapsed = float(r.seconds)

    else:
        raise ValueError(f"Unknown solver_key: {solver_key}")

    peak_rss = get_peak_rss_bytes()
    raw_cut = verify_gset_cut_from_raw_file(gpath, best_spins)
    audit_ok = abs(raw_cut - best_cut) < 1e-8

    bks_pct = 100.0 * (best_cut / bks)

    rec = {
        "instance": instance,
        "n_spins": n,
        "n_edges": problem.n_edges,
        "weights": meta["weights"],
        "topology": meta["topology"],
        "bks": bks,
        "solver": solver_key,
        "batch_size": 1,
        "n_steps": n_steps,
        "best_cut": best_cut,
        "mean_cut": mean_cut,
        "best_energy": best_e,
        "bks_percent": bks_pct,
        "seconds": elapsed,
        "peak_rss_mib": peak_rss / (1024.0 * 1024.0),
        "audit_passed": audit_ok,
        "extra_info": extra_info,
    }
    save_json(out_json, rec)
    print(
        f"[{instance:3s} | {solver_key:22s}] Cut={best_cut:8.1f} / {bks:.0f} ({bks_pct:6.2f}% BKS) | "
        f"time={elapsed:7.3f}s | PeakRSS={rec['peak_rss_mib']:5.1f} MiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=str, default="G77")
    parser.add_argument("--solver", type=str, default="duffing_virtual_audio")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--in-dev", type=int, default=15)
    parser.add_argument("--out-dev", type=int, default=14)
    parser.add_argument("--gain", type=float, default=0.20)
    parser.add_argument("--n-steps", type=int, default=90)
    args = parser.parse_args()

    if args.worker:
        run_one_worker(
            instance=args.instance,
            solver_key=args.solver,
            out_json=Path(args.out_json),
            in_dev=args.in_dev,
            out_dev=args.out_dev,
            gain=args.gain,
            n_steps=args.n_steps,
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
        env[k] = "1"

    instances = ["G77", "G81"]
    solvers = ["duffing_virtual_audio", "phase_ode", "dsb", "sa", "duffing_real_audio"]

    for inst in instances:
        for skey in solvers:
            out_file = RESULTS_DIR / f"{inst}__{skey}.json"
            print(f"\n>>> Running {inst} | {skey} ...", flush=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--instance", inst,
                "--solver", skey,
                "--out-json", str(out_file),
                "--in-dev", str(args.in_dev),
                "--out-dev", str(args.out_dev),
                "--gain", str(args.gain),
                "--n-steps", str(args.n_steps),
            ]
            subprocess.run(cmd, env=env, check=True)

    recs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*__*.json"))]
    (RESULTS_DIR / "summary.json").write_text(json.dumps(recs, indent=2) + "\n")
    print(f"\nAll finished! Saved {len(recs)} records to {RESULTS_DIR / 'summary.json'}.")


if __name__ == "__main__":
    main()

