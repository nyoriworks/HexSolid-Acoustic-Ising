"""Benchmark runner with per-case isolated subprocess Peak RSS measurement and Phase 2 loopback swap verification."""
from __future__ import annotations

import os
import sys

# Enforce single-threaded BLAS before importing numpy/scipy
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
if any(os.environ.get(k) != "1" for k in _THREAD_VARS):
    env = dict(os.environ)
    for k in _THREAD_VARS:
        env[k] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import argparse
import json
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import scipy

from backends.audio_loopback import AudioLoopbackBackend
from backends.ode_backend import PhaseODEBackend
from frequency_map import Musical12ToneMap, TopOctaveCombMap
from prepare import compute_source_hashes
from problem import (
    MAXCUT24_DIR,
    ROOT,
    SK24_DIR,
    IsingProblem,
    save_json,
    sha256_file,
    steepest_greedy_baseline,
    verify_external_references,
)
from schedule import AnnealingSchedule
from solver import IsingAnnealer, derive_problem_rng

PHASE1_CASES = (
    "maxcut12",
    "sk12",
    "maxcut24",
    "sk24",
    "maxcut100",
    "sk100",
    "maxcut500",
    "sk500",
    "maxcut1000",
    "sk1000",
)

LOOPBACK_SWAPS = (
    ("loopback_maxcut12_musical12", "maxcut12", "musical12", 32),
    ("loopback_maxcut24_comb", "maxcut24", "comb", 32),
    ("loopback_sk24_comb", "sk24", "comb", 32),
    ("loopback_sk500_comb", "sk500", "comb", 8),
)


def current_peak_rss_bytes() -> int:
    """Return process peak RSS in bytes (ru_maxrss is bytes on macOS Darwin, KiB on Linux)."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def load_case_problem(case_name: str) -> IsingProblem:
    """Load ONLY the IsingProblem coupling/graph data (NEVER loads any oracle file)."""
    inst_dir = ROOT / "instances"
    if case_name == "maxcut24":
        return IsingProblem.from_maxcut_json(MAXCUT24_DIR / "graph.json", name="maxcut24")
    if case_name == "sk24":
        return IsingProblem.from_sk_npy(SK24_DIR / "couplings.npy", name="sk24")
    if case_name.startswith("maxcut"):
        return IsingProblem.from_maxcut_json(inst_dir / f"{case_name}-graph.json", name=case_name)
    if case_name.startswith("sk"):
        return IsingProblem.from_sk_npy(inst_dir / f"{case_name}-couplings.npy", name=case_name)
    raise ValueError(f"Unknown case {case_name!r}")


def worker_oim(case_name: str, out_path: Path) -> None:
    """Isolated child worker: solves one problem with PhaseODEBackend and records standalone Peak RSS."""
    master_seed = (ROOT / "instances" / "oim-master-seed.bin").read_bytes()
    t_load0 = time.perf_counter()
    problem = load_case_problem(case_name)
    load_seconds = time.perf_counter() - t_load0

    schedule = AnnealingSchedule()
    backend = PhaseODEBackend()
    annealer = IsingAnnealer(schedule=schedule, backend=backend, integrator="heun")
    rng = derive_problem_rng(master_seed, case_name, extra_tag="phase1_ode")

    res = annealer.solve(problem, rng=rng, batch_size=128, prefix_sizes=(16, 64, 128))
    rss_bytes = current_peak_rss_bytes()

    payload: dict[str, object] = {
        "case": case_name,
        "kind": problem.kind,
        "n_spins": problem.n_spins,
        "n_edges": problem.n_edges,
        "is_sparse": problem.is_sparse,
        "coupling_storage_bytes": problem.coupling_storage_bytes,
        "j_scale": problem.j_scale,
        "backend": res.backend_name,
        "integrator": res.integrator,
        "batch_size": res.batch_size,
        "n_steps": res.n_steps,
        "load_seconds": load_seconds,
        "solve_seconds": res.seconds,
        "total_seconds": load_seconds + res.seconds,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_mib": rss_bytes / (1024.0 * 1024.0),
        "best_energy": res.best_energy,
        "best_energy_per_spin": res.best_energy / float(problem.n_spins),
        "mean_energy": res.mean_energy,
        "std_energy": res.std_energy,
        "best_index_in_batch": res.best_index_in_batch,
        "best_bitstring": res.best_bitstring,
        "prefixes": res.prefixes,
        "energies": res.energies,
        "backend_calls": backend.calls,
    }
    if problem.kind == "maxcut":
        payload["total_edge_weight"] = problem.total_edge_weight
        payload["best_cut"] = res.best_cut
        payload["mean_cut"] = res.mean_cut
        payload["std_cut"] = res.std_cut
    save_json(out_path, payload)


def worker_greedy(case_name: str, out_path: Path) -> None:
    """Isolated child worker: runs 128-start steepest 1-spin-flip Greedy and records standalone Peak RSS."""
    t_load0 = time.perf_counter()
    problem = load_case_problem(case_name)
    load_seconds = time.perf_counter() - t_load0

    greedy_res = steepest_greedy_baseline(problem, n_starts=128)
    rss_bytes = current_peak_rss_bytes()
    greedy_res["load_seconds"] = load_seconds
    greedy_res["total_seconds"] = load_seconds + float(greedy_res["seconds"])
    greedy_res["peak_rss_bytes"] = rss_bytes
    greedy_res["peak_rss_mib"] = rss_bytes / (1024.0 * 1024.0)
    save_json(out_path, greedy_res)


def worker_loopback(swap_id: str, out_path: Path) -> None:
    """Isolated child worker: runs UNMODIFIED IsingAnnealer with AudioLoopbackBackend (48 kHz iRFFT -> room -> RFFT)."""
    spec = next((row for row in LOOPBACK_SWAPS if row[0] == swap_id), None)
    if spec is None:
        raise ValueError(f"Unknown loopback swap {swap_id!r}")
    _, case_name, map_kind, batch_size = spec

    master_seed = (ROOT / "instances" / "oim-master-seed.bin").read_bytes()
    problem = load_case_problem(case_name)
    fmap = Musical12ToneMap(problem.n_spins) if map_kind == "musical12" else TopOctaveCombMap(problem.n_spins)

    schedule = AnnealingSchedule()
    backend = AudioLoopbackBackend(frequency_map=fmap, cyclic_prefix_samples=480, snr_db=42.0, cal_averages=4)
    # Identical IsingAnnealer class — zero code modification to problem, schedule, or solver!
    annealer = IsingAnnealer(schedule=schedule, backend=backend, integrator="euler")
    rng = derive_problem_rng(master_seed, case_name, extra_tag=f"phase2_{swap_id}")

    rec_idx = 0 if swap_id == "loopback_maxcut12_musical12" else None
    res = annealer.solve(
        problem,
        rng=rng,
        batch_size=batch_size,
        prefix_sizes=(8, 16, 32),
        record_trajectory_index=rec_idx,
    )
    rss_bytes = current_peak_rss_bytes()

    wav_info = None
    if res.phi_history is not None:
        # Subsample every 20 steps (41 frames * 0.1 s = 4.1 s audible C5..B5 lock-in WAV)
        sub_hist = res.phi_history[::20]
        wav_path = ROOT / "results" / "maxcut12-chord-evolution.wav"
        wav_info = backend.export_trajectory_wav(sub_hist, wav_path, audible_chord=True)

    payload: dict[str, object] = {
        "swap_id": swap_id,
        "case": case_name,
        "kind": problem.kind,
        "n_spins": problem.n_spins,
        "backend": res.backend_name,
        "frequency_map": fmap.validate(),
        "integrator": res.integrator,
        "snr_db": backend.snr_db,
        "cyclic_prefix_samples": backend.cyclic_prefix_samples,
        "batch_size": res.batch_size,
        "n_steps": res.n_steps,
        "solve_seconds": res.seconds,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_mib": rss_bytes / (1024.0 * 1024.0),
        "last_crest_factor": backend.last_crest_factor,
        "best_energy": res.best_energy,
        "mean_energy": res.mean_energy,
        "std_energy": res.std_energy,
        "best_bitstring": res.best_bitstring,
        "energies": res.energies,
        "backend_calls": backend.calls,
        "wav_export": wav_info,
    }
    if problem.kind == "maxcut":
        payload["best_cut"] = res.best_cut
        payload["mean_cut"] = res.mean_cut
    save_json(out_path, payload)


def evaluate_oracle_match(case_name: str, energies: np.ndarray, problem_kind: str) -> dict[str, object] | None:
    """Post-hoc oracle comparison in parent orchestrator ONLY (never called by solver workers)."""
    inst_dir = ROOT / "instances"
    if case_name == "maxcut12":
        oracle = json.loads((inst_dir / "maxcut12-oracle.json").read_text())
        ground_e = float(oracle["ground_energy"])
        opt_cut = float(oracle["optimal_cut"])
    elif case_name == "sk12":
        oracle = json.loads((inst_dir / "sk12-oracle.json").read_text())
        ground_e = float(oracle["ground_energy"])
        opt_cut = None
    elif case_name == "maxcut24":
        oracle = json.loads((MAXCUT24_DIR / "oracle.json").read_text())
        opt_cut = float(oracle["optimal_cut"])
        w_tot = float(oracle["total_weight"])
        ground_e = w_tot - 2.0 * opt_cut
    elif case_name == "sk24":
        oracle = json.loads((SK24_DIR / "oracle.json").read_text())
        ground_e = float(oracle["ground_energy"])
        opt_cut = None
    else:
        return None

    hits = int(np.sum(np.abs(energies - ground_e) < 1e-8))
    best_e = float(np.min(energies))
    info: dict[str, object] = {
        "exact_ground_energy": ground_e,
        "achieved_best_energy": best_e,
        "energy_gap_to_ground": best_e - ground_e,
        "ground_hits": hits,
        "total_trajectories": int(len(energies)),
        "ground_hit_rate": float(hits) / float(len(energies)),
        "reached_exact_ground": bool(abs(best_e - ground_e) < 1e-8),
    }
    if problem_kind == "maxcut" and opt_cut is not None:
        info["exact_optimal_cut"] = opt_cut
    return info


def run_child(flag: str, target: str, out_path: Path) -> None:
    env = dict(os.environ)
    for k in _THREAD_VARS:
        env[k] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), flag, target, "--out", str(out_path)]
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-oim", type=str)
    parser.add_argument("--worker-greedy", type=str)
    parser.add_argument("--worker-loopback", type=str)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.worker_oim:
        assert args.out is not None
        worker_oim(args.worker_oim, args.out)
        return
    if args.worker_greedy:
        assert args.out is not None
        worker_greedy(args.worker_greedy, args.out)
        return
    if args.worker_loopback:
        assert args.out is not None
        worker_loopback(args.worker_loopback, args.out)
        return

    # Parent orchestrator: verify frozen state first
    verify_external_references()
    frozen_sources = json.loads((ROOT / "source-hashes.json").read_text())
    if compute_source_hashes() != frozen_sources:
        raise RuntimeError("Tracked source files changed since prepare.py froze source-hashes.json")
    prep = json.loads((ROOT / "preparation.json").read_text())
    for rel, digest in prep["artifacts"].items():
        if sha256_file(ROOT / rel) != digest:
            raise RuntimeError(f"Prepared artifact hash mismatch: {rel}")

    res_dir = ROOT / "results"
    cases_dir = res_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    phase1_rows: list[dict[str, object]] = []
    for case_name in PHASE1_CASES:
        print(f"[Phase 1 ODE] running isolated workers for {case_name} ...", flush=True)
        oim_file = cases_dir / f"{case_name}-oim.json"
        greedy_file = cases_dir / f"{case_name}-greedy.json"
        run_child("--worker-oim", case_name, oim_file)
        run_child("--worker-greedy", case_name, greedy_file)

        oim_data = json.loads(oim_file.read_text())
        greedy_data = json.loads(greedy_file.read_text())
        oim_energies = np.array(oim_data["energies"], dtype=np.float64)
        greedy_energies = np.array(greedy_data["energies"], dtype=np.float64)

        oracle_oim = evaluate_oracle_match(case_name, oim_energies, str(oim_data["kind"]))
        oracle_greedy = evaluate_oracle_match(case_name, greedy_energies, str(oim_data["kind"]))

        # Remove bulky 128-element raw array from top-level summary (preserved in cases/<case>-*.json)
        oim_summary = {k: v for k, v in oim_data.items() if k != "energies"}
        greedy_summary = {k: v for k, v in greedy_data.items() if k != "energies"}
        row = {
            "case": case_name,
            "oim": oim_summary,
            "greedy": greedy_summary,
            "oracle_oim": oracle_oim,
            "oracle_greedy": oracle_greedy,
            "oim_minus_greedy_best_energy": float(oim_data["best_energy"]) - float(greedy_data["best_energy"]),
            "oim_minus_greedy_mean_energy": float(oim_data["mean_energy"]) - float(greedy_data["mean_energy"]),
        }
        phase1_rows.append(row)
        msg = (
            f"  -> {case_name}: OIM best_E={oim_data['best_energy']:.6f} "
            f"(mean={oim_data['mean_energy']:.4f}, {oim_data['solve_seconds']:.4f}s, RSS={oim_data['peak_rss_mib']:.1f} MiB) | "
            f"Greedy best_E={greedy_data['best_energy']:.6f} ({greedy_data['seconds']:.4f}s)"
        )
        if oracle_oim is not None:
            msg += f" | exact_hits={oracle_oim['ground_hits']}/128"
        print(msg, flush=True)

    loopback_rows: list[dict[str, object]] = []
    for swap_id, case_name, _, _ in LOOPBACK_SWAPS:
        print(f"[Phase 2 AudioLoopback] running isolated worker for {swap_id} ...", flush=True)
        lb_file = cases_dir / f"{swap_id}.json"
        run_child("--worker-loopback", swap_id, lb_file)
        lb_data = json.loads(lb_file.read_text())
        lb_energies = np.array(lb_data["energies"], dtype=np.float64)
        oracle_lb = evaluate_oracle_match(case_name, lb_energies, str(lb_data["kind"]))
        lb_summary = {k: v for k, v in lb_data.items() if k != "energies"}
        lb_summary["oracle"] = oracle_lb
        loopback_rows.append(lb_summary)
        msg = (
            f"  -> {swap_id}: best_E={lb_data['best_energy']:.6f} "
            f"({lb_data['solve_seconds']:.4f}s, RSS={lb_data['peak_rss_mib']:.1f} MiB, CF={lb_data['last_crest_factor']:.2f})"
        )
        if oracle_lb is not None:
            msg += f" | exact_hits={oracle_lb['ground_hits']}/{lb_data['batch_size']}"
        print(msg, flush=True)

    # Historical quantum-circuit / MPS / TTN reference figures from read-only HexSolid artifacts (explicitly labeled)
    sk24_past = json.loads((SK24_DIR / "results.json").read_text())
    mc24_past_oracle = json.loads((MAXCUT24_DIR / "oracle.json").read_text())
    historical_context = {
        "note": (
            "Read-only historical values from tensor-sk-qaoa-v1 and tensor-maxcut-qaoa-v1 for scale context; "
            "OIM solves classical combinatorial Ising minimization (no quantum superposition/Shor gates)."
        ),
        "sk24_bulk_ttn_svd_seconds": float(sk24_past["svd_seconds"]),
        "sk24_bulk_ttn_peak_rss_bytes": int(sk24_past["peak_rss_bytes"]),
        "sk24_qaoa_p1_mean_energy": float(sk24_past["dense_readout"]["mean_energy"]),
        "sk24_qaoa_p1_ground_probability": float(sk24_past["dense_readout"]["ground_probability"]),
        "maxcut24_exact_enumeration_seconds": float(mc24_past_oracle["enumeration_seconds"]),
    }

    # Strict pass criteria assertions (non-zero exit if any fail!)
    exact_phase1_ok = all(
        row["oracle_oim"] is not None and bool(row["oracle_oim"]["reached_exact_ground"])
        for row in phase1_rows
        if row["case"] in ("maxcut12", "sk12", "maxcut24", "sk24")
    )
    exact_loopback_ok = all(
        row["oracle"] is not None and bool(row["oracle"]["reached_exact_ground"])
        for row in loopback_rows
        if row["case"] in ("maxcut12", "maxcut24", "sk24")
    )
    non_exploding_rss_ok = all(
        float(row["oim"]["peak_rss_mib"]) < 256.0 for row in phase1_rows
    )
    passed_all = exact_phase1_ok and exact_loopback_ok and non_exploding_rss_ok

    summary = {
        "passed": passed_all,
        "gates": {
            "exact_n12_n24_phase1_ground_reached": exact_phase1_ok,
            "exact_n12_n24_loopback_ground_reached": exact_loopback_ok,
            "all_cases_peak_rss_under_256_mib": non_exploding_rss_ok,
        },
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "blas_threads": {k: os.environ.get(k) for k in _THREAD_VARS},
            "rss_measurement": "isolated fresh child process per case via resource.RUSAGE_SELF",
        },
        "schedule": AnnealingSchedule().to_dict(),
        "phase1_ode_benchmarks": phase1_rows,
        "phase2_audio_loopback_swaps": loopback_rows,
        "historical_reference_context": historical_context,
        "source_hashes": compute_source_hashes(),
    }
    save_json(res_dir / "summary.json", summary)

    if not passed_all:
        raise SystemExit(f"Benchmark pass gate failed: {summary['gates']}")
    print("All Phase 1 (ODE) and Phase 2 (48kHz AudioLoopback swap) benchmark gates PASSED.", flush=True)


if __name__ == "__main__":
    main()

