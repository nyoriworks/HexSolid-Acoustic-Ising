"""Run isolated child-process benchmarks at N = 2000, 3000, 5000, 10000 for Phase 1 ODE and Phase 2 48kHz AudioLoopback."""
from __future__ import annotations

import os
import sys

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
from scipy import sparse

from backends.audio_loopback import AudioLoopbackBackend
from backends.ode_backend import PhaseODEBackend
from frequency_map import TopOctaveCombMap
from prepare import compute_source_hashes
from prepare_scale10k import build_seeded_maxcut_problem, build_streamed_sk_problem
from problem import (
    IsingProblem,
    ROOT,
    initial_greedy_spins,
    save_json,
    spins_to_bitstring,
    verify_external_references,
)
from schedule import AnnealingSchedule
from solver import IsingAnnealer, derive_problem_rng

PHASE1_SPECS = (
    ("maxcut2000", 32),
    ("sk2000", 32),
    ("maxcut3000", 32),
    ("sk3000", 32),
    ("maxcut5000", 32),
    ("sk5000", 16),
    ("maxcut10000", 32),
    ("sk10000", 16),
)

# (swap_id, case_name, frame_samples, batch_size)
LOOPBACK10K_SPECS = (
    ("loopback_maxcut2000_comb", "maxcut2000", 19200, 4),
    ("loopback_sk2000_comb", "sk2000", 19200, 4),
    ("loopback_maxcut3000_comb", "maxcut3000", 28800, 4),
    ("loopback_maxcut5000_comb", "maxcut5000", 48000, 2),
    ("loopback_maxcut10000_comb", "maxcut10000", 96000, 2),
)


def current_peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def load_scale10k_problem(case_name: str) -> tuple[IsingProblem, dict[str, object]]:
    seeds_doc = json.loads((ROOT / "instances_scale10k" / "seeds.json").read_text())
    meta = seeds_doc["instances"][case_name]
    n = int(meta["n_spins"])
    seed_hex = str(meta["seed_hex"])
    if meta["kind"] == "maxcut":
        prob = build_seeded_maxcut_problem(n, int(meta["n_edges"]), seed_hex, name=case_name)
    else:
        prob = build_streamed_sk_problem(n, seed_hex, name=case_name)
    return prob, seeds_doc


def incremental_steepest_greedy(problem: IsingProblem, n_starts: int) -> dict[str, object]:
    """Exact steepest 1-spin-flip local search with O(N)/O(deg) incremental field maintenance."""
    t0 = time.perf_counter()
    energies = np.empty(n_starts, dtype=np.float64)
    steps_arr = np.empty(n_starts, dtype=np.int64)
    best_energy = float("inf")
    best_spins: np.ndarray | None = None
    best_seed = -1

    is_sp = sparse.issparse(problem.couplings)
    csr = problem.couplings if is_sp else None
    dense = None if is_sp else problem.couplings

    for seed in range(n_starts):
        z = initial_greedy_spins(problem.name, problem.n_spins, seed)
        field = problem.matvec(z) + problem.field
        steps = 0
        while True:
            delta = -2.0 * z * field
            q = int(np.argmin(delta))
            if delta[q] >= -1e-12:
                break
            old_zq = float(z[q])
            z[q] = -old_zq
            dz = -2.0 * old_zq
            if is_sp:
                assert csr is not None
                beg, end = int(csr.indptr[q]), int(csr.indptr[q + 1])
                field[csr.indices[beg:end]] += dz * csr.data[beg:end]
            else:
                assert dense is not None
                field += dz * dense[q]
            steps += 1
            if steps % 512 == 0:
                field = problem.matvec(z) + problem.field

        while True:
            exact_delta = -2.0 * z * (problem.matvec(z) + problem.field)
            q = int(np.argmin(exact_delta))
            if exact_delta[q] >= -1e-12:
                break
            z[q] *= -1.0
            steps += 1

        e = float(problem.energy(z))
        energies[seed] = e
        steps_arr[seed] = steps
        if e < best_energy:
            best_energy = e
            best_spins = z.copy()
            best_seed = seed

    elapsed = time.perf_counter() - t0
    assert best_spins is not None
    out: dict[str, object] = {
        "n_starts": n_starts,
        "seconds": elapsed,
        "best_energy": best_energy,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "best_seed": best_seed,
        "best_bitstring": spins_to_bitstring(best_spins),
        "mean_steps": float(np.mean(steps_arr)),
        "energies": energies,
    }
    if problem.kind == "maxcut":
        cuts = 0.5 * (problem.total_edge_weight - energies)
        out["best_cut"] = float(np.max(cuts))
        out["mean_cut"] = float(np.mean(cuts))
        out["std_cut"] = float(np.std(cuts))
    return out


def worker_oim(case_name: str, batch_size: int, out_path: Path) -> None:
    t0 = time.perf_counter()
    problem, seeds_doc = load_scale10k_problem(case_name)
    build_seconds = time.perf_counter() - t0

    master_seed = bytes.fromhex(str(seeds_doc["solver_master_seed_hex"]))
    schedule = AnnealingSchedule()
    backend = PhaseODEBackend()
    annealer = IsingAnnealer(schedule=schedule, backend=backend, integrator="heun")
    rng = derive_problem_rng(master_seed, case_name, extra_tag="scale10k_ode")

    res = annealer.solve(problem, rng=rng, batch_size=batch_size, prefix_sizes=(4, 16, batch_size))
    rss_bytes = current_peak_rss_bytes()

    payload: dict[str, object] = {
        "case": case_name,
        "kind": problem.kind,
        "n_spins": problem.n_spins,
        "n_edges": problem.n_edges,
        "is_sparse": problem.is_sparse,
        "coupling_storage_bytes": problem.coupling_storage_bytes,
        "coupling_storage_mib": problem.coupling_storage_bytes / (1024.0 * 1024.0),
        "j_scale": problem.j_scale,
        "backend": res.backend_name,
        "integrator": res.integrator,
        "batch_size": res.batch_size,
        "n_steps": res.n_steps,
        "build_seconds": build_seconds,
        "solve_seconds": res.seconds,
        "total_seconds": build_seconds + res.seconds,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_mib": rss_bytes / (1024.0 * 1024.0),
        "best_energy": res.best_energy,
        "best_energy_per_spin": res.best_energy / float(problem.n_spins),
        "mean_energy": res.mean_energy,
        "std_energy": res.std_energy,
        "best_index_in_batch": res.best_index_in_batch,
        "best_bitstring": res.best_bitstring,
        "energies": res.energies,
        "backend_calls": backend.calls,
    }
    if problem.kind == "maxcut":
        payload["total_edge_weight"] = problem.total_edge_weight
        payload["best_cut"] = res.best_cut
        payload["mean_cut"] = res.mean_cut
        payload["std_cut"] = res.std_cut
    save_json(out_path, payload)


def worker_greedy(case_name: str, n_starts: int, out_path: Path) -> None:
    t0 = time.perf_counter()
    problem, _ = load_scale10k_problem(case_name)
    build_seconds = time.perf_counter() - t0
    greedy_res = incremental_steepest_greedy(problem, n_starts=n_starts)
    rss_bytes = current_peak_rss_bytes()
    greedy_res["build_seconds"] = build_seconds
    greedy_res["total_seconds"] = build_seconds + float(greedy_res["seconds"])
    greedy_res["peak_rss_bytes"] = rss_bytes
    greedy_res["peak_rss_mib"] = rss_bytes / (1024.0 * 1024.0)
    save_json(out_path, greedy_res)


def worker_loopback(swap_id: str, out_path: Path) -> None:
    spec = next((r for r in LOOPBACK10K_SPECS if r[0] == swap_id), None)
    if spec is None:
        raise ValueError(f"Unknown swap {swap_id!r}")
    _, case_name, frame_samples, batch_size = spec
    problem, seeds_doc = load_scale10k_problem(case_name)
    master_seed = bytes.fromhex(str(seeds_doc["solver_master_seed_hex"]))

    fmap = TopOctaveCombMap(problem.n_spins, frame_samples=frame_samples)
    schedule = AnnealingSchedule()
    backend = AudioLoopbackBackend(frequency_map=fmap, cyclic_prefix_samples=480, snr_db=42.0, cal_averages=2)
    annealer = IsingAnnealer(schedule=schedule, backend=backend, integrator="euler")
    rng = derive_problem_rng(master_seed, case_name, extra_tag=f"scale10k_{swap_id}")

    res = annealer.solve(problem, rng=rng, batch_size=batch_size, prefix_sizes=(batch_size,))
    rss_bytes = current_peak_rss_bytes()

    payload: dict[str, object] = {
        "swap_id": swap_id,
        "case": case_name,
        "kind": problem.kind,
        "n_spins": problem.n_spins,
        "backend": res.backend_name,
        "frequency_map": fmap.validate(),
        "frame_duration_seconds": fmap.frame_duration_seconds,
        "physical_realtime_equivalent_minutes": (res.n_steps * fmap.frame_duration_seconds) / 60.0,
        "integrator": res.integrator,
        "snr_db": backend.snr_db,
        "batch_size": res.batch_size,
        "n_steps": res.n_steps,
        "solve_seconds": res.seconds,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_mib": rss_bytes / (1024.0 * 1024.0),
        "last_crest_factor": backend.last_crest_factor,
        "best_energy": res.best_energy,
        "best_energy_per_spin": res.best_energy / float(problem.n_spins),
        "mean_energy": res.mean_energy,
        "best_bitstring": res.best_bitstring,
        "energies": res.energies,
        "backend_calls": backend.calls,
    }
    if problem.kind == "maxcut":
        payload["total_edge_weight"] = problem.total_edge_weight
        payload["best_cut"] = res.best_cut
        payload["mean_cut"] = res.mean_cut
    save_json(out_path, payload)


def run_child(args_list: list[str]) -> None:
    env = dict(os.environ)
    for k in _THREAD_VARS:
        env[k] = "1"
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args_list], env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-oim", type=str)
    parser.add_argument("--worker-greedy", type=str)
    parser.add_argument("--worker-loopback", type=str)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.worker_oim:
        assert args.out is not None
        worker_oim(args.worker_oim, args.batch, args.out)
        return
    if args.worker_greedy:
        assert args.out is not None
        worker_greedy(args.worker_greedy, args.batch, args.out)
        return
    if args.worker_loopback:
        assert args.out is not None
        worker_loopback(args.worker_loopback, args.out)
        return

    verify_external_references()
    v1_sources = json.loads((ROOT / "source-hashes.json").read_text())
    if compute_source_hashes() != v1_sources:
        raise RuntimeError("v1 tracked core sources changed!")

    res_dir = ROOT / "results_scale10k"
    cases_dir = res_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    phase1_rows: list[dict[str, object]] = []
    for case_name, batch_size in PHASE1_SPECS:
        oim_file = cases_dir / f"{case_name}-oim.json"
        grd_file = cases_dir / f"{case_name}-greedy.json"
        if not args.aggregate_only:
            print(f"[Scale10k Phase 1 ODE] running {case_name} (batch={batch_size}) ...", flush=True)
            run_child(["--worker-oim", case_name, "--batch", str(batch_size), "--out", str(oim_file)])
            run_child(["--worker-greedy", case_name, "--batch", str(batch_size), "--out", str(grd_file)])

        oim_data = json.loads(oim_file.read_text())
        grd_data = json.loads(grd_file.read_text())
        oim_summary = {k: v for k, v in oim_data.items() if k != "energies"}
        grd_summary = {k: v for k, v in grd_data.items() if k != "energies"}
        row = {
            "case": case_name,
            "oim": oim_summary,
            "greedy": grd_summary,
            "oim_minus_greedy_best_energy": float(oim_data["best_energy"]) - float(grd_data["best_energy"]),
            "oim_minus_greedy_mean_energy": float(oim_data["mean_energy"]) - float(grd_data["mean_energy"]),
        }
        phase1_rows.append(row)
        print(
            f"  -> {case_name}: OIM best_E={oim_data['best_energy']:.4f} (E/N={oim_data['best_energy_per_spin']:.5f}, "
            f"solve={oim_data['solve_seconds']:.3f}s, RSS={oim_data['peak_rss_mib']:.1f} MiB) | "
            f"Greedy best_E={grd_data['best_energy']:.4f} ({grd_data['seconds']:.3f}s)",
            flush=True,
        )

    loopback_rows: list[dict[str, object]] = []
    for swap_id, _, _, _ in LOOPBACK10K_SPECS:
        lb_file = cases_dir / f"{swap_id}.json"
        if not args.aggregate_only:
            print(f"[Scale10k Phase 2 AudioLoopback] running {swap_id} ...", flush=True)
            run_child(["--worker-loopback", swap_id, "--out", str(lb_file)])
        lb_data = json.loads(lb_file.read_text())
        lb_summary = {k: v for k, v in lb_data.items() if k != "energies"}
        loopback_rows.append(lb_summary)
        print(
            f"  -> {swap_id}: best_E={lb_data['best_energy']:.4f} (solve={lb_data['solve_seconds']:.3f}s, "
            f"RSS={lb_data['peak_rss_mib']:.1f} MiB, CF={lb_data['last_crest_factor']:.2f})",
            flush=True,
        )

    sparse_rss_ok = all(
        float(r["oim"]["peak_rss_mib"]) < 512.0 for r in phase1_rows if r["oim"]["is_sparse"]
    )
    dense_rss_ok = all(
        float(r["oim"]["peak_rss_mib"]) < 2048.0 for r in phase1_rows if not r["oim"]["is_sparse"]
    )
    finite_ok = all(np.isfinite(float(r["oim"]["best_energy"])) for r in phase1_rows) and all(
        np.isfinite(float(r["best_energy"])) for r in loopback_rows
    )
    passed = sparse_rss_ok and dense_rss_ok and finite_ok

    summary = {
        "passed": passed,
        "gates": {
            "all_sparse_maxcut_up_to_10000_rss_under_512_mib": sparse_rss_ok,
            "all_dense_sk_up_to_10000_rss_under_2048_mib": dense_rss_ok,
            "all_phase1_and_phase2_solutions_finite": finite_ok,
        },
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "blas_threads": {k: os.environ.get(k) for k in _THREAD_VARS},
        },
        "phase1_ode_scale10k": phase1_rows,
        "phase2_loopback_scale10k": loopback_rows,
    }
    save_json(res_dir / "summary.json", summary)
    if not passed:
        raise SystemExit(f"Scale10k pass gate failed: {summary['gates']}")
    print("All N = 2000, 3000, 5000, 10000 Phase 1 and Phase 2 benchmarks PASSED.", flush=True)


if __name__ == "__main__":
    main()

