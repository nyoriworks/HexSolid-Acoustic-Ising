"""One-time problem instance generation from OS entropy, small-N exact oracles, and physics self-checks."""
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

from pathlib import Path
import time
import numpy as np

from backends.audio_loopback import AudioLoopbackBackend
from backends.ode_backend import PhaseODEBackend
from frequency_map import Musical12ToneMap, TopOctaveCombMap
from problem import (
    ROOT,
    IsingProblem,
    draw_maxcut_from_bytes,
    draw_sk_from_bytes,
    mitm_energies_exact,
    save_json,
    sha256_file,
    spins_to_bitstring,
    verify_external_references,
    xor_energies_exact,
)
from schedule import AnnealingSchedule
from solver import IsingAnnealer

TRACKED_SOURCES = (
    "PROTOCOL.md",
    "frequency_map.py",
    "backends/__init__.py",
    "backends/base.py",
    "backends/ode_backend.py",
    "backends/audio_loopback.py",
    "backends/real_acoustic.py",
    "problem.py",
    "schedule.py",
    "solver.py",
    "prepare.py",
    "run.py",
)


def compute_source_hashes() -> dict[str, str]:
    return {rel: sha256_file(ROOT / rel) for rel in TRACKED_SOURCES}


def build_exact_oracle(problem: IsingProblem) -> dict[str, object]:
    """Compute exact 2^N ground state via two independent methods (XOR and MITM spin-matrix)."""
    if problem.n_spins > 20:
        raise ValueError("build_exact_oracle in prepare.py is intended for N <= 20")
    k_dense = problem.couplings.toarray() if problem.is_sparse else np.asarray(problem.couplings)
    t0 = time.perf_counter()
    e_xor = xor_energies_exact(k_dense)
    t_xor = time.perf_counter() - t0
    t1 = time.perf_counter()
    e_mitm = mitm_energies_exact(k_dense)
    t_mitm = time.perf_counter() - t1

    max_diff = float(np.max(np.abs(e_xor - e_mitm)))
    if max_diff >= 1e-10:
        raise ValueError(f"Exact energy tables disagree for {problem.name}: max_diff={max_diff}")

    min_e = float(np.min(e_xor))
    ground_idx = np.flatnonzero(np.abs(e_xor - min_e) < 1e-10)
    excited_mask = np.abs(e_xor - min_e) >= 1e-10
    first_excited = float(np.min(e_xor[excited_mask])) if np.any(excited_mask) else min_e

    witnesses = [format(int(idx), f"0{problem.n_spins}b") for idx in ground_idx]
    # Verify independent energy evaluation on ground witness
    for w in witnesses:
        s = np.array([1.0 if ch == "0" else -1.0 for ch in w], dtype=np.float64)
        if abs(float(problem.energy(s)) - min_e) >= 1e-10:
            raise ValueError(f"Witness energy mismatch on {problem.name}")

    oracle: dict[str, object] = {
        "problem": problem.name,
        "n_spins": problem.n_spins,
        "states_enumerated": int(1 << problem.n_spins),
        "ground_energy": min_e,
        "energy_per_spin": min_e / float(problem.n_spins),
        "first_excited_energy": first_excited,
        "gap": first_excited - min_e,
        "degeneracy": int(len(ground_idx)),
        "ground_indices": [int(i) for i in ground_idx],
        "witnesses": witnesses,
        "uniform_mean_energy": float(np.mean(e_xor)),
        "table_max_difference": max_diff,
        "xor_seconds": t_xor,
        "mitm_seconds": t_mitm,
    }
    if problem.kind == "maxcut":
        opt_cut = 0.5 * (problem.total_edge_weight - min_e)
        oracle["optimal_cut"] = float(opt_cut)
        oracle["uniform_expected_cut"] = float(0.5 * (problem.total_edge_weight - float(np.mean(e_xor))))
    return oracle


def run_self_checks() -> dict[str, object]:
    """Validate Lyapunov gradient identity, FFT bin orthogonality, and Schroeder crest-factor reduction."""
    rng = np.random.default_rng(20261103)
    n = 12
    g = rng.standard_normal((n, n))
    k = np.triu(-g / np.sqrt(n), 1)
    k = k + k.T
    field = 0.2 * rng.standard_normal(n)
    prob = IsingProblem(
        name="grad_check",
        kind="sk",
        n_spins=n,
        couplings=k,
        field=field,
        total_edge_weight=0.0,
        n_edges=n * (n - 1) // 2,
        j_scale=IsingProblem._compute_j_scale(k, field),
    )
    sched = AnnealingSchedule()
    annealer = IsingAnnealer(sched, PhaseODEBackend())
    annealer.backend.reset(n, rng)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = 0.37
    kc, ks = sched.kc(u), sched.ks(u)
    z = annealer.backend.measure_phasors(phi)
    analytic_neg_grad = annealer._drift_from_phasors(prob, z, u)
    eps = 1e-7
    fd_neg_grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p_plus = phi.copy()
        p_minus = phi.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        v_plus = float(prob.continuous_potential(p_plus, kc, ks))
        v_minus = float(prob.continuous_potential(p_minus, kc, ks))
        fd_neg_grad[i] = -(v_plus - v_minus) / (2.0 * eps)
    grad_err = float(np.max(np.abs(analytic_neg_grad - fd_neg_grad)))
    if grad_err >= 1e-8:
        raise ValueError(f"Lyapunov gradient check failed: max error {grad_err}")

    # Validate frequency maps & noiseless loopback phase recovery
    fmap_checks = [Musical12ToneMap(12).validate()]
    for sz in (24, 100, 500, 1000):
        fmap_checks.append(TopOctaveCombMap(sz).validate())

    # Crest factor reduction check at N=500 with aligned phases phi=0
    fmap500 = TopOctaveCombMap(500)
    lb_no_schroeder = AudioLoopbackBackend(fmap500, snr_db=None, use_schroeder_offset=False)
    lb_schroeder = AudioLoopbackBackend(fmap500, snr_db=None, use_schroeder_offset=True)
    lb_no_schroeder.reset(500, np.random.default_rng(1))
    lb_schroeder.reset(500, np.random.default_rng(1))
    aligned = np.zeros(500, dtype=np.float64)
    lb_no_schroeder.synthesize_frame(aligned)
    lb_schroeder.synthesize_frame(aligned)
    cf_raw = lb_no_schroeder.last_crest_factor
    cf_schroeder = lb_schroeder.last_crest_factor

    test_phi = rng.uniform(0.0, 2.0 * np.pi, size=(4, 500))
    z_expected = np.cos(test_phi) + 1j * np.sin(test_phi)
    z_actual = lb_schroeder.measure_phasors(test_phi)
    phasor_err = float(np.max(np.abs(z_actual - z_expected)))
    if phasor_err >= 1e-12 or not (cf_schroeder < 0.35 * cf_raw):
        raise ValueError(f"Noiseless loopback/Schroeder check failed: err={phasor_err}, cf={cf_schroeder}/{cf_raw}")

    return {
        "lyapunov_neg_grad_max_error": grad_err,
        "noiseless_500tone_multipath_phasor_error": phasor_err,
        "aligned_500tone_crest_factor_uncompensated": cf_raw,
        "aligned_500tone_crest_factor_schroeder": cf_schroeder,
        "frequency_maps": fmap_checks,
        "passed": True,
    }


def main() -> None:
    prep_path = ROOT / "preparation.json"
    if prep_path.exists():
        raise FileExistsError("preparation.json already exists; refusing to overwrite frozen instances")

    ext_report = verify_external_references()
    checks = run_self_checks()

    inst_dir = ROOT / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)

    # Master OS seed for OIM solver trajectories
    master_seed = os.urandom(32)
    (inst_dir / "oim-master-seed.bin").write_bytes(master_seed)

    # Generate SK instances: N = 12, 100, 500, 1000
    for n in (12, 100, 500, 1000):
        count = n * (n - 1) // 2
        byte_len = 8 * 2 * ((count + 1) // 2)
        raw = os.urandom(byte_len)
        draw_file = inst_dir / f"sk{n}-draw.bin"
        npy_file = inst_dir / f"sk{n}-couplings.npy"
        draw_file.write_bytes(raw)
        k = draw_sk_from_bytes(n, raw)
        np.save(npy_file, k, allow_pickle=False)

    # Generate Max-Cut instances: N = 12 (|E|=18), 100 (|E|=300), 500 (|E|=1500), 1000 (|E|=3000)
    mc_specs = [(12, 18), (100, 300), (500, 1500), (1000, 3000)]
    for n, n_edges in mc_specs:
        # Generous draw buffer (24 bytes * 8 * n_edges) so random chord rejection never runs short
        raw = os.urandom(24 * 8 * n_edges)
        draw_file = inst_dir / f"maxcut{n}-draw.bin"
        graph_file = inst_dir / f"maxcut{n}-graph.json"
        draw_file.write_bytes(raw)
        g = draw_maxcut_from_bytes(n, n_edges, raw)
        save_json(graph_file, g)

    # Build exact 2^12 = 4096 oracles for N=12 benchmarks
    prob_mc12 = IsingProblem.from_maxcut_json(inst_dir / "maxcut12-graph.json", name="maxcut12")
    prob_sk12 = IsingProblem.from_sk_npy(inst_dir / "sk12-couplings.npy", name="sk12")
    oracle_mc12 = build_exact_oracle(prob_mc12)
    oracle_sk12 = build_exact_oracle(prob_sk12)
    save_json(inst_dir / "maxcut12-oracle.json", oracle_mc12)
    save_json(inst_dir / "sk12-oracle.json", oracle_sk12)

    src_hashes = compute_source_hashes()
    save_json(ROOT / "source-hashes.json", src_hashes)

    artifact_hashes = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in sorted(inst_dir.iterdir())
        if p.is_file()
    }
    prep_summary = {
        "created_by": "prepare.py",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "self_checks": checks,
        "external_reference_checks": ext_report,
        "n12_oracles": {
            "maxcut12": {
                "optimal_cut": oracle_mc12["optimal_cut"],
                "ground_energy": oracle_mc12["ground_energy"],
                "degeneracy": oracle_mc12["degeneracy"],
            },
            "sk12": {
                "ground_energy": oracle_sk12["ground_energy"],
                "degeneracy": oracle_sk12["degeneracy"],
            },
        },
        "artifacts": artifact_hashes,
        "source_hashes": src_hashes,
    }
    save_json(prep_path, prep_summary)
    print(
        f"Prepared frozen instances successfully. "
        f"maxcut12 optimal_cut={oracle_mc12['optimal_cut']}, "
        f"sk12 ground_energy={oracle_sk12['ground_energy']:.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

