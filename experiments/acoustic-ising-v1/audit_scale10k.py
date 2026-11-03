"""Independent O(N)-memory streaming audit of N = 2000, 3000, 5000, 10000 witnesses + v1/external hash invariance."""
from __future__ import annotations

import os
import sys

_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
if any(os.environ.get(k) != "1" for k in _THREAD_VARS):
    env = dict(os.environ)
    for k in _THREAD_VARS:
        env[k] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import json
import math
from pathlib import Path
import time
import numpy as np

from frequency_map import TopOctaveCombMap
from prepare import compute_source_hashes
from prepare_scale10k import generate_maxcut_edges_from_seed, rng_from_hex_seed
from problem import EXTERNAL_REFERENCE_HASHES, ROOT, save_json, sha256_file


def stream_sk_energy_o_n_memory(n: int, seed_hex: str, bitstring: str) -> float:
    """
    Independently compute sum_{i<j} K_ij s_i s_j by streaming one row at a time from PCG64(seed).
    Peak memory is O(N) (< 0.2 MiB even at N = 10,000) without allocating an N x N matrix.
    """
    if len(bitstring) != n or any(ch not in ("0", "1") for ch in bitstring):
        raise ValueError(f"Invalid bitstring length {len(bitstring)} for N={n}")
    s = np.array([1.0 if ch == "0" else -1.0 for ch in bitstring], dtype=np.float64)
    rng = rng_from_hex_seed(seed_hex)
    scale = -1.0 / math.sqrt(float(n))
    total = 0.0
    for i in range(n - 1):
        row = rng.standard_normal(n - 1 - i) * scale
        total += float(s[i]) * float(np.dot(row, s[i + 1 :]))
    return total


def independent_seeded_maxcut_score(n: int, n_edges: int, seed_hex: str, bitstring: str) -> tuple[float, float]:
    if len(bitstring) != n or any(ch not in ("0", "1") for ch in bitstring):
        raise ValueError(f"Invalid bitstring length {len(bitstring)} for N={n}")
    s = [1 if ch == "0" else -1 for ch in bitstring]
    edges = generate_maxcut_edges_from_seed(n, n_edges, seed_hex)
    cut = 0.0
    ising_e = 0.0
    for u, v, w in edges:
        if s[u] != s[v]:
            cut += w
            ising_e -= w
        else:
            ising_e += w
    return ising_e, cut


def main() -> None:
    t0 = time.perf_counter()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, **details: object) -> None:
        entry = {"name": name, "passed": bool(passed), **details}
        checks.append(entry)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}", flush=True)
        if not passed:
            raise AssertionError(f"Scale10k audit failed: {entry}")

    # 1. External read-only HexSolid files untouched
    for ext_path, expected_sha in EXTERNAL_REFERENCE_HASHES.items():
        check(
            f"external unchanged: {ext_path.name} ({ext_path.parent.name})",
            sha256_file(ext_path) == expected_sha,
        )

    # 2. v1 core tracked sources & v1 preparation artifacts untouched
    v1_src = json.loads((ROOT / "source-hashes.json").read_text())
    check("v1 frozen core sources unchanged", compute_source_hashes() == v1_src)
    v1_prep = json.loads((ROOT / "preparation.json").read_text())
    check(
        "v1 preparation artifacts unchanged",
        all(sha256_file(ROOT / rel) == d for rel, d in v1_prep["artifacts"].items()),
    )

    # 3. Verify all Phase 1 (N = 2000, 3000, 5000, 10000) OIM & Greedy witnesses independently
    seeds_doc = json.loads((ROOT / "instances_scale10k" / "seeds.json").read_text())
    summary = json.loads((ROOT / "results_scale10k" / "summary.json").read_text())
    cases_dir = ROOT / "results_scale10k" / "cases"

    for row in summary["phase1_ode_scale10k"]:
        cname = str(row["case"])
        meta = seeds_doc["instances"][cname]
        n = int(meta["n_spins"])
        seed_hex = str(meta["seed_hex"])
        oim_case = json.loads((cases_dir / f"{cname}-oim.json").read_text())
        grd_case = json.loads((cases_dir / f"{cname}-greedy.json").read_text())

        if meta["kind"] == "maxcut":
            n_edges = int(meta["n_edges"])
            e_oim, c_oim = independent_seeded_maxcut_score(n, n_edges, seed_hex, str(oim_case["best_bitstring"]))
            e_grd, c_grd = independent_seeded_maxcut_score(n, n_edges, seed_hex, str(grd_case["best_bitstring"]))
            check(
                f"{cname} independent edge-loop witness (OIM & Greedy)",
                abs(e_oim - float(oim_case["best_energy"])) < 1e-8
                and abs(c_oim - float(oim_case["best_cut"])) < 1e-8
                and abs(e_grd - float(grd_case["best_energy"])) < 1e-8
                and abs(c_grd - float(grd_case["best_cut"])) < 1e-8,
                oim_cut=c_oim,
                greedy_cut=c_grd,
            )
        else:
            e_oim = stream_sk_energy_o_n_memory(n, seed_hex, str(oim_case["best_bitstring"]))
            e_grd = stream_sk_energy_o_n_memory(n, seed_hex, str(grd_case["best_bitstring"]))
            check(
                f"{cname} O(N)-streamed SK witness (OIM & Greedy)",
                abs(e_oim - float(oim_case["best_energy"])) < 1e-7
                and abs(e_grd - float(grd_case["best_energy"])) < 1e-7,
                oim_energy=e_oim,
                greedy_energy=e_grd,
            )

        arr = np.array(oim_case["energies"], dtype=np.float64)
        check(
            f"{cname} trajectory stats consistency",
            abs(float(np.min(arr)) - float(oim_case["best_energy"])) < 1e-10
            and abs(float(np.mean(arr)) - float(oim_case["mean_energy"])) < 1e-10,
        )

    # 4. Verify all Phase 2 AudioLoopback witnesses & harmonic-free comb properties up to N=10000
    for lb_row in summary["phase2_loopback_scale10k"]:
        sid = str(lb_row["swap_id"])
        cname = str(lb_row["case"])
        meta = seeds_doc["instances"][cname]
        n = int(meta["n_spins"])
        seed_hex = str(meta["seed_hex"])
        lb_case = json.loads((cases_dir / f"{sid}.json").read_text())
        fm_info = lb_case["frequency_map"]
        fmap_recheck = TopOctaveCombMap(n, frame_samples=int(fm_info["frame_samples"])).validate()
        check(
            f"{sid} harmonic-free comb validation (>=2-bin spacing)",
            bool(fmap_recheck["passed"]) and int(fmap_recheck["min_bin_gap"]) >= 2,
            min_bin_gap=fmap_recheck["min_bin_gap"],
            bin_hz=fmap_recheck["bin_hz"],
        )
        if meta["kind"] == "maxcut":
            e_lb, c_lb = independent_seeded_maxcut_score(
                n, int(meta["n_edges"]), seed_hex, str(lb_case["best_bitstring"])
            )
            check(
                f"{sid} independent witness score",
                abs(e_lb - float(lb_case["best_energy"])) < 1e-8
                and abs(c_lb - float(lb_case["best_cut"])) < 1e-8,
                best_cut=c_lb,
            )
        else:
            e_lb = stream_sk_energy_o_n_memory(n, seed_hex, str(lb_case["best_bitstring"]))
            check(
                f"{sid} independent O(N)-streamed witness score",
                abs(e_lb - float(lb_case["best_energy"])) < 1e-7,
                best_energy=e_lb,
            )

    out = {
        "passed": all(bool(c["passed"]) for c in checks),
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "seconds": time.perf_counter() - t0,
        "checks": checks,
    }
    save_json(ROOT / "artifact-audit-scale10k.json", out)
    print(f"Scale10k Audit PASSED: {out['passed_checks']}/{out['total_checks']} checks verified.", flush=True)


if __name__ == "__main__":
    main()

