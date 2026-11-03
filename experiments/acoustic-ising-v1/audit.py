"""Independent read-only audit of saved OIM artifacts, raw entropy draws, spin witnesses, and external file invariance."""
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
import wave
import numpy as np

from backends.ode_backend import PhaseODEBackend
from backends.real_acoustic import RealAcousticBackend
from frequency_map import Musical12ToneMap, TopOctaveCombMap
from prepare import compute_source_hashes
from problem import (
    EXTERNAL_REFERENCE_HASHES,
    MAXCUT24_DIR,
    ROOT,
    SK24_DIR,
    save_json,
    sha256_file,
)
from run import load_case_problem
from schedule import AnnealingSchedule
from solver import IsingAnnealer, derive_problem_rng


def independent_maxcut_score(graph_path: Path, bitstring: str) -> tuple[float, float]:
    """Pure-Python edge loop scoring independent of problem.py."""
    g = json.loads(graph_path.read_text())
    n = int(g["vertices"])
    if len(bitstring) != n or any(ch not in ("0", "1") for ch in bitstring):
        raise ValueError(f"Invalid bitstring of length {len(bitstring)} for N={n}")
    s = [1 if ch == "0" else -1 for ch in bitstring]
    cut = 0.0
    ising_e = 0.0
    for edge in g["edges"]:
        u, v, w = int(edge["u"]), int(edge["v"]), float(edge["weight"])
        if s[u] != s[v]:
            cut += w
            ising_e -= w
        else:
            ising_e += w
    return ising_e, cut


def independent_sk_score(npy_path: Path, bitstring: str) -> float:
    """Independent upper-triangular quadratic form sum_{i<j} K_ij s_i s_j without problem.py."""
    k = np.load(npy_path, mmap_mode="r")
    n = k.shape[0]
    if len(bitstring) != n or any(ch not in ("0", "1") for ch in bitstring):
        raise ValueError(f"Invalid bitstring of length {len(bitstring)} for N={n}")
    s = np.array([1.0 if ch == "0" else -1.0 for ch in bitstring], dtype=np.float64)
    k_upper = np.triu(np.asarray(k, dtype=np.float64), 1)
    return float(s @ k_upper @ s)


def replay_box_muller_pure_bytes(raw: bytes, n: int) -> np.ndarray:
    """Pure little-endian byte decoding + Box-Muller replay matching tensor-sk-qaoa-v1/audit.py."""
    count = n * (n - 1) // 2
    needed = 8 * 2 * ((count + 1) // 2)
    u = ((np.frombuffer(raw[:needed], dtype="<u8") >> 11).astype(np.float64) + 0.5) / (2.0 ** 53)
    vals = np.empty(len(u), dtype=np.float64)
    r = np.sqrt(-2.0 * np.log(u[0::2]))
    th = 2.0 * math.pi * u[1::2]
    vals[0::2] = r * np.cos(th)
    vals[1::2] = r * np.sin(th)
    return -vals[:count] / math.sqrt(float(n))


def replay_maxcut_pure_python(raw: bytes, n: int, n_edges: int) -> list[dict[str, int]]:
    words = [int.from_bytes(raw[i : i + 8], "little") for i in range(0, len(raw) - (len(raw) % 8), 8)]
    edges: dict[tuple[int, int], int] = {}
    pos = 0
    for q in range(n):
        a, b = (q, (q + 1) % n)
        if a > b:
            a, b = b, a
        edges[(a, b)] = 1 + (words[pos] % 4)
        pos += 1
    while len(edges) < n_edges:
        u = words[pos] % n
        v = words[pos + 1] % n
        w = 1 + (words[pos + 2] % 4)
        pos += 3
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edges:
                edges[(a, b)] = w
    return [{"u": a, "v": b, "weight": edges[(a, b)]} for (a, b) in sorted(edges)]


def main() -> None:
    t0 = time.perf_counter()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, **details: object) -> None:
        entry = {"name": name, "passed": bool(passed), **details}
        checks.append(entry)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}", flush=True)
        if not passed:
            raise AssertionError(f"Audit check failed: {entry}")

    # 1. External read-only reference files untouched
    for ext_path, expected_sha in EXTERNAL_REFERENCE_HASHES.items():
        actual_sha = sha256_file(ext_path)
        check(
            f"external unchanged: {ext_path.name} ({ext_path.parent.name})",
            actual_sha == expected_sha,
            expected=expected_sha,
            actual=actual_sha,
        )

    # 2. Frozen source hashes and preparation artifact hashes
    frozen_src = json.loads((ROOT / "source-hashes.json").read_text())
    check("frozen tracked sources unchanged", compute_source_hashes() == frozen_src)

    prep = json.loads((ROOT / "preparation.json").read_text())
    check(
        "preparation instance artifact hashes",
        all(sha256_file(ROOT / rel) == digest for rel, digest in prep["artifacts"].items()),
    )

    # 3. Structural no-oracle-leakage & speaker safety guard checks
    solver_files = (
        ROOT / "solver.py",
        ROOT / "schedule.py",
        ROOT / "backends" / "ode_backend.py",
        ROOT / "backends" / "audio_loopback.py",
        ROOT / "backends" / "real_acoustic.py",
    )
    forbidden_tokens = ("oracle.json", "-16.548016", "optimal_cut")
    no_leak = all(
        tok not in sf.read_text()
        for sf in solver_files
        for tok in forbidden_tokens
    )
    check("no oracle tokens in solver/schedule/backends", no_leak)

    guard_blocked = False
    try:
        RealAcousticBackend()
    except PermissionError:
        guard_blocked = True
    check("RealAcousticBackend default speaker safety guard", guard_blocked)

    # 4. Replay raw OS byte draws for SK (12, 100, 500, 1000) and Max-Cut (12, 100, 500, 1000)
    inst_dir = ROOT / "instances"
    for n in (12, 100, 500, 1000):
        raw_sk = (inst_dir / f"sk{n}-draw.bin").read_bytes()
        k_saved = np.load(inst_dir / f"sk{n}-couplings.npy", mmap_mode="r")
        replayed = replay_box_muller_pure_bytes(raw_sk, n)
        max_err = float(np.max(np.abs(k_saved[np.triu_indices(n, 1)] - replayed)))
        sym_err = float(np.max(np.abs(k_saved - k_saved.T)))
        diag_err = float(np.max(np.abs(np.diag(k_saved))))
        check(
            f"SK{n} Box-Muller byte replay & symmetry",
            max_err < 1e-14 and sym_err == 0.0 and diag_err == 0.0,
            max_error=max_err,
        )

    for n, n_edges in ((12, 18), (100, 300), (500, 1500), (1000, 3000)):
        raw_mc = (inst_dir / f"maxcut{n}-draw.bin").read_bytes()
        g_saved = json.loads((inst_dir / f"maxcut{n}-graph.json").read_text())
        replayed_edges = replay_maxcut_pure_python(raw_mc, n, n_edges)
        check(
            f"MaxCut{n} pure-Python byte replay ({n_edges} edges)",
            g_saved["edges"] == replayed_edges and int(g_saved["vertices"]) == n,
        )

    # 5. Independent complete 2^12 = 4096 brute-force verification of N=12 oracles
    orc_mc12 = json.loads((inst_dir / "maxcut12-oracle.json").read_text())
    mc12_scores = [
        independent_maxcut_score(inst_dir / "maxcut12-graph.json", format(idx, "012b"))
        for idx in range(1 << 12)
    ]
    min_e_mc12 = min(x[0] for x in mc12_scores)
    max_cut_mc12 = max(x[1] for x in mc12_scores)
    check(
        "MaxCut12 independent 4096-state enumeration",
        abs(min_e_mc12 - float(orc_mc12["ground_energy"])) < 1e-10
        and abs(max_cut_mc12 - float(orc_mc12["optimal_cut"])) < 1e-10,
        optimal_cut=max_cut_mc12,
    )

    orc_sk12 = json.loads((inst_dir / "sk12-oracle.json").read_text())
    sk12_scores = [
        independent_sk_score(inst_dir / "sk12-couplings.npy", format(idx, "012b"))
        for idx in range(1 << 12)
    ]
    min_e_sk12 = min(sk12_scores)
    check(
        "SK12 independent 4096-state enumeration",
        abs(min_e_sk12 - float(orc_sk12["ground_energy"])) < 1e-10,
        ground_energy=min_e_sk12,
    )

    # 6. Independent witness re-scoring across ALL Phase 1 OIM + Greedy runs (N=12..1000)
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    cases_dir = ROOT / "results" / "cases"

    for row in summary["phase1_ode_benchmarks"]:
        cname = str(row["case"])
        oim_case = json.loads((cases_dir / f"{cname}-oim.json").read_text())
        grd_case = json.loads((cases_dir / f"{cname}-greedy.json").read_text())

        if oim_case["kind"] == "maxcut":
            gpath = (MAXCUT24_DIR / "graph.json") if cname == "maxcut24" else (inst_dir / f"{cname}-graph.json")
            e_oim, c_oim = independent_maxcut_score(gpath, str(oim_case["best_bitstring"]))
            e_grd, c_grd = independent_maxcut_score(gpath, str(grd_case["best_bitstring"]))
            check(
                f"{cname} independent witness score (OIM & Greedy)",
                abs(e_oim - float(oim_case["best_energy"])) < 1e-9
                and abs(c_oim - float(oim_case["best_cut"])) < 1e-9
                and abs(e_grd - float(grd_case["best_energy"])) < 1e-9
                and abs(c_grd - float(grd_case["best_cut"])) < 1e-9,
                oim_cut=c_oim,
                greedy_cut=c_grd,
            )
            for pref in oim_case["prefixes"]:
                ep, cp = independent_maxcut_score(gpath, str(pref["best_bitstring"]))
                check(
                    f"{cname} prefix {pref['trajectories']} witness",
                    abs(ep - float(pref["best_energy"])) < 1e-9 and abs(cp - float(pref["best_cut"])) < 1e-9,
                )
        else:
            kpath = (SK24_DIR / "couplings.npy") if cname == "sk24" else (inst_dir / f"{cname}-couplings.npy")
            e_oim = independent_sk_score(kpath, str(oim_case["best_bitstring"]))
            e_grd = independent_sk_score(kpath, str(grd_case["best_bitstring"]))
            check(
                f"{cname} independent witness score (OIM & Greedy)",
                abs(e_oim - float(oim_case["best_energy"])) < 1e-9
                and abs(e_grd - float(grd_case["best_energy"])) < 1e-9,
                oim_energy=e_oim,
                greedy_energy=e_grd,
            )
            for pref in oim_case["prefixes"]:
                ep = independent_sk_score(kpath, str(pref["best_bitstring"]))
                check(
                    f"{cname} prefix {pref['trajectories']} witness",
                    abs(ep - float(pref["best_energy"])) < 1e-9,
                )

        # Verify mean/min consistency with saved 128-trajectory array
        arr = np.array(oim_case["energies"], dtype=np.float64)
        check(
            f"{cname} 128-trajectory statistics & isolated RSS < 256 MiB",
            len(arr) == 128
            and abs(float(np.min(arr)) - float(oim_case["best_energy"])) < 1e-12
            and abs(float(np.mean(arr)) - float(oim_case["mean_energy"])) < 1e-12
            and 0.0 < float(oim_case["peak_rss_mib"]) < 256.0,
            peak_rss_mib=float(oim_case["peak_rss_mib"]),
        )

    # Verify exact 24-spin benchmarks against read-only HexSolid targets
    mc24_row = next(r for r in summary["phase1_ode_benchmarks"] if r["case"] == "maxcut24")
    sk24_row = next(r for r in summary["phase1_ode_benchmarks"] if r["case"] == "sk24")
    check("MaxCut24 exact optimum 62 reached in Phase 1", abs(float(mc24_row["oim"]["best_cut"]) - 62.0) < 1e-10)
    check(
        "SK24 exact ground -16.548016339179107 reached in Phase 1",
        abs(float(sk24_row["oim"]["best_energy"]) - (-16.548016339179107)) < 1e-10,
    )

    # 7. Independent witness re-scoring of Phase 2 AudioLoopback swaps + WAV verification
    for lb_row in summary["phase2_audio_loopback_swaps"]:
        sid = str(lb_row["swap_id"])
        cname = str(lb_row["case"])
        lb_case = json.loads((cases_dir / f"{sid}.json").read_text())
        if lb_case["kind"] == "maxcut":
            gpath = (MAXCUT24_DIR / "graph.json") if cname == "maxcut24" else (inst_dir / f"{cname}-graph.json")
            e_lb, c_lb = independent_maxcut_score(gpath, str(lb_case["best_bitstring"]))
            check(
                f"{sid} independent witness score",
                abs(e_lb - float(lb_case["best_energy"])) < 1e-9
                and abs(c_lb - float(lb_case["best_cut"])) < 1e-9,
                best_cut=c_lb,
            )
        else:
            kpath = (SK24_DIR / "couplings.npy") if cname == "sk24" else (inst_dir / f"{cname}-couplings.npy")
            e_lb = independent_sk_score(kpath, str(lb_case["best_bitstring"]))
            check(
                f"{sid} independent witness score",
                abs(e_lb - float(lb_case["best_energy"])) < 1e-9,
                best_energy=e_lb,
            )

    wav_path = ROOT / "results" / "maxcut12-chord-evolution.wav"
    with wave.open(str(wav_path), "rb") as wf:
        nch, sw, sr, nframes = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        pcm = np.frombuffer(wf.readframes(nframes), dtype="<i2").astype(np.float64)
    fmap12 = Musical12ToneMap(12)
    last_frame = pcm[-fmap12.frame_samples :]
    spec_last = np.fft.rfft(last_frame)
    tone_power = float(np.sum(np.abs(spec_last[fmap12.bins]) ** 2))
    total_power = float(np.sum(np.abs(spec_last) ** 2))
    check(
        "Exported 12-tone WAV format & spectral concentration on C5..B5 bins",
        nch == 1 and sw == 2 and sr == 48000 and nframes == 41 * 4800 and (tone_power / max(total_power, 1e-12)) > 0.999,
        spectral_purity=tone_power / max(total_power, 1e-12),
    )

    # 8. Deterministic trajectory replay on maxcut24 and sk24
    master_seed = (inst_dir / "oim-master-seed.bin").read_bytes()
    for cname in ("maxcut24", "sk24"):
        prob = load_case_problem(cname)
        annealer = IsingAnnealer(AnnealingSchedule(), PhaseODEBackend(), integrator="heun")
        rng = derive_problem_rng(master_seed, cname, extra_tag="phase1_ode")
        replayed_res = annealer.solve(prob, rng=rng, batch_size=128)
        saved_case = json.loads((cases_dir / f"{cname}-oim.json").read_text())
        diff = float(np.max(np.abs(replayed_res.energies - np.array(saved_case["energies"], dtype=np.float64))))
        check(f"{cname} deterministic 128-trajectory replay", diff < 1e-12, max_energy_diff=diff)

    # Validate TopOctaveCombMap up to N=1000 harmonic-free & bin orthogonality
    comb1000 = TopOctaveCombMap(1000).validate()
    check("TopOctaveCombMap(1000) harmonic-free & >=2-bin spacing", bool(comb1000["passed"]) and int(comb1000["min_bin_gap"]) >= 2)

    artifact_hashes = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and p.name not in ("artifact-audit.json", "REPORT.md", "README.md") and "__pycache__" not in p.parts
    }
    audit_out = {
        "passed": all(bool(c["passed"]) for c in checks),
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "seconds": time.perf_counter() - t0,
        "checks": checks,
        "artifact_hashes": artifact_hashes,
    }
    save_json(ROOT / "artifact-audit.json", audit_out)
    print(f"Audit PASSED: {audit_out['passed_checks']}/{audit_out['total_checks']} checks verified.", flush=True)


if __name__ == "__main__":
    main()

