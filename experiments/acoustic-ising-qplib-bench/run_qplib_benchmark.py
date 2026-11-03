"""Full Hans Mittelmann QUBO / QPLIB Benchmark Runner.

Benchmarks all 23 unconstrained binary non-convex instances:
3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738, 3745, 3822,
3832, 3838, 3850, 3852, 3877, 5721, 5725, 5755, 5875, 5881,
5882, 5909, 5922

Compares:
1. デジタル新方式 (Duffing-Parametron, B=1 and B=32)
2. Simulated Bifurcation (dSB, B=1 and B=32)
3. Simulated Annealing (SA, B=1)

Measures:
- 速度 (Execution time in seconds)
- 精度 (Objective value / Proven optimal ratio %)
- メモリ使用量 (Win32 PeakWorkingSetSize Peak RSS in MiB)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

# Force single-threaded BLAS for strict algorithmic comparison
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from qplib_problem import QPLIBProblem, get_peak_rss_bytes
from solvers import solve_duffing, solve_dsb, solve_sa

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

QPLIB_IDS = [
    3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738, 3745, 3822,
    3832, 3838, 3850, 3852, 3877, 5721, 5725, 5755, 5875, 5881,
    5882, 5909, 5922,
]

# Official Mittelmann benchmark results from http://plato.asu.edu/ftp/qubo.html
# (Run on AMD Ryzen 9 5900X, 12 cores, 128GB RAM; 't' = >3600s timeout)
MITTELMANN_OFFICIAL_TIMES = {
    3506: {"BARON": 80, "SCIP": "t", "MCSPARSE": 407, "QUBOWL": 1, "BIQBIN": "t", "SHOT": 274, "COPT": 251, "QUPLANE": 6},
    3565: {"BARON": 3, "SCIP": 42, "MCSPARSE": 13, "QUBOWL": 1, "BIQBIN": 12, "SHOT": 2, "COPT": 1, "QUPLANE": 2},
    3642: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 4, "BIQBIN": "f", "SHOT": "t", "COPT": "t", "QUPLANE": 214},
    3650: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 2, "BIQBIN": "t", "SHOT": "t", "COPT": "t", "QUPLANE": 92},
    3693: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 10, "BIQBIN": "f", "SHOT": "t", "COPT": "t", "QUPLANE": 747},
    3705: {"BARON": 5, "SCIP": 68, "MCSPARSE": 20, "QUBOWL": 1, "BIQBIN": 232, "SHOT": 4, "COPT": 1, "QUPLANE": 3},
    3706: {"BARON": 833, "SCIP": "t", "MCSPARSE": 2418, "QUBOWL": 1, "BIQBIN": "t", "SHOT": "t", "COPT": "t", "QUPLANE": 16},
    3738: {"BARON": 84, "SCIP": 2905, "MCSPARSE": 137, "QUBOWL": 1, "BIQBIN": "t", "SHOT": 49, "COPT": 1, "QUPLANE": 8},
    3745: {"BARON": 29, "SCIP": 584, "MCSPARSE": 55, "QUBOWL": 1, "BIQBIN": "t", "SHOT": 10, "COPT": 28, "QUPLANE": 2},
    3822: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 1, "BIQBIN": "t", "SHOT": "t", "COPT": "t", "QUPLANE": 51},
    3832: {"BARON": 169, "SCIP": "t", "MCSPARSE": 465, "QUBOWL": 1, "BIQBIN": "t", "SHOT": 368, "COPT": 2, "QUPLANE": 8},
    3838: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 1, "BIQBIN": "t", "SHOT": "t", "COPT": "t", "QUPLANE": 25},
    3850: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 24, "BIQBIN": "f", "SHOT": "t", "COPT": "t", "QUPLANE": 1358},
    3852: {"BARON": 2, "SCIP": 24, "MCSPARSE": 36, "QUBOWL": 1, "BIQBIN": 411, "SHOT": 2, "COPT": 6, "QUPLANE": 2},
    3877: {"BARON": 210, "SCIP": "t", "MCSPARSE": 448, "QUBOWL": 1, "BIQBIN": "t", "SHOT": 906, "COPT": 1688, "QUPLANE": 11},
    5721: {"BARON": 18, "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 1, "BIQBIN": 2026, "SHOT": 1, "COPT": 1, "QUPLANE": 4},
    5725: {"BARON": 11, "SCIP": 104, "MCSPARSE": 4, "QUBOWL": 1, "BIQBIN": 39, "SHOT": 7, "COPT": 21, "QUPLANE": 226},
    5755: {"BARON": 2, "SCIP": 2, "MCSPARSE": 1, "QUBOWL": 1, "BIQBIN": 1317, "SHOT": 1, "COPT": 1, "QUPLANE": 52},
    5875: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 721, "BIQBIN": "t", "SHOT": "t", "COPT": "t", "QUPLANE": 900},
    5881: {"BARON": 442, "SCIP": 2967, "MCSPARSE": 75, "QUBOWL": 9, "BIQBIN": 10, "SHOT": 61, "COPT": 128, "QUPLANE": 176},
    5882: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 145, "BIQBIN": 232, "SHOT": "t", "COPT": "t", "QUPLANE": 422},
    5909: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": 1281, "BIQBIN": 1842, "SHOT": "t", "COPT": "t", "QUPLANE": 1293},
    5922: {"BARON": "t", "SCIP": "t", "MCSPARSE": "t", "QUBOWL": "t", "BIQBIN": "t", "SHOT": "t", "COPT": "t", "QUPLANE": "t"},
}


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "qplib_benchmark_summary.json"

    print("=" * 80)
    print(" Hans Mittelmann QUBO / QPLIB Benchmark (23 Instances)")
    print(" Comparing: デジタル新方式 (Duffing-Parametron) vs dSB vs SA")
    print("=" * 80)

    # Automatically fetch benchmark files from official ZIB QPLIB repository if missing
    missing_files = [qid for qid in QPLIB_IDS if not (DATA_DIR / f"QPLIB_{qid}.qplib").exists()]
    if missing_files:
        print(f"Missing {len(missing_files)} QPLIB benchmark files locally.")
        print("Fetching directly from official ZIB archive (https://qplib.zib.de/)...")
        try:
            import download_qplib
            download_qplib.main()
        except Exception as ex:
            print(f"Auto-download failed: {ex}. Proceeding with available files.")

    summary_records = []
    t_start_all = time.perf_counter()

    for i, qid in enumerate(QPLIB_IDS, 1):
        qp = DATA_DIR / f"QPLIB_{qid}.qplib"
        sp = DATA_DIR / f"QPLIB_{qid}.sol"

        if not qp.exists():
            print(f"[{i:02d}/23] QPLIB_{qid} NOT FOUND, skipping.")
            continue

        prob = QPLIBProblem.load(qp, sp)
        quad_nz = prob.Q_sym.nnz // 2
        density = (2.0 * quad_nz / (prob.n_vars * (prob.n_vars - 1))) * 100.0 if prob.n_vars > 1 else 0.0

        print(f"\n[{i:02d}/23] QPLIB_{qid}: N={prob.n_vars}, QuadNZ={quad_nz} (Dens={density:5.2f}%), OptObj={prob.opt_obj}")

        # Run algorithms:
        # 1. デジタル新方式 B=1
        res_duff1 = solve_duffing(prob, steps=200, batch_size=1, seed=42)
        rss_duff1 = get_peak_rss_bytes() / (1024.0 * 1024.0)
        res_duff1["peak_rss_mib"] = rss_duff1

        # 2. dSB B=1
        res_dsb1 = solve_dsb(prob, steps=200, batch_size=1, seed=42)
        rss_dsb1 = get_peak_rss_bytes() / (1024.0 * 1024.0)
        res_dsb1["peak_rss_mib"] = rss_dsb1

        # 3. SA B=1
        res_sa1 = solve_sa(prob, sweeps=200, seed=42)
        rss_sa1 = get_peak_rss_bytes() / (1024.0 * 1024.0)
        res_sa1["peak_rss_mib"] = rss_sa1

        # 4. デジタル新方式 B=32 (Multi-start ensemble)
        res_duff32 = solve_duffing(prob, steps=200, batch_size=32, seed=42)
        rss_duff32 = get_peak_rss_bytes() / (1024.0 * 1024.0)
        res_duff32["peak_rss_mib"] = rss_duff32

        # 5. dSB B=32 (Multi-start ensemble)
        res_dsb32 = solve_dsb(prob, steps=200, batch_size=32, seed=42)
        rss_dsb32 = get_peak_rss_bytes() / (1024.0 * 1024.0)
        res_dsb32["peak_rss_mib"] = rss_dsb32

        # Print comparison line
        print(f"  Duffing B=1:  Obj={res_duff1['objective']:12.1f} ({res_duff1['opt_ratio_pct']:6.2f}%) | Time={res_duff1['seconds']*1000:6.1f} ms | RSS={rss_duff1:5.1f} MiB")
        print(f"  dSB B=1:      Obj={res_dsb1['objective']:12.1f} ({res_dsb1['opt_ratio_pct']:6.2f}%) | Time={res_dsb1['seconds']*1000:6.1f} ms | RSS={rss_dsb1:5.1f} MiB")
        print(f"  SA B=1:       Obj={res_sa1['objective']:12.1f} ({res_sa1['opt_ratio_pct']:6.2f}%) | Time={res_sa1['seconds']*1000:6.1f} ms | RSS={rss_sa1:5.1f} MiB")
        print(f"  Duffing B=32: Obj={res_duff32['objective']:12.1f} ({res_duff32['opt_ratio_pct']:6.2f}%) | Time={res_duff32['seconds']*1000:6.1f} ms | RSS={rss_duff32:5.1f} MiB")
        print(f"  dSB B=32:     Obj={res_dsb32['objective']:12.1f} ({res_dsb32['opt_ratio_pct']:6.2f}%) | Time={res_dsb32['seconds']*1000:6.1f} ms | RSS={rss_dsb32:5.1f} MiB")

        record = {
            "qid": qid,
            "name": prob.name,
            "n_vars": prob.n_vars,
            "quad_nz": quad_nz,
            "density_pct": density,
            "sense": prob.sense,
            "opt_obj": prob.opt_obj,
            "mittelmann_times": MITTELMANN_OFFICIAL_TIMES.get(qid, {}),
            "results": {
                "duffing_b1": {
                    "objective": res_duff1["objective"],
                    "opt_ratio_pct": res_duff1["opt_ratio_pct"],
                    "seconds": res_duff1["seconds"],
                    "peak_rss_mib": res_duff1["peak_rss_mib"],
                },
                "dsb_b1": {
                    "objective": res_dsb1["objective"],
                    "opt_ratio_pct": res_dsb1["opt_ratio_pct"],
                    "seconds": res_dsb1["seconds"],
                    "peak_rss_mib": res_dsb1["peak_rss_mib"],
                },
                "sa_b1": {
                    "objective": res_sa1["objective"],
                    "opt_ratio_pct": res_sa1["opt_ratio_pct"],
                    "seconds": res_sa1["seconds"],
                    "peak_rss_mib": res_sa1["peak_rss_mib"],
                },
                "duffing_b32": {
                    "objective": res_duff32["objective"],
                    "opt_ratio_pct": res_duff32["opt_ratio_pct"],
                    "seconds": res_duff32["seconds"],
                    "peak_rss_mib": res_duff32["peak_rss_mib"],
                },
                "dsb_b32": {
                    "objective": res_dsb32["objective"],
                    "opt_ratio_pct": res_dsb32["opt_ratio_pct"],
                    "seconds": res_dsb32["seconds"],
                    "peak_rss_mib": res_dsb32["peak_rss_mib"],
                },
            },
        }
        summary_records.append(record)

    total_time = time.perf_counter() - t_start_all

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "total_instances": len(summary_records),
                "total_bench_seconds": total_time,
                "instances": summary_records,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f" Benchmark Complete! Saved to {out_json}")
    print(f" Total elapsed: {total_time:.2f} s across {len(summary_records)} instances.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

