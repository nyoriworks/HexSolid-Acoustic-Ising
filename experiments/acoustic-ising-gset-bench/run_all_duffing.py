"""Master orchestration script for Duffing-Parametron G-set benchmarks across Virtual and Real Audio."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results_duffing"
PY_EXE = sys.executable

ALL_INSTANCES = ("G1", "G11", "G14", "G22", "G55", "G67", "G70")


def run_benchmark_suite(
    instances: list[str],
    solvers: list[str],
    batches: list[int],
    in_dev: int = 15,
    out_dev: int = 14,
    gain: float = 0.20,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
        env[k] = "1"

    total_runs = len(instances) * len(solvers) * len(batches)
    current = 0

    for inst in instances:
        for solver in solvers:
            for b in batches:
                current += 1
                out_json = RESULTS_DIR / f"{inst}__{solver}__b{b}.json"
                print(f"\n>>> [{current}/{total_runs}] Running {inst} | {solver} | B={b} ...", flush=True)
                cmd = [
                    PY_EXE,
                    str(THIS_DIR / "duffing_bench.py"),
                    "--instance", inst,
                    "--solver", solver,
                    "--batch", str(b),
                    "--out-json", str(out_json),
                    "--in-dev", str(in_dev),
                    "--out-dev", str(out_dev),
                    "--gain", str(gain),
                ]
                subprocess.run(cmd, env=env, check=True)

    # Merge into summary_duffing.json
    merged = []
    for p in sorted(RESULTS_DIR.glob("*__*__b*.json")):
        merged.append(json.loads(p.read_text()))
    (RESULTS_DIR / "summary_duffing.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nAll finished! Saved {len(merged)} records to {RESULTS_DIR / 'summary_duffing.json'}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=list(ALL_INSTANCES))
    parser.add_argument("--solvers", nargs="+", default=["virtual_audio"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--in-dev", type=int, default=15)
    parser.add_argument("--out-dev", type=int, default=14)
    parser.add_argument("--gain", type=float, default=0.20)
    args = parser.parse_args()

    run_benchmark_suite(
        instances=args.instances,
        solvers=args.solvers,
        batches=args.batches,
        in_dev=args.in_dev,
        out_dev=args.out_dev,
        gain=args.gain,
    )

