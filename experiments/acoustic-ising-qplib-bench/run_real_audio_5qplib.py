"""Benchmark Real Audio New Method on the 5 troublesome QPLIB instances.

Instances:
  - QPLIB_3506 (N=496)
  - QPLIB_3706 (N=703)
  - QPLIB_3838 (N=780)
  - QPLIB_3877 (N=630)
  - QPLIB_5725 (N=343)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

THIS_DIR = pathlib.Path(__file__).resolve().parent
GSET_DIR = THIS_DIR.parent / "acoustic-ising-gset-bench"
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
LIVE_DIR = THIS_DIR.parent / "acoustic-ising-live-test"

for p in (THIS_DIR, GSET_DIR, V1_DIR, LIVE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qplib_problem import QPLIBProblem, get_peak_rss_bytes
from frequency_map import TopOctaveCombMap
from duffing_bench import SkirtedAcousticBackend

TARGET_QIDS = [3506, 3706, 3838, 3877, 5725]
DATA_DIR = THIS_DIR / "data"
RESULTS_DIR = THIS_DIR / "results"


def run_real_audio_instance(
    qid: int,
    n_steps: int = 70,
    dt: float = 0.30,
    gain: float = 0.20,
    in_dev: int = 15,
    out_dev: int = 14,
    seed: int = 42,
) -> dict:
    qp = DATA_DIR / f"QPLIB_{qid}.qplib"
    sp = DATA_DIR / f"QPLIB_{qid}.sol"
    prob = QPLIBProblem.load(qp, sp)
    n = prob.n_vars
    m_frame = 12000  # 0.25 s per frame

    fmap = TopOctaveCombMap(
        n_spins=n,
        sample_rate=48000,
        frame_samples=m_frame,
        f_min_hz=6000.0,
        f_max_hz=11000.0,
    )

    backend = SkirtedAcousticBackend(
        frequency_map=fmap,
        allow_speaker_output=True,
        output_gain=gain,
        cal_averages=2,
        input_device=in_dev,
        output_device=out_dev,
        same_clock=True,
        out_channel="l",
        n_guard_each_side=160,
    )

    rng = np.random.default_rng(seed)
    backend.reset(n, rng)

    u = rng.uniform(-0.05, 0.05, size=n)
    v = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    for step in range(n_steps):
        pr = float(step) / float(max(n_steps - 1, 1))
        pump = 0.2 + 0.85 * pr
        damp = 0.50
        c_gain = 0.55 * pr
        beta = 1.0 + 4.0 * (pr ** 2)

        m_ideal = np.tanh(beta * u)
        phi = np.arccos(np.clip(m_ideal, -0.999999, 0.999999))
        z_rx = backend.measure_phasors(phi)
        m_hat = z_rx.real

        force = -c_gain * (prob.K_norm @ m_hat + prob.h_norm)
        v += (-damp * v + (pump - 0.8) * u - (u ** 3) + force) * dt
        u += v * dt

    s = np.where(u >= 0.0, 1.0, -1.0)
    s_ref = prob.greedy_refine(s)
    obj = prob.eval_s(s_ref)
    elapsed = time.perf_counter() - t0
    opt_ratio = (obj / prob.opt_obj * 100.0) if prob.opt_obj else 0.0
    peak_rss = get_peak_rss_bytes() / (1024.0 * 1024.0)

    return {
        "qid": qid,
        "n_vars": n,
        "opt_obj": prob.opt_obj,
        "obj": float(obj),
        "opt_ratio_pct": float(opt_ratio),
        "seconds": float(elapsed),
        "peak_rss_mib": float(peak_rss),
        "rx_peak": float(backend.last_rx_peak),
        "cal_snr_db": float(backend.cal_snr_median_db),
        "cal_phase_std_deg": float(backend.cal_phase_std_deg),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "real_audio_5qplib_summary.json"

    print("=" * 80)
    print(" Real Audio New Method: 5 Troublesome QPLIB Instances Benchmark")
    print(" (L-only Skirted Top-Octave Comb, 6.0-11.0 kHz, WASAPI Dev 14/15)")
    print("=" * 80)

    results = []
    t_start = time.perf_counter()
    for i, qid in enumerate(TARGET_QIDS, 1):
        print(f"\n[{i}/5] Running Real Audio on QPLIB_{qid}...")
        res = run_real_audio_instance(qid)
        print(
            f"  Result: Obj={res['obj']:12.1f} / {res['opt_obj']:12.1f} "
            f"({res['opt_ratio_pct']:6.2f}%) | Time={res['seconds']:5.1f}s | "
            f"SNR={res['cal_snr_db']:4.1f}dB | Peak={res['rx_peak']:.3f} | RSS={res['peak_rss_mib']:5.1f}MiB"
        )
        results.append(res)

    total_time = time.perf_counter() - t_start
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "total_seconds": total_time,
                "instances": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f" Real Audio Benchmark Complete! Saved to {out_json}")
    print(f" Total elapsed: {total_time:.1f} s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

