"""Test 100% classical public-domain zero-extra-memory OIM / Duffing / Daido-Fourier dynamics on G-set."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
for p in (THIS_DIR, V1_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gset_loader import load_gset_problem  # noqa: E402
from solver import derive_problem_rng  # noqa: E402


def solve_daido_phase_oim(problem, rng, n_steps=800, batch_size=32, dt=0.12):
    """
    1-variable Phase OIM with Daido (1992) / Fourier square-wave 1st+3rd harmonic coupling:
      sin(phi_i - phi_j) + alpha * sin(3*(phi_i - phi_j))
    + per-node degree-balanced SHIL. 100% 1-variable phase-only (compatible with unit-phasor audio!).
    """
    t0 = time.perf_counter()
    n = problem.n_spins
    rsq = np.asarray(problem.couplings.multiply(problem.couplings).sum(axis=1)).ravel()
    deg_rms = np.sqrt(np.maximum(rsq, 1e-12))
    # Gentle degree balancing (blend global j_scale and local deg_rms)
    inv_node_scale = 1.0 / (0.5 * problem.j_scale + 0.5 * deg_rms)

    shape = (batch_size, n) if batch_size > 1 else (n,)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    for step in range(n_steps):
        u = float(step) / float(max(n_steps - 1, 1))
        c1, s1 = np.cos(phi), np.sin(phi)
        # Exact Chebyshev 3rd harmonic without extra trig calls: cos(3x)=4c^3-3c, sin(3x)=3s-4s^3
        c3 = 4.0 * (c1 * c1 * c1) - 3.0 * c1
        s3 = 3.0 * s1 - 4.0 * (s1 * s1 * s1)
        alpha = 0.28 * u  # Gradually sharpen sine into smooth trapezoid (Fourier 1822 / Daido 1992)
        torque1 = s1 * problem.matvec(c1) - c1 * problem.matvec(s1)
        torque3 = s3 * problem.matvec(c3) - c3 * problem.matvec(s3)
        kc = 1.0 + 0.8 * u
        osc = math.sin(math.pi * 4.0 * min(u / 0.90, 1.0)) ** 2
        ks = (0.05 + 0.45 * (u ** 1.5)) + 0.65 * math.sin(math.pi * min(u / 0.95, 1.0)) * osc
        sig = 0.22 * ((1.0 - u) ** 1.8) if u < 1.0 else 0.0
        dw = (sig * math.sqrt(dt)) * rng.standard_normal(shape) if sig > 0 else 0.0
        dphi = kc * inv_node_scale * (torque1 + alpha * torque3) - (2.0 * ks) * (s1 * c1)
        phi = np.mod(phi + dt * dphi + dw, 2.0 * math.pi)
    spins = np.where(np.cos(phi) >= 0.0, 1.0, -1.0)
    cuts = np.atleast_1d(np.asarray(problem.cut_value(spins), dtype=np.float64))
    return float(np.max(cuts)), time.perf_counter() - t0


def solve_duffing_rlc(problem, rng, n_steps=500, batch_size=32, dt=0.35):
    """
    1918 Georg Duffing / 1954 Goto Parametron damped cubic RLC oscillator:
      dv/dt = -gamma(t)*v + (p(t) - 1)*u - u^3 - c0 * K @ tanh(beta * u)
      du/dt = v
    100% smooth cubic restoring potential u^4/4 (ZERO walls, ZERO resets!), ONLY 1 matvec/step!
    """
    t0 = time.perf_counter()
    n = problem.n_spins
    rsq = np.asarray(problem.couplings.multiply(problem.couplings).sum(axis=1)).ravel()
    c0 = 0.65 / float(np.sqrt(np.mean(rsq)))
    shape = (batch_size, n) if batch_size > 1 else (n,)
    u = rng.normal(0.0, 0.05, size=shape)
    v = np.zeros_like(u)
    for step in range(n_steps):
        pr = float(step) / float(max(n_steps - 1, 1))
        pump = 1.8 * pr  # Parametron subharmonic pump (Eiichi Goto 1954!)
        damp = 0.45 * (1.0 - 0.75 * pr)
        beta = 1.0 + 6.0 * (pr ** 2)
        f = -c0 * problem.matvec(np.tanh(beta * u))
        v = v + (-damp * v + (pump - 0.8) * u - (u * u * u) + f) * dt
        u = u + v * dt
    spins = np.where(u >= 0.0, 1.0, -1.0)
    cuts = np.atleast_1d(np.asarray(problem.cut_value(spins), dtype=np.float64))
    return float(np.max(cuts)), time.perf_counter() - t0


def main() -> None:
    manifest = json.loads((THIS_DIR / "data" / "manifest.json").read_text())
    seed = hashlib.sha256(b"HexSolid-Gset-Public-Benchmark-v1").digest()
    print("=== Comparison: Daido Phase OIM vs 1918/1954 Duffing-Parametron (1 matvec/step, NO walls) ===")
    for g in ("G1", "G11", "G14", "G22", "G55", "G67", "G70"):
        prob = load_gset_problem(THIS_DIR / "data" / f"{g}.txt", g)
        bks = float(manifest[g]["bks"])
        c_d1, t_d1 = solve_duffing_rlc(prob, derive_problem_rng(seed, g, "duf1"), n_steps=110, batch_size=1)
        c_d32, t_d32 = solve_duffing_rlc(prob, derive_problem_rng(seed, g, "duf32"), n_steps=500, batch_size=32)
        print(
            f"  {g:3s} (BKS={bks:6.0f}): Duffing-Parametron 1-traj={c_d1:7.0f} ({100*c_d1/bks:6.2f}%, {t_d1:.3f}s) | "
            f"32-traj={c_d32:7.0f} ({100*c_d32/bks:6.2f}%, {t_d32:.3f}s)"
        )


if __name__ == "__main__":
    main()

