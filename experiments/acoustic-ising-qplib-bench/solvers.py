"""Solvers for QPLIB / QUBO Benchmarks.

Implements:
1. デジタル新方式 (duffing_digital): Parametron / Duffing 2nd-order non-linear dynamics
   (Eiichi Goto 1954 / Georg Duffing 1918, classical physics models, 1 matvec/step)
2. Simulated Bifurcation (dSB): Discrete Kerr-bifurcation Hamiltonian dynamics (Goto et al. 2021)
3. Simulated Annealing (SA): Chromatic / Metropolis sweeps with O(deg) incremental updates
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

from qplib_problem import QPLIBProblem


def solve_duffing(
    problem: QPLIBProblem,
    steps: int = 150,
    dt: float = 0.20,
    gamma: float = 0.50,
    batch_size: int = 1,
    seed: int = 42,
    greedy_refine: bool = True,
) -> Dict[str, Any]:
    """Parametron-Duffing nonlinear oscillator solver.
    
    Uses 1 sparse matvec per step, zero auxiliary memory, velocity-Verlet symplectic updates.
    """
    rng = np.random.default_rng(seed)
    n = problem.n_vars
    B = batch_size

    K_norm = problem.K_norm
    h_norm = problem.h_norm

    t0 = time.perf_counter()

    if B == 1:
        u = rng.uniform(-0.05, 0.05, size=n)
        v = np.zeros(n, dtype=np.float64)
        for step in range(steps):
            t_ratio = step / float(steps)
            p_param = 0.2 + 0.8 * t_ratio
            c_gain = 0.5 * t_ratio
            m_hat = np.clip(u, -1.0, 1.0)
            force = -c_gain * (K_norm @ m_hat + h_norm)
            v += (-gamma * v + (p_param - 0.8) * u - (u ** 3) + force) * dt
            u += v * dt

        s_raw = np.where(u >= 0.0, 1.0, -1.0)
        s_final = problem.greedy_refine(s_raw) if greedy_refine else s_raw
        best_obj = problem.eval_s(s_final)
        best_s = s_final
    else:
        u = rng.uniform(-0.05, 0.05, size=(n, B))
        v = np.zeros((n, B), dtype=np.float64)
        h_mat = h_norm[:, None]
        for step in range(steps):
            t_ratio = step / float(steps)
            p_param = 0.2 + 0.85 * t_ratio
            c_gain = 0.55 * t_ratio
            m_hat = np.clip(u, -1.0, 1.0)
            force = -c_gain * (K_norm @ m_hat + h_mat)
            v += (-gamma * v + (p_param - 0.8) * u - (u ** 3) + force) * dt
            u += v * dt

        best_obj = -1e30 if problem.sense == "maximize" else 1e30
        best_s = None
        for b in range(B):
            s_raw = np.where(u[:, b] >= 0.0, 1.0, -1.0)
            s_cand = problem.greedy_refine(s_raw) if greedy_refine else s_raw
            obj = problem.eval_s(s_cand)
            if problem.sense == "maximize":
                if obj > best_obj:
                    best_obj = obj
                    best_s = s_cand
            else:
                if obj < best_obj:
                    best_obj = obj
                    best_s = s_cand

    elapsed = time.perf_counter() - t0
    opt_ratio = (
        (best_obj / problem.opt_obj * 100.0)
        if problem.opt_obj is not None and problem.opt_obj != 0
        else 0.0
    )

    return {
        "solver": f"duffing_digital_b{B}",
        "objective": float(best_obj),
        "opt_obj": problem.opt_obj,
        "opt_ratio_pct": float(opt_ratio),
        "seconds": float(elapsed),
        "steps": steps,
        "batch_size": B,
        "best_s": best_s,
    }


def solve_dsb(
    problem: QPLIBProblem,
    steps: int = 150,
    dt: float = 0.50,
    batch_size: int = 1,
    seed: int = 42,
    greedy_refine: bool = True,
) -> Dict[str, Any]:
    """Discrete Simulated Bifurcation (dSB, Goto et al. 2021)."""
    rng = np.random.default_rng(seed)
    n = problem.n_vars
    B = batch_size

    K_norm = problem.K_norm
    h_norm = problem.h_norm
    c0_param = 0.50

    t0 = time.perf_counter()

    if B == 1:
        x = rng.uniform(-0.05, 0.05, size=n)
        y = np.zeros(n, dtype=np.float64)
        for step in range(steps):
            t_ratio = step / float(steps)
            a_param = t_ratio
            c_param = t_ratio
            x += dt * c0_param * y
            x = np.clip(x, -1.0, 1.0)
            force = - (K_norm @ np.sign(x) + h_norm)
            y += dt * (-(a_param - 1.0) * x + c_param * force)

        s_raw = np.where(x >= 0.0, 1.0, -1.0)
        s_final = problem.greedy_refine(s_raw) if greedy_refine else s_raw
        best_obj = problem.eval_s(s_final)
        best_s = s_final
    else:
        x = rng.uniform(-0.05, 0.05, size=(n, B))
        y = np.zeros((n, B), dtype=np.float64)
        h_mat = h_norm[:, None]
        for step in range(steps):
            t_ratio = step / float(steps)
            a_param = t_ratio
            c_param = t_ratio
            x += dt * c0_param * y
            x = np.clip(x, -1.0, 1.0)
            force = - (K_norm @ np.sign(x) + h_mat)
            y += dt * (-(a_param - 1.0) * x + c_param * force)

        best_obj = -1e30 if problem.sense == "maximize" else 1e30
        best_s = None
        for b in range(B):
            s_raw = np.where(x[:, b] >= 0.0, 1.0, -1.0)
            s_cand = problem.greedy_refine(s_raw) if greedy_refine else s_raw
            obj = problem.eval_s(s_cand)
            if problem.sense == "maximize":
                if obj > best_obj:
                    best_obj = obj
                    best_s = s_cand
            else:
                if obj < best_obj:
                    best_obj = obj
                    best_s = s_cand

    elapsed = time.perf_counter() - t0
    opt_ratio = (
        (best_obj / problem.opt_obj * 100.0)
        if problem.opt_obj is not None and problem.opt_obj != 0
        else 0.0
    )

    return {
        "solver": f"dsb_b{B}",
        "objective": float(best_obj),
        "opt_obj": problem.opt_obj,
        "opt_ratio_pct": float(opt_ratio),
        "seconds": float(elapsed),
        "steps": steps,
        "batch_size": B,
        "best_s": best_s,
    }


def solve_sa(
    problem: QPLIBProblem,
    sweeps: int = 150,
    seed: int = 42,
    greedy_refine: bool = True,
) -> Dict[str, Any]:
    """Simulated Annealing with incremental gradient updates."""
    rng = np.random.default_rng(seed)
    n = problem.n_vars

    K = problem.K
    h = problem.h

    t0 = time.perf_counter()

    s = rng.choice([-1.0, 1.0], size=n)
    grad = K @ s + h

    # Scale initial temperature by typical energy change magnitude
    delta_typical = np.mean(np.abs(h)) + problem.max_deg * 0.25
    T_start = max(float(delta_typical), 1.0)
    T_end = T_start * 1e-4

    order = np.arange(n)
    for sweep in range(sweeps):
        t_ratio = sweep / float(sweeps)
        T = T_start * ((T_end / T_start) ** t_ratio)
        rng.shuffle(order)
        for i in order:
            delta = -2.0 * s[i] * grad[i]
            if delta <= 0.0 or (T > 0 and np.exp(-delta / T) > rng.random()):
                s[i] = -s[i]
                col = K.getcol(i).toarray().ravel()
                grad += 2.0 * s[i] * col

    s_final = problem.greedy_refine(s) if greedy_refine else s
    best_obj = problem.eval_s(s_final)
    elapsed = time.perf_counter() - t0

    opt_ratio = (
        (best_obj / problem.opt_obj * 100.0)
        if problem.opt_obj is not None and problem.opt_obj != 0
        else 0.0
    )

    return {
        "solver": "sa_b1",
        "objective": float(best_obj),
        "opt_obj": problem.opt_obj,
        "opt_ratio_pct": float(opt_ratio),
        "seconds": float(elapsed),
        "sweeps": sweeps,
        "batch_size": 1,
        "best_s": s_final,
    }

