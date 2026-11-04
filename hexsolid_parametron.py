"""HexSolid Parametron QUBO / Ising Solver.

A fast combinatorial optimization solver based on:
  - 1954 Parametron Principle (Dr. Eiichi Goto)
  - 1918 Duffing Nonlinear Oscillator (Georg Duffing)

SPDX-License-Identifier: Apache-2.0
Copyright 2026 nyoriworks.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


def solve_qubo(
    Q: Union[np.ndarray, sp.spmatrix],
    c: Optional[np.ndarray] = None,
    q0: float = 0.0,
    sense: str = "min",
    batch_size: int = 8,
    steps: int = 200,
    dt: float = 0.20,
    gamma: float = 0.50,
    seed: int = 42,
    greedy_refine: bool = True,
) -> Dict[str, Any]:
    """Solve an unconstrained binary quadratic program (QUBO):
    
        min (or max)  0.5 * x^T Q x + c^T x + q0   subject to x in {0, 1}^N

    Parameters:
      Q: (N, N) symmetric quadratic coefficient matrix (dense or sparse CSR)
      c: (N,) linear coefficient vector (optional, defaults to zeros)
      q0: objective constant scalar
      sense: 'min' or 'max'
      batch_size: Number of parallel trajectories (default 8 for AVX SIMD sweet spot)
      steps: Integration steps (default 200)
      dt: Integration step size (default 0.20)
      gamma: Damping coefficient (default 0.50)
      seed: Random seed for initial state perturbations
      greedy_refine: Whether to perform 1-spin steepest descent local search at finish

    Returns:
      dict with:
        'x': (N,) best binary solution vector in {0, 1}^N
        'objective': Best objective value achieved
        'seconds': Execution time in seconds
        'batch_size': Number of trajectories executed
    """
    t0 = time.perf_counter()
    n = Q.shape[0]

    # Convert to CSR sparse matrix
    if not sp.issparse(Q):
        Q_csr = sp.csr_matrix(Q, dtype=np.float64)
    else:
        Q_csr = Q.tocsr().astype(np.float64)

    # Symmetrize if not symmetric
    Q_sym = 0.5 * (Q_csr + Q_csr.T)

    if c is None:
        c_vec = np.zeros(n, dtype=np.float64)
    else:
        c_vec = np.asarray(c, dtype=np.float64)

    sign = -1.0 if sense.lower().startswith("max") else 1.0

    # Minimization target Hamiltonian:
    Q_tilde = sign * Q_sym
    c_tilde = sign * c_vec
    q0_tilde = sign * q0

    # Absorb diagonal elements into linear term (since x_i^2 = x_i for x_i in {0, 1})
    Q_diag = Q_tilde.diagonal()
    c_prime = c_tilde + 0.5 * Q_diag
    Q_nodiag = Q_tilde.copy()
    Q_nodiag.setdiag(0.0)

    # Standardized Ising Hamiltonian: H(s) = 0.5 * s^T K s + h^T s + offset
    # with x = (1 + s) / 2
    K = (0.25 * Q_nodiag).tocsr()
    h = 0.5 * c_prime + 0.25 * np.array(Q_nodiag.sum(axis=1)).ravel()

    row_sum_abs = np.array(np.abs(K).sum(axis=1)).ravel()
    max_deg = max(float(np.max(row_sum_abs)), 1.0)
    K_norm = (K / max_deg).tocsr()
    h_norm = h / max_deg

    rng = np.random.default_rng(seed)
    B = max(int(batch_size), 1)

    # Parametron-Duffing symplectic simulation
    if B == 1:
        u = rng.uniform(-0.05, 0.05, size=n)
        v = np.zeros(n, dtype=np.float64)
        for step in range(steps):
            t_ratio = step / float(steps)
            p_param = 0.2 + 0.85 * t_ratio
            c_gain = 0.55 * t_ratio
            m_hat = np.clip(u, -1.0, 1.0)
            force = -c_gain * (K_norm @ m_hat + h_norm)
            v += (-gamma * v + (p_param - 0.8) * u - (u ** 3) + force) * dt
            u += v * dt
        spins = np.where(u >= 0.0, 1.0, -1.0)[None, :]
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
        spins = np.where(u >= 0.0, 1.0, -1.0).T  # (B, N)

    def eval_orig(x_bin: np.ndarray) -> float:
        # A shared constant must not hide differences between candidates.
        return 0.5 * float(x_bin @ (Q_sym @ x_bin)) + float(c_vec @ x_bin)

    def local_descent(s: np.ndarray) -> np.ndarray:
        s_curr = s.copy()
        grad = K @ s_curr + h
        for _ in range(1000):
            delta = -2.0 * s_curr * grad
            best_i = int(np.argmin(delta))
            if delta[best_i] >= -1e-9:
                break
            s_curr[best_i] = -s_curr[best_i]
            col = K.getcol(best_i).toarray().ravel()
            grad += 2.0 * s_curr[best_i] * col
        return s_curr

    best_obj = np.inf if sign > 0 else -np.inf
    best_x = None

    for b in range(B):
        s_cand = spins[b]
        if greedy_refine:
            s_cand = local_descent(s_cand)
        x_cand = (1.0 + s_cand) / 2.0
        obj_cand = eval_orig(x_cand)

        if sign > 0:  # minimization
            if obj_cand < best_obj:
                best_obj = obj_cand
                best_x = x_cand
        else:  # maximization
            if obj_cand > best_obj:
                best_obj = obj_cand
                best_x = x_cand

    elapsed = time.perf_counter() - t0
    return {
        "x": best_x.astype(np.int8),
        "objective": float(best_obj + q0),
        "seconds": float(elapsed),
        "batch_size": B,
        "n_vars": n,
    }

