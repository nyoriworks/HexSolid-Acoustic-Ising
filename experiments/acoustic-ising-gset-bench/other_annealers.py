"""Reference annealing solvers (Discrete Simulated Bifurcation dSB & Exact Chromatic Metropolis SA) for fair G-set comparison."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from problem import IsingProblem, spins_to_bitstring  # noqa: E402


@dataclass
class AnnealerBenchResult:
    solver_name: str
    batch_size: int
    n_steps: int
    seconds: float
    best_energy: float
    mean_energy: float
    std_energy: float
    best_cut: float
    mean_cut: float
    std_cut: float
    best_spins: np.ndarray
    best_bitstring: str


def solve_dsb(
    problem: IsingProblem,
    rng: np.random.Generator,
    n_steps: int = 800,
    batch_size: int = 32,
    dt: float = 0.50,
    a0: float = 1.0,
) -> AnnealerBenchResult:
    """
    Discrete Simulated Bifurcation (dSB, Goto et al., Science Advances 2021).
    Minimizes E(s) = 0.5 * s^T K s + h^T s using coupled Kerr-nonlinear oscillator bifurcation
    with inelastic walls at x_i = +/- 1 and discrete sign interaction.
    """
    t0 = time.perf_counter()
    n = problem.n_spins
    if sparse.issparse(problem.couplings):
        row_sq = np.asarray(problem.couplings.multiply(problem.couplings).sum(axis=1)).ravel()
    else:
        row_sq = np.sum(problem.couplings * problem.couplings, axis=1)
    denom = float(np.sqrt(np.mean(row_sq + problem.field * problem.field)))
    c0 = 0.5 / max(denom, 1e-12)

    shape = (batch_size, n) if batch_size > 1 else (n,)
    x = rng.uniform(-0.1, 0.1, size=shape)
    y = rng.uniform(-0.1, 0.1, size=shape)
    has_field = bool(np.any(problem.field != 0.0))

    for step in range(n_steps):
        a_t = a0 * float(step + 1) / float(n_steps)
        sgn_x = np.sign(x)
        force = -problem.matvec(sgn_x)
        if has_field:
            force = force - problem.field
        y = y + (-(a0 - a_t) * x + c0 * force) * dt
        x = x + (a0 * y) * dt
        hit = np.abs(x) > 1.0
        if np.any(hit):
            x = np.clip(x, -1.0, 1.0)
            y = np.where(hit, 0.0, y)

    spins = np.where(x >= 0.0, 1.0, -1.0)
    energies = np.atleast_1d(np.asarray(problem.energy(spins), dtype=np.float64))
    cuts = np.atleast_1d(np.asarray(problem.cut_value(spins), dtype=np.float64))
    best_idx = int(np.argmin(energies))
    best_s = spins[best_idx].copy() if batch_size > 1 else spins.copy()
    elapsed = time.perf_counter() - t0
    return AnnealerBenchResult(
        solver_name=f"dSB(Goto2021, steps={n_steps}, dt={dt})",
        batch_size=batch_size,
        n_steps=n_steps,
        seconds=elapsed,
        best_energy=float(energies[best_idx]),
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
        best_cut=float(cuts[best_idx]),
        mean_cut=float(np.mean(cuts)),
        std_cut=float(np.std(cuts)),
        best_spins=best_s,
        best_bitstring=spins_to_bitstring(best_s),
    )


def _greedy_graph_coloring(csr: sparse.csr_matrix) -> list[np.ndarray]:
    """Welsh-Powell degree-ordered greedy graph coloring into exact independent sets."""
    n = csr.shape[0]
    indptr = csr.indptr
    indices = csr.indices
    degrees = indptr[1:] - indptr[:-1]
    order = np.argsort(-degrees)
    colors = np.full(n, -1, dtype=np.int32)
    used_stamp = np.full(n + 1, -1, dtype=np.int32)
    max_c = -1
    for idx_pos, u in enumerate(order):
        u_int = int(u)
        for v in indices[indptr[u_int] : indptr[u_int + 1]]:
            c = colors[v]
            if c >= 0:
                used_stamp[c] = idx_pos
        c_pick = 0
        while used_stamp[c_pick] == idx_pos:
            c_pick += 1
        colors[u_int] = c_pick
        if c_pick > max_c:
            max_c = c_pick
    return [np.where(colors == c)[0].astype(np.int32) for c in range(max_c + 1)]


def solve_sa(
    problem: IsingProblem,
    rng: np.random.Generator,
    n_sweeps: int = 800,
    batch_size: int = 32,
) -> AnnealerBenchResult:
    """
    Exact single-spin-flip Metropolis Simulated Annealing accelerated via exact chromatic
    independent-set partitioning (zero parallel-flip conflict; algebraically identical to
    sequential Gibbs/Metropolis sweeps in chromatic order).
    """
    t0 = time.perf_counter()
    n = problem.n_spins
    csr = problem.couplings if sparse.issparse(problem.couplings) else sparse.csr_matrix(problem.couplings)
    color_classes = _greedy_graph_coloring(csr)
    sub_mats = [csr[c_idx, :] for c_idx in color_classes]
    sub_fields = [problem.field[c_idx][:, None] for c_idx in color_classes]
    has_field = bool(np.any(problem.field != 0.0))

    # Shape (N, B) for contiguous sparse-dense matmul sub_mat @ s
    s = rng.choice(np.array([-1.0, 1.0]), size=(n, batch_size))
    best_s_mat = s.copy()
    best_e_vec = np.atleast_1d(np.asarray(problem.energy(s.T), dtype=np.float64))

    t_init = max(2.5 * problem.j_scale, 2.0)
    t_final = 0.08
    log_ratio = math.log(t_final / t_init)

    for sw in range(n_sweeps):
        frac = float(sw) / float(max(n_sweeps - 1, 1))
        beta = 1.0 / (t_init * math.exp(log_ratio * frac))
        for c_idx, sub_m, sub_f in zip(color_classes, sub_mats, sub_fields):
            h_loc = sub_m @ s
            if has_field:
                h_loc = h_loc + sub_f
            s_sub = s[c_idx, :]
            de = -2.0 * s_sub * h_loc
            accept = de <= 0.0
            uphill = ~accept
            if np.any(uphill):
                u_rand = rng.random(size=de.shape)
                accept = accept | (u_rand < np.exp(np.clip(-beta * de, -60.0, 0.0)))
            s[c_idx, :] = np.where(accept, -s_sub, s_sub)

        if (sw + 1) % 50 == 0 or sw == n_sweeps - 1:
            cur_e = np.atleast_1d(np.asarray(problem.energy(s.T), dtype=np.float64))
            improved = cur_e < best_e_vec
            if np.any(improved):
                best_e_vec = np.where(improved, cur_e, best_e_vec)
                best_s_mat[:, improved] = s[:, improved]

    spins_bn = best_s_mat.T
    energies = best_e_vec
    cuts = np.atleast_1d(np.asarray(problem.cut_value(spins_bn), dtype=np.float64))
    best_idx = int(np.argmin(energies))
    best_s = spins_bn[best_idx].copy()
    elapsed = time.perf_counter() - t0
    return AnnealerBenchResult(
        solver_name=f"SA(Metropolis, sweeps={n_sweeps}, colors={len(color_classes)})",
        batch_size=batch_size,
        n_steps=n_sweeps,
        seconds=elapsed,
        best_energy=float(energies[best_idx]),
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
        best_cut=float(cuts[best_idx]),
        mean_cut=float(np.mean(cuts)),
        std_cut=float(np.std(cuts)),
        best_spins=best_s,
        best_bitstring=spins_to_bitstring(best_s),
    )

