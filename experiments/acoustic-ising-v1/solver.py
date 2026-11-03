"""Backend-agnostic Oscillator Ising Machine (OIM) solver."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
import numpy as np

from backends.base import PhaseTransportBackend
from problem import IsingProblem, spins_to_bitstring
from schedule import AnnealingSchedule


def derive_problem_rng(master_seed_bytes: bytes, problem_name: str, extra_tag: str = "") -> np.random.Generator:
    """Deterministically derive an independent PCG64 stream from recorded OS entropy + problem tag."""
    digest = hashlib.sha256(master_seed_bytes + b":" + problem_name.encode() + b":" + extra_tag.encode()).digest()
    words = np.frombuffer(digest, dtype="<u4")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


@dataclass
class OIMResult:
    problem_name: str
    backend_name: str
    integrator: str
    batch_size: int
    n_steps: int
    seconds: float
    best_energy: float
    mean_energy: float
    std_energy: float
    best_index_in_batch: int
    best_bitstring: str
    best_spins: np.ndarray
    energies: np.ndarray
    prefixes: list[dict[str, object]]
    best_cut: float | None = None
    mean_cut: float | None = None
    std_cut: float | None = None
    phi_history: np.ndarray | None = None


class IsingAnnealer:
    """
    Evolves continuous phase spins phi(t) using ONLY the abstract PhaseTransportBackend interface.
    Identical solver code runs PhaseODEBackend, AudioLoopbackBackend, and RealAcousticBackend.
    """

    def __init__(
        self,
        schedule: AnnealingSchedule,
        backend: PhaseTransportBackend,
        integrator: str = "heun",
    ) -> None:
        if integrator not in ("heun", "euler"):
            raise ValueError(f"Unsupported integrator {integrator!r}; expected 'heun' or 'euler'")
        self.schedule = schedule
        self.backend = backend
        self.integrator = integrator

    def _drift_from_phasors(self, problem: IsingProblem, z_obs: np.ndarray, u: float) -> np.ndarray:
        """
        Compute -dV/dphi from observed unit phasors z_obs = cos(phi_hat) + 1j * sin(phi_hat):
        +K_c * sum_j K_tilde_ij sin(phi_hat_i - phi_hat_j) + K_c * h_tilde_i sin(phi_hat_i) - K_s * sin(2 phi_hat_i)
        """
        kc = self.schedule.kc(u)
        ks = self.schedule.ks(u)
        c = z_obs.real
        s = z_obs.imag
        inv_scale = 1.0 / problem.j_scale
        kc_vec = problem.matvec(c) * inv_scale
        ks_vec = problem.matvec(s) * inv_scale
        coupling_torque = kc * (s * kc_vec - c * ks_vec)
        if np.any(problem.field != 0.0):
            coupling_torque = coupling_torque + (kc * inv_scale) * (s * problem.field)
        # Exact identity sin(2*phi_hat) = 2 * sin(phi_hat) * cos(phi_hat)
        shil_torque = (2.0 * ks) * (s * c)
        return coupling_torque - shil_torque

    def solve(
        self,
        problem: IsingProblem,
        rng: np.random.Generator,
        batch_size: int = 128,
        prefix_sizes: tuple[int, ...] = (16, 64, 128),
        record_trajectory_index: int | None = None,
    ) -> OIMResult:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        t0 = time.perf_counter()
        self.backend.reset(problem.n_spins, rng)

        shape = (batch_size, problem.n_spins) if batch_size > 1 else (problem.n_spins,)
        phi = rng.uniform(0.0, 2.0 * math.pi, size=shape)
        dt = self.schedule.dt
        sqrt_dt = math.sqrt(dt)
        n_steps = self.schedule.n_steps

        history: list[np.ndarray] = []
        if record_trajectory_index is not None:
            history.append(phi[record_trajectory_index].copy() if batch_size > 1 else phi.copy())

        for step in range(n_steps):
            u = self.schedule.progress(step)
            sig = self.schedule.sigma(u)
            dw = (sig * sqrt_dt) * rng.standard_normal(shape) if sig > 1e-12 else 0.0

            z1 = self.backend.measure_phasors(phi)
            k1 = self._drift_from_phasors(problem, z1, u)
            phi_pred = phi + dt * k1 + dw

            if self.integrator == "heun":
                u_next = self.schedule.progress(min(step + 1, n_steps - 1))
                z2 = self.backend.measure_phasors(phi_pred)
                k2 = self._drift_from_phasors(problem, z2, u_next)
                phi = np.mod(phi + 0.5 * dt * (k1 + k2) + dw, 2.0 * math.pi)
            else:
                phi = np.mod(phi_pred, 2.0 * math.pi)

            if record_trajectory_index is not None:
                history.append(phi[record_trajectory_index].copy() if batch_size > 1 else phi.copy())

        # Final readout strictly via backend observation and sign(Re(z_obs)) — NO greedy post-polish
        z_final = self.backend.measure_phasors(phi)
        spins = np.where(z_final.real >= 0.0, 1.0, -1.0)
        elapsed = time.perf_counter() - t0

        if batch_size == 1:
            spins_2d = spins[None, :]
            energies = np.array([float(problem.energy(spins))], dtype=np.float64)
        else:
            spins_2d = spins
            energies = np.asarray(problem.energy(spins_2d), dtype=np.float64)

        best_idx = int(np.argmin(energies))
        best_spins = spins_2d[best_idx].copy()
        best_energy = float(energies[best_idx])

        prefixes: list[dict[str, object]] = []
        for p_count in prefix_sizes:
            if p_count <= batch_size:
                sub = energies[:p_count]
                sub_idx = int(np.argmin(sub))
                row: dict[str, object] = {
                    "trajectories": int(p_count),
                    "best_energy": float(sub[sub_idx]),
                    "mean_energy": float(np.mean(sub)),
                    "best_trajectory": sub_idx,
                    "best_bitstring": spins_to_bitstring(spins_2d[sub_idx]),
                }
                if problem.kind == "maxcut":
                    row["best_cut"] = float(0.5 * (problem.total_edge_weight - float(sub[sub_idx])))
                    row["mean_cut"] = float(0.5 * (problem.total_edge_weight - float(np.mean(sub))))
                prefixes.append(row)

        res = OIMResult(
            problem_name=problem.name,
            backend_name=self.backend.name,
            integrator=self.integrator,
            batch_size=batch_size,
            n_steps=n_steps,
            seconds=elapsed,
            best_energy=best_energy,
            mean_energy=float(np.mean(energies)),
            std_energy=float(np.std(energies)),
            best_index_in_batch=best_idx,
            best_bitstring=spins_to_bitstring(best_spins),
            best_spins=best_spins,
            energies=energies,
            prefixes=prefixes,
            phi_history=np.stack(history, axis=0) if history else None,
        )
        if problem.kind == "maxcut":
            cuts = 0.5 * (problem.total_edge_weight - energies)
            res.best_cut = float(np.max(cuts))
            res.mean_cut = float(np.mean(cuts))
            res.std_cut = float(np.std(cuts))
        return res

