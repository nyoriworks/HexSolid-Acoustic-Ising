"""QPLIB Problem Loader and Exact Ising Transformation.

Handles unconstrained binary quadratic programs (QBN) from QPLIB,
transforms them into standardized Ising spin-glass minimization models,
and provides fast sparse matrix operations and exact objective evaluation.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import pathlib
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_peak_rss_bytes() -> int:
    """Query OS-level PeakWorkingSetSize on Windows via Win32 API."""
    if sys.platform == "win32":
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = k32.GetCurrentProcess()
        if k32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize)
    try:
        import resource
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw if sys.platform == "darwin" else raw * 1024
    except Exception:
        return 0


class QPLIBProblem:
    """Represents a QPLIB unconstrained binary quadratic instance."""

    def __init__(
        self,
        name: str,
        qid: int,
        n_vars: int,
        sense: str,
        Q_sym: sp.csr_matrix,
        c: np.ndarray,
        q0: float,
        opt_obj: Optional[float] = None,
    ) -> None:
        self.name = name
        self.qid = qid
        self.n_vars = n_vars
        self.sense = sense.lower()  # 'maximize' or 'minimize'
        self.Q_sym = Q_sym
        self.c = c
        self.q0 = q0
        self.opt_obj = opt_obj

        # Standardized minimization Ising Hamiltonian:
        # H(s) = 0.5 * s^T K s + h^T s + offset
        # where s in {-1, +1}^N, x = (1 + s) / 2 in {0, 1}^N
        # If sense == 'maximize', E_qubo(x) = -f(x), so minimizing H(s) maximizes f(x).
        sign = -1.0 if self.sense == "maximize" else 1.0
        self.sign_factor = sign

        Q_tilde = sign * self.Q_sym
        c_tilde = sign * self.c
        q0_tilde = sign * self.q0

        # Absorb diagonal terms Q_ii into linear term
        Q_diag = Q_tilde.diagonal()
        c_prime = c_tilde + 0.5 * Q_diag
        Q_tilde_nodiag = Q_tilde.copy()
        Q_tilde_nodiag.setdiag(0.0)

        self.K = (0.25 * Q_tilde_nodiag).tocsr()
        self.h = (0.5 * c_prime + 0.25 * np.array(Q_tilde_nodiag.sum(axis=1)).ravel()).astype(np.float64)
        self.offset = float(q0_tilde + 0.5 * np.sum(c_prime) + 0.125 * float(Q_tilde_nodiag.sum()))

        # Precompute degree and spectral normalization
        row_sum_abs = np.array(np.abs(self.K).sum(axis=1)).ravel()
        self.max_deg = max(float(np.max(row_sum_abs)), 1.0)
        self.K_norm = (self.K / self.max_deg).tocsr()
        self.h_norm = self.h / self.max_deg

    @classmethod
    def load(cls, qplib_path: pathlib.Path, sol_path: Optional[pathlib.Path] = None) -> QPLIBProblem:
        with open(qplib_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        name = lines[0]
        qid = int(name.split("_")[1]) if "_" in name else 0
        prob_type = lines[1]
        sense = lines[2].lower()
        n_vars = int(lines[3].split()[0])
        n_quad = int(lines[4].split()[0])

        rows, cols, data = [], [], []
        idx = 5
        for _ in range(n_quad):
            parts = lines[idx].split()
            h, k, v = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            if h == k:
                rows.append(h); cols.append(k); data.append(v)
            else:
                rows.append(h); cols.append(k); data.append(0.5 * v)
                rows.append(k); cols.append(h); data.append(0.5 * v)
            idx += 1

        Q_sym = sp.csr_matrix((data, (rows, cols)), shape=(n_vars, n_vars), dtype=np.float64)

        def_lin = float(lines[idx].split()[0]); idx += 1
        n_lin = int(lines[idx].split()[0]); idx += 1
        c = np.full(n_vars, def_lin, dtype=np.float64)
        for _ in range(n_lin):
            parts = lines[idx].split()
            c[int(parts[0]) - 1] = float(parts[1])
            idx += 1

        q0 = float(lines[idx].split()[0])

        opt_obj = None
        if sol_path and sol_path.exists():
            with open(sol_path, "r", encoding="utf-8", errors="ignore") as sf:
                for line in sf:
                    parts = line.strip().split()
                    if parts and parts[0] == "objvar":
                        opt_obj = float(parts[1])
                        break

        return cls(
            name=name,
            qid=qid,
            n_vars=n_vars,
            sense=sense,
            Q_sym=Q_sym,
            c=c,
            q0=q0,
            opt_obj=opt_obj,
        )

    def eval_x(self, x: np.ndarray) -> float:
        """Evaluate the original QPLIB objective value on binary vector x in {0, 1}^N."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            quad_val = 0.5 * float(x @ (self.Q_sym @ x))
            lin_val = float(self.c @ x)
            return quad_val + lin_val + self.q0
        else:
            # Batch mode: shape (N, B)
            quad_vals = 0.5 * np.sum(x * (self.Q_sym @ x), axis=0)
            lin_vals = np.dot(self.c, x)
            return quad_vals + lin_vals + self.q0

    def eval_s(self, s: np.ndarray) -> float:
        """Evaluate the original QPLIB objective value on spin vector s in {-1, +1}^N."""
        x = (1.0 + np.asarray(s, dtype=np.float64)) / 2.0
        return self.eval_x(x)

    def eval_ising_energy(self, s: np.ndarray) -> float:
        """Evaluate standardized Ising Hamiltonian H(s) = 0.5 s^T K s + h^T s + offset."""
        s = np.asarray(s, dtype=np.float64)
        if s.ndim == 1:
            return 0.5 * float(s @ (self.K @ s)) + float(self.h @ s) + self.offset
        else:
            quad = 0.5 * np.sum(s * (self.K @ s), axis=0)
            lin = np.dot(self.h, s)
            return quad + lin + self.offset

    def greedy_refine(self, s: np.ndarray, max_flips: int = 2000) -> np.ndarray:
        """1-spin steepest descent local search on Ising Hamiltonian."""
        s = s.copy().astype(np.float64)
        grad = self.K @ s + self.h
        for _ in range(max_flips):
            delta = -2.0 * s * grad
            best_i = int(np.argmin(delta))
            if delta[best_i] >= -1e-9:
                break
            s[best_i] = -s[best_i]
            col = self.K.getcol(best_i).toarray().ravel()
            grad += 2.0 * s[best_i] * col
        return s

