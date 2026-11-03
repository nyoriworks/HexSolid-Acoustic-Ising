"""Stanford G-set Max-Cut loader, independent cut verifier, and Windows Peak RSS measurement."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from problem import IsingProblem  # noqa: E402


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
    """Query OS-level PeakWorkingSetSize (Peak RSS in bytes) on Windows via Win32 API."""
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
    import resource

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def load_gset_problem(path: Path, name: str | None = None) -> IsingProblem:
    """
    Parse a standard 1-indexed Stanford G-set file:
      Line 1: N M
      Lines 2..M+1: u v weight
    into a sparse CSR IsingProblem with exact Max-Cut energy/cut mapping.
    """
    gname = name or path.stem
    lines = [ln.strip() for ln in path.read_text(encoding="ascii").splitlines() if ln.strip()]
    hdr = lines[0].split()
    n, m = int(hdr[0]), int(hdr[1])
    if len(lines) - 1 != m:
        raise ValueError(f"{path}: header declares M={m}, found {len(lines) - 1} edges")

    u_arr = np.empty(m, dtype=np.int32)
    v_arr = np.empty(m, dtype=np.int32)
    w_arr = np.empty(m, dtype=np.float64)
    for idx, ln in enumerate(lines[1:]):
        parts = ln.split()
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        w = float(parts[2]) if len(parts) >= 3 else 1.0
        if u == v or not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"{path}: invalid edge ({u+1}, {v+1}) on line {idx+2}")
        u_arr[idx] = u
        v_arr[idx] = v
        w_arr[idx] = w

    rows = np.concatenate([u_arr, v_arr])
    cols = np.concatenate([v_arr, u_arr])
    vals = np.concatenate([w_arr, w_arr])
    csr = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    csr.sum_duplicates()
    field = np.zeros(n, dtype=np.float64)
    w_tot = float(np.sum(w_arr))
    row_sq = np.asarray(csr.multiply(csr).sum(axis=1)).ravel()
    row_rms = float(np.sqrt(np.mean(row_sq)))
    # Estimate the relevant ground-state spectral edge max(row_rms, 0.5 * max(0, -lambda_min(K)))
    # rather than the ferromagnetic +d Perron root on high-degree positive-weight graphs.
    rng_pow = np.random.default_rng(0)
    v = rng_pow.standard_normal(n)
    v /= max(float(np.linalg.norm(v)), 1e-15)
    for _ in range(32):
        v = np.asarray(csr @ v, dtype=np.float64)
        v /= max(float(np.linalg.norm(v)), 1e-15)
    spec_abs = float(np.linalg.norm(np.asarray(csr @ v, dtype=np.float64)))
    u_vec = rng_pow.standard_normal(n)
    u_vec /= max(float(np.linalg.norm(u_vec)), 1e-15)
    for _ in range(48):
        u_vec = spec_abs * u_vec - np.asarray(csr @ u_vec, dtype=np.float64)
        u_vec /= max(float(np.linalg.norm(u_vec)), 1e-15)
    ku = np.asarray(csr @ u_vec, dtype=np.float64)
    neg_lambda_min = max(0.0, float(-np.dot(u_vec, ku)))
    j_scale = max(row_rms, 0.5 * neg_lambda_min, 1e-12)
    return IsingProblem(
        name=gname,
        kind="maxcut",
        n_spins=n,
        couplings=csr,
        field=field,
        total_edge_weight=w_tot,
        n_edges=m,
        j_scale=j_scale,
    )


def verify_gset_cut_from_raw_file(path: Path, spins_1d: np.ndarray) -> float:
    """Independent ground-truth evaluator reading raw G-set edge lines directly."""
    s = np.where(spins_1d >= 0.0, 1, -1)
    cut = 0.0
    lines = [ln.strip() for ln in path.read_text(encoding="ascii").splitlines() if ln.strip()]
    for ln in lines[1:]:
        parts = ln.split()
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        w = float(parts[2]) if len(parts) >= 3 else 1.0
        if s[u] != s[v]:
            cut += w
    return float(cut)
