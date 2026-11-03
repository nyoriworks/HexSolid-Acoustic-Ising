"""Freeze one-time 32-byte OS random seeds for N = 2000, 3000, 5000, 10000 benchmarks without multi-GB disk bloat."""
from __future__ import annotations

import os
import sys

_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
if any(os.environ.get(k) != "1" for k in _THREAD_VARS):
    env = dict(os.environ)
    for k in _THREAD_VARS:
        env[k] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import json
import math
from pathlib import Path
import numpy as np
from scipy import sparse

from prepare import compute_source_hashes
from problem import IsingProblem, ROOT, save_json, verify_external_references

SCALE10K_SIZES = (2000, 3000, 5000, 10000)


def rng_from_hex_seed(seed_hex: str) -> np.random.Generator:
    raw = bytes.fromhex(seed_hex)
    words = np.frombuffer(raw, dtype="<u4")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def build_streamed_sk_problem(n: int, seed_hex: str, name: str) -> IsingProblem:
    """Construct symmetric float64 SK matrix row-by-row with O(N) temporary memory (no triu_indices spike)."""
    rng = rng_from_hex_seed(seed_hex)
    k = np.zeros((n, n), dtype=np.float64)
    scale = -1.0 / math.sqrt(float(n))
    for i in range(n - 1):
        row = rng.standard_normal(n - 1 - i) * scale
        k[i, i + 1 :] = row
        k[i + 1 :, i] = row
    field = np.zeros(n, dtype=np.float64)
    return IsingProblem(
        name=name,
        kind="sk",
        n_spins=n,
        couplings=k,
        field=field,
        total_edge_weight=0.0,
        n_edges=n * (n - 1) // 2,
        j_scale=IsingProblem._compute_j_scale(k, field),
    )


def generate_maxcut_edges_from_seed(n: int, n_edges: int, seed_hex: str) -> list[tuple[int, int, int]]:
    """Deterministic ring + random chords (weights 1..4) from frozen 32-byte OS seed."""
    rng = rng_from_hex_seed(seed_hex)
    edge_map: dict[tuple[int, int], int] = {}
    ring_weights = rng.integers(1, 5, size=n, dtype=np.int64)
    for q in range(n):
        a, b = (q, (q + 1) % n)
        if a > b:
            a, b = b, a
        edge_map[(a, b)] = int(ring_weights[q])

    while len(edge_map) < n_edges:
        need = max((n_edges - len(edge_map)) * 2, 256)
        us = rng.integers(0, n, size=need, dtype=np.int64)
        vs = rng.integers(0, n, size=need, dtype=np.int64)
        ws = rng.integers(1, 5, size=need, dtype=np.int64)
        for u_raw, v_raw, w_raw in zip(us, vs, ws):
            u, v, w = int(u_raw), int(v_raw), int(w_raw)
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in edge_map:
                    edge_map[(a, b)] = w
                    if len(edge_map) == n_edges:
                        break
    return [(a, b, edge_map[(a, b)]) for (a, b) in sorted(edge_map)]


def build_seeded_maxcut_problem(n: int, n_edges: int, seed_hex: str, name: str) -> IsingProblem:
    edges = generate_maxcut_edges_from_seed(n, n_edges, seed_hex)
    rows = np.empty(2 * len(edges), dtype=np.int32)
    cols = np.empty(2 * len(edges), dtype=np.int32)
    vals = np.empty(2 * len(edges), dtype=np.float64)
    w_tot = 0.0
    for idx, (u, v, w) in enumerate(edges):
        rows[2 * idx] = u
        cols[2 * idx] = v
        vals[2 * idx] = float(w)
        rows[2 * idx + 1] = v
        cols[2 * idx + 1] = u
        vals[2 * idx + 1] = float(w)
        w_tot += float(w)
    csr = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    csr.sum_duplicates()
    field = np.zeros(n, dtype=np.float64)
    return IsingProblem(
        name=name,
        kind="maxcut",
        n_spins=n,
        couplings=csr,
        field=field,
        total_edge_weight=w_tot,
        n_edges=len(edges),
        j_scale=IsingProblem._compute_j_scale(csr, field),
    )


def main() -> None:
    seeds_path = ROOT / "instances_scale10k" / "seeds.json"
    if seeds_path.exists():
        raise FileExistsError("instances_scale10k/seeds.json already exists; refusing to overwrite frozen seeds")

    verify_external_references()
    v1_sources = json.loads((ROOT / "source-hashes.json").read_text())
    if compute_source_hashes() != v1_sources:
        raise RuntimeError("v1 tracked core sources changed!")

    instance_seeds: dict[str, dict[str, object]] = {}
    for n in SCALE10K_SIZES:
        instance_seeds[f"maxcut{n}"] = {
            "kind": "maxcut",
            "n_spins": n,
            "n_edges": 3 * n,
            "seed_hex": os.urandom(32).hex(),
        }
        instance_seeds[f"sk{n}"] = {
            "kind": "sk",
            "n_spins": n,
            "n_edges": n * (n - 1) // 2,
            "seed_hex": os.urandom(32).hex(),
        }

    payload = {
        "created_by": "prepare_scale10k.py",
        "solver_master_seed_hex": os.urandom(32).hex(),
        "instances": instance_seeds,
    }
    save_json(seeds_path, payload)
    print(f"Prepared frozen 32-byte OS seeds for {list(instance_seeds.keys())} -> {seeds_path}", flush=True)


if __name__ == "__main__":
    main()

