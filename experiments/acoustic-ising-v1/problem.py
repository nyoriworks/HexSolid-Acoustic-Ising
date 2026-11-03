"""Ising / Max-Cut problem representations, scale normalization, exact small-N oracles, and classical greedy baseline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
HEXSOLID_ROOT = ROOT.parent.parent
MAXCUT24_DIR = ROOT.parent / "tensor-maxcut-qaoa-v1"
SK24_DIR = ROOT.parent / "tensor-sk-qaoa-v1"

EXTERNAL_REFERENCE_HASHES: dict[Path, str] = {
    HEXSOLID_ROOT / "HANDOVER_ACOUSTIC_ISING_CURRENT.md": "0ca7fed329454a60d61523146dcc290464b694d209b1bfbc0c90c490cd63289c",
    HEXSOLID_ROOT / "HANDOVER_THIS_CHAT_CURRENT.md": "8a7ee6ffd74eae34407279c7e9bf9b22600487e35491ccbfbb40e3b031d6a792",
    HEXSOLID_ROOT / "HANDOVER_TENSOR_ALGORITHMS.md": "afa5b28cedc14d1a9e1a844f288e1e013d09c2840d004e77bc9906660b8cf4df",
    MAXCUT24_DIR / "graph.json": "1e1f37e5a666620aecffc7efaf468b500460207f018e6e8486257a3e58d360c5",
    MAXCUT24_DIR / "oracle.json": "4ad57e67bd9aaa9afd2c8e7978c2d9416f8fa1bd712df5a189d8a79de45d9f50",
    SK24_DIR / "couplings.npy": "d12ac3abe398ad922066a0c2635aebfb3f06cc90b51e9c4fe0985639c314b526",
    SK24_DIR / "oracle.json": "d9daef8a9d8bdb53ac0cb5cc1f0977802d18445319cc25fc9cf1a85bfe6e1e03",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_external_references() -> dict[str, dict[str, object]]:
    """Verify all external HexSolid reference files remain completely untouched."""
    report: dict[str, dict[str, object]] = {}
    for path, expected in EXTERNAL_REFERENCE_HASHES.items():
        actual = sha256_file(path)
        ok = actual == expected
        report[str(path)] = {"expected": expected, "actual": actual, "passed": ok}
        if not ok:
            raise RuntimeError(f"External read-only file hash mismatch: {path}")
    return report


def save_json(path: Path, data: object) -> None:
    def _default(x: object) -> object:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default, allow_nan=False) + "\n")


def spins_to_bitstring(spins_1d: np.ndarray) -> str:
    """q0_is_MSB = True: +1 -> '0', -1 -> '1'."""
    return "".join("0" if float(s) >= 0.0 else "1" for s in spins_1d)


def bitstring_to_spins(bitstring: str) -> np.ndarray:
    return np.array([1.0 if ch == "0" else -1.0 for ch in bitstring], dtype=np.float64)


@dataclass
class IsingProblem:
    """
    Ising minimization problem E(s) = sum_{i<j} K_ij s_i s_j + sum_i h_i s_i, s_i in {+1, -1}.
    Contains NO ground-state / oracle fields so solvers cannot access optimal labels.
    """

    name: str
    kind: str  # "sk" or "maxcut"
    n_spins: int
    couplings: np.ndarray | sparse.csr_matrix
    field: np.ndarray
    total_edge_weight: float
    n_edges: int
    j_scale: float

    @property
    def is_sparse(self) -> bool:
        return sparse.issparse(self.couplings)

    @property
    def coupling_storage_bytes(self) -> int:
        if sparse.issparse(self.couplings):
            csr = self.couplings
            return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes + self.field.nbytes)
        return int(self.couplings.nbytes + self.field.nbytes)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute x @ K for shape (N,) or (B, N) without creating N x N temporaries."""
        if sparse.issparse(self.couplings):
            if x.ndim == 1:
                return np.asarray(self.couplings @ x, dtype=np.float64)
            return np.asarray((self.couplings @ x.T).T, dtype=np.float64)
        return x @ self.couplings

    def energy(self, spins: np.ndarray) -> np.ndarray | float:
        """Exact Ising energy E(s) = 0.5 * sum_{i,j} K_ij s_i s_j + sum_i h_i s_i."""
        s = np.where(spins >= 0.0, 1.0, -1.0)
        ks = self.matvec(s)
        if s.ndim == 1:
            return float(0.5 * np.dot(s, ks) + np.dot(self.field, s))
        return 0.5 * np.sum(s * ks, axis=-1) + (s @ self.field)

    def cut_value(self, spins: np.ndarray) -> np.ndarray | float:
        """Weighted Max-Cut objective Cut(s) = 0.5 * (W_tot - E(s))."""
        if self.kind != "maxcut":
            raise ValueError("cut_value is defined only for maxcut problems")
        e = self.energy(spins)
        if isinstance(e, float):
            return float(0.5 * (self.total_edge_weight - e))
        return 0.5 * (self.total_edge_weight - e)

    def continuous_potential(self, phi: np.ndarray, kc: float, ks: float) -> np.ndarray | float:
        """Dimensionless continuous Lyapunov potential V(phi; Kc, Ks)."""
        c = np.cos(phi)
        s = np.sin(phi)
        kc_vec = self.matvec(c) / self.j_scale
        ks_vec = self.matvec(s) / self.j_scale
        hn = self.field / self.j_scale
        if phi.ndim == 1:
            pair = 0.5 * kc * float(np.dot(c, kc_vec) + np.dot(s, ks_vec))
            zeeman = kc * float(np.dot(hn, c))
            shil = -0.5 * ks * float(np.sum(np.cos(2.0 * phi)))
            return pair + zeeman + shil
        pair = 0.5 * kc * np.sum(c * kc_vec + s * ks_vec, axis=-1)
        zeeman = kc * (c @ hn)
        shil = -0.5 * ks * np.sum(np.cos(2.0 * phi), axis=-1)
        return pair + zeeman + shil

    @staticmethod
    def _compute_j_scale(couplings: np.ndarray | sparse.csr_matrix, field: np.ndarray) -> float:
        n = len(field)
        if sparse.issparse(couplings):
            row_sq = np.asarray(couplings.multiply(couplings).sum(axis=1)).ravel()
        else:
            row_sq = np.sum(couplings * couplings, axis=1)
        row_rms = float(np.sqrt(np.mean(row_sq + field * field)))
        v = np.ones(n, dtype=np.float64) / np.sqrt(max(n, 1))
        for _ in range(32):
            v = np.asarray(couplings @ v, dtype=np.float64)
            nrm = float(np.linalg.norm(v))
            if nrm <= 1e-15:
                break
            v /= nrm
        spec = float(np.linalg.norm(np.asarray(couplings @ v, dtype=np.float64)))
        return max(row_rms, 0.5 * spec, 1e-12)

    @classmethod
    def from_sk_npy(cls, path: Path, name: str) -> "IsingProblem":
        k = np.array(np.load(path), dtype=np.float64, copy=True)
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError(f"Expected square coupling matrix in {path}, got {k.shape}")
        if np.max(np.abs(k - k.T)) > 1e-14 or np.max(np.abs(np.diag(k))) > 1e-14:
            raise ValueError(f"Coupling matrix in {path} must be symmetric with zero diagonal")
        n = k.shape[0]
        field = np.zeros(n, dtype=np.float64)
        n_edges = int(n * (n - 1) // 2)
        return cls(
            name=name,
            kind="sk",
            n_spins=n,
            couplings=k,
            field=field,
            total_edge_weight=0.0,
            n_edges=n_edges,
            j_scale=cls._compute_j_scale(k, field),
        )

    @classmethod
    def from_maxcut_json(cls, path: Path, name: str, use_sparse: bool = True) -> "IsingProblem":
        g = json.loads(path.read_text())
        n = int(g["vertices"])
        edges = g["edges"]
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        w_tot = 0.0
        for e in edges:
            u, v, w = int(e["u"]), int(e["v"]), float(e["weight"])
            if u == v or not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"Invalid edge ({u}, {v}) in {path}")
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([w, w])
            w_tot += w
        csr = sparse.csr_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(n, n))
        csr.sum_duplicates()
        mat: np.ndarray | sparse.csr_matrix = csr if use_sparse else csr.toarray()
        field = np.zeros(n, dtype=np.float64)
        return cls(
            name=name,
            kind="maxcut",
            n_spins=n,
            couplings=mat,
            field=field,
            total_edge_weight=w_tot,
            n_edges=len(edges),
            j_scale=cls._compute_j_scale(mat, field),
        )


def draw_sk_from_bytes(n: int, raw: bytes) -> np.ndarray:
    """Deterministic Box-Muller mapping from raw OS bytes matching tensor-sk-qaoa-v1/sk.py."""
    count = n * (n - 1) // 2
    needed = 8 * 2 * ((count + 1) // 2)
    if len(raw) < needed:
        raise ValueError(f"Need at least {needed} bytes for SK({n}), got {len(raw)}")
    u = (np.frombuffer(raw[:needed], dtype="<u8") >> 11).astype(np.float64)
    u = (u + 0.5) / (2.0 ** 53)
    radius = np.sqrt(-2.0 * np.log(u[::2]))
    angle = 2.0 * np.pi * u[1::2]
    g = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]).ravel()[:count]
    k = np.zeros((n, n), dtype=np.float64)
    ii, jj = np.triu_indices(n, 1)
    k[ii, jj] = -g / np.sqrt(float(n))
    k += k.T
    return k


def draw_maxcut_from_bytes(n: int, n_edges: int, raw: bytes) -> dict[str, object]:
    """Deterministic ring + chord weighted graph from raw little-endian <u8 OS random bytes."""
    if n_edges < n or n_edges > n * (n - 1) // 2:
        raise ValueError(f"Invalid target edge count {n_edges} for N={n}")
    words = [int.from_bytes(raw[offset : offset + 8], "little") for offset in range(0, len(raw) - (len(raw) % 8), 8)]
    if len(words) < n + 3 * (n_edges - n):
        raise ValueError("Insufficient random words for chorded Max-Cut graph")
    edge_map: dict[tuple[int, int], int] = {}
    pos = 0
    for q in range(n):
        u, v = (q, (q + 1) % n)
        a, b = (u, v) if u < v else (v, u)
        edge_map[(a, b)] = 1 + (words[pos] % 4)
        pos += 1
    while len(edge_map) < n_edges:
        if pos + 3 > len(words):
            raise RuntimeError("Exhausted random byte stream before reaching target n_edges")
        u = words[pos] % n
        v = words[pos + 1] % n
        w = 1 + (words[pos + 2] % 4)
        pos += 3
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edge_map:
                edge_map[(a, b)] = w
    edges = [{"u": a, "v": b, "weight": edge_map[(a, b)]} for (a, b) in sorted(edge_map.keys())]
    return {
        "vertices": n,
        "edges": edges,
        "q0_is_MSB": True,
        "generator": "ring_plus_os_random_chords_weights_1_to_4",
        "words_consumed": pos,
    }


def xor_energies_exact(k_dense: np.ndarray) -> np.ndarray:
    """Bitwise XOR complete 2^N energy table (for small N <= 24 oracle construction/audit only)."""
    n = len(k_dense)
    index = np.arange(1 << n, dtype=np.uint32)
    out = np.zeros(1 << n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(k_dense[i, j])
            if w != 0.0:
                parity = (((index >> (n - 1 - i)) ^ (index >> (n - 1 - j))) & 1).astype(np.int8)
                out += w * (1.0 - 2.0 * parity)
    return out


def mitm_energies_exact(k_dense: np.ndarray) -> np.ndarray:
    """Independent half-split spin-matrix 2^N energy table."""
    n = len(k_dense)
    m = n // 2
    s_left = 1.0 - 2.0 * ((np.arange(1 << m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.float64)
    s_right = 1.0 - 2.0 * ((np.arange(1 << (n - m))[:, None] >> np.arange(n - m - 1, -1, -1)) & 1).astype(np.float64)
    ea = 0.5 * np.sum((s_left @ k_dense[:m, :m]) * s_left, axis=1)
    eb = 0.5 * np.sum((s_right @ k_dense[m:, m:]) * s_right, axis=1)
    return (ea[:, None] + eb[None, :] + (s_left @ k_dense[:m, m:]) @ s_right.T).ravel()


def initial_greedy_spins(problem_name: str, n_spins: int, seed: int) -> np.ndarray:
    """Deterministic SHA-256 bit unpacking; matches HexSolid-SK-greedy-v1:{seed} on sk24."""
    blocks: list[bytes] = []
    block_idx = 0
    while len(blocks) * 256 < n_spins:
        if problem_name == "sk24" and block_idx == 0:
            tag = f"HexSolid-SK-greedy-v1:{seed}".encode()
        else:
            tag = f"HexSolid-OIM-greedy-v1:{problem_name}:{seed}:{block_idx}".encode()
        blocks.append(hashlib.sha256(tag).digest())
        block_idx += 1
    bits = np.unpackbits(np.frombuffer(b"".join(blocks), dtype=np.uint8))[:n_spins]
    return 1.0 - 2.0 * bits.astype(np.float64)


def steepest_greedy_baseline(problem: IsingProblem, n_starts: int = 128) -> dict[str, object]:
    """Steepest 1-spin-flip local search from 128 deterministic initial states."""
    t0 = time.perf_counter()
    energies = np.empty(n_starts, dtype=np.float64)
    steps_arr = np.empty(n_starts, dtype=np.int64)
    best_energy = float("inf")
    best_spins: np.ndarray | None = None
    best_seed = -1

    for seed in range(n_starts):
        z = initial_greedy_spins(problem.name, problem.n_spins, seed)
        steps = 0
        while True:
            delta = -2.0 * z * (problem.matvec(z) + problem.field)
            q = int(np.argmin(delta))
            if delta[q] >= -1e-12:
                break
            z[q] *= -1.0
            steps += 1
        e = float(problem.energy(z))
        energies[seed] = e
        steps_arr[seed] = steps
        if e < best_energy:
            best_energy = e
            best_spins = z.copy()
            best_seed = seed

    elapsed = time.perf_counter() - t0
    assert best_spins is not None
    out: dict[str, object] = {
        "n_starts": n_starts,
        "seconds": elapsed,
        "best_energy": best_energy,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "best_seed": best_seed,
        "best_bitstring": spins_to_bitstring(best_spins),
        "mean_steps": float(np.mean(steps_arr)),
        "energies": energies,
    }
    if problem.kind == "maxcut":
        cuts = 0.5 * (problem.total_edge_weight - energies)
        out["best_cut"] = float(np.max(cuts))
        out["mean_cut"] = float(np.mean(cuts))
        out["std_cut"] = float(np.std(cuts))
    return out

