# Large-Scale Acoustic OIM Extension Protocol (`N = 2,000 / 3,000 / 5,000 / 10,000`)

Frozen before generating `scale10k` random seeds or running `N >= 2,000` benchmarks.

## 1. Non-Destructive Extension Rules

1. **Zero Modification of Existing Deliverables and v1 Frozen Core**:
   - External HexSolid deliverables and all 12 frozen `acoustic-ising-v1` core modules (`PROTOCOL.md`, `problem.py`, `schedule.py`, `solver.py`, `frequency_map.py`, `backends/*`, `prepare.py`, `run.py`, `instances/*`, `results/*`) remain 100% unmodified.
   - `scale10k` uses the exact unmodified `IsingProblem`, `AnnealingSchedule`, `IsingAnnealer`, `PhaseODEBackend`, `AudioLoopbackBackend`, and `TopOctaveCombMap` classes.
2. **Disk Hygiene via Frozen 32-Byte OS Entropy Seeds**:
   - Because dense `float64` coupling matrices at $N = 10,000$ occupy $762.9\,\text{MiB}$ ($49,995,000$ pairs), `prepare_scale10k.py` records one 32-byte `os.urandom(32)` seed per instance in `instances_scale10k/seeds.json` (`< 4 KB` on disk) and constructs couplings/graphs deterministically in memory row-by-row via `PCG64` without multi-gigabyte disk files or intermediate `triu_indices` memory spikes.

## 2. Benchmark Matrix (`N = 2,000, 3,000, 5,000, 10,000`)

1. **Phase 1 (`PhaseODEBackend`, 1-thread BLAS, isolated process per case, Heun 800 steps, universal `AnnealingSchedule`)**:
   - **Sparse Weighted Max-Cut** (`|E| = 3N`, ring + random chords, weights $1..4$):
     - `maxcut2000` ($|E| = 6,000$, `batch = 32`)
     - `maxcut3000` ($|E| = 9,000$, `batch = 32`)
     - `maxcut5000` ($|E| = 15,000$, `batch = 32`)
     - `maxcut10000` ($|E| = 30,000$, `batch = 32`)
   - **Dense All-to-All Zero-Field Gaussian SK** ($K_{ij} = -g_{ij}/\sqrt{N}$):
     - `sk2000` ($1,999,000$ pairs, `batch = 32`)
     - `sk3000` ($4,498,500$ pairs, `batch = 32`)
     - `sk5000` ($12,497,500$ pairs, `batch = 16`)
     - `sk10000` ($49,995,000$ pairs, `batch = 16`)
   - **Classical Baseline**: Incremental $O(N)$/flip steepest 1-spin-flip Greedy run in its own isolated single-thread child process with the exact same number of starts (`32` or `16`).
2. **Phase 2 (`AudioLoopbackBackend`, 48 kHz `[10 kHz, 20 kHz)` harmonic-free comb + multipath + 42 dB AWGN)**:
   - `loopback_maxcut2000_comb`: $M = 19,200$ ($0.4\,\text{s}, \Delta f = 2.50\,\text{Hz}$), `batch = 4`
   - `loopback_maxcut3000_comb`: $M = 28,800$ ($0.6\,\text{s}, \Delta f = 1.67\,\text{Hz}$), `batch = 4`
   - `loopback_maxcut5000_comb`: $M = 48,000$ ($1.0\,\text{s}, \Delta f = 1.00\,\text{Hz}$), `batch = 2`
   - `loopback_maxcut10000_comb`: $M = 96,000$ ($2.0\,\text{s}, \Delta f = 0.50\,\text{Hz}$), `batch = 2`
   - `loopback_sk2000_comb`: $M = 19,200$ ($0.4\,\text{s}, \Delta f = 2.50\,\text{Hz}$), `batch = 4`

## 3. Pass Criteria & Independent Streaming Audit

- Every case must complete with finite valid binary spin witnesses (`s_i in {+1, -1}`), zero NaN/Inf, and bounded polynomial memory (`Max-Cut 10000` RSS `< 512 MiB`, dense `float64` `SK 10000` RSS `< 2,048 MiB`, accounting for the `762.9 MiB` coupling matrix `K` plus `_compute_j_scale` temporary array).
- `audit_scale10k.py` independently streams each SK row and Max-Cut edge list from the frozen 32-byte seeds in $O(N)$ memory and verifies every saved witness energy/cut to $< 10^{-8}$, plus invariance of all external and v1 frozen hashes.

