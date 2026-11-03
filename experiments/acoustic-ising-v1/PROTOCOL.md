# Acoustic Oscillator Ising Machine (OIM), version 1 — Pre-Registered Protocol

Frozen before generating new random problem instances or observing benchmark outputs.

## 1. Scope, Terminology, and Strict Boundaries

1. **Classical Phase Spin Semantics**:
   - State variables are classical continuous oscillator phases $\phi_i \in [0, 2\pi)$ (`Phase Spin` / `OIM Spin`), read out as binary Ising spins $s_i = \operatorname{sign}(\cos\phi_i) \in \{+1, -1\}$ ($\cos\phi_i \ge 0 \mapsto +1$, $\cos\phi_i < 0 \mapsto -1$).
   - **Prohibited claims**: Do NOT describe variables as qubits (`Qubit`), quantum entanglement, or quantum advantage. This architecture solves classical combinatorial Ising/QUBO optimization in $O(N)$ state memory ($O(|E|)$ sparse / $O(N^2)$ dense coupling storage) and does not execute Shor's algorithm or universal quantum gate circuits.
2. **Complete Protection of Existing Deliverables**:
   - No file outside `experiments/acoustic-ising-v1/` may be created, modified, or deleted.
   - Read-only external reference SHA-256 digests verified before and after execution:
     - `HANDOVER_ACOUSTIC_ISING_CURRENT.md`: `0ca7fed329454a60d61523146dcc290464b694d209b1bfbc0c90c490cd63289c`
     - `HANDOVER_THIS_CHAT_CURRENT.md`: `8a7ee6ffd74eae34407279c7e9bf9b22600487e35491ccbfbb40e3b031d6a792`
     - `HANDOVER_TENSOR_ALGORITHMS.md`: `afa5b28cedc14d1a9e1a844f288e1e013d09c2840d004e77bc9906660b8cf4df`
     - `experiments/tensor-maxcut-qaoa-v1/graph.json`: `1e1f37e5a666620aecffc7efaf468b500460207f018e6e8486257a3e58d360c5`
     - `experiments/tensor-maxcut-qaoa-v1/oracle.json`: `4ad57e67bd9aaa9afd2c8e7978c2d9416f8fa1bd712df5a189d8a79de45d9f50`
     - `experiments/tensor-sk-qaoa-v1/couplings.npy`: `d12ac3abe398ad922066a0c2635aebfb3f06cc90b51e9c4fe0985639c314b526`
     - `experiments/tensor-sk-qaoa-v1/oracle.json`: `d9daef8a9d8bdb53ac0cb5cc1f0977802d18445319cc25fc9cf1a85bfe6e1e03`
3. **Zero Oracle Leakage & Zero Post-Hoc Polish**:
   - Ground-state energies/cuts (`62` for Max-Cut 24, `-16.548016339179107` for SK 24, and exact $N=12$ enumerations) are NEVER passed to `IsingProblem`, `AnnealingSchedule`, `IsingAnnealer`, stopping criteria, or parameter selection.
   - Final spin readout is strictly raw phase binarization $s_i = +1$ if $\operatorname{Re}(z_{\text{obs}, i}(T)) \ge 0$ else $-1$. No post-hoc greedy local search polish is applied to OIM outputs.

## 2. Mathematical Model and Sign Conventions

1. **Ising Energy (Minimization)**:
   $$E(s) = \sum_{0 \le i < j < N} K_{ij} s_i s_j + \sum_{i=0}^{N-1} h_i s_i, \quad s_i \in \{+1, -1\}$$
   with symmetric zero-diagonal $K_{ij} = K_{ji}$, $K_{ii} = 0$. Bitstring encoding (`q0_is_MSB = true`) maps `'0'` $\leftrightarrow s_i = +1$ and `'1'` $\leftrightarrow s_i = -1$.
   - **Weighted Max-Cut**: For undirected edge $(u, v)$ with weight $w_{uv} > 0$, set $K_{uv} = K_{vu} = +w_{uv}$ and $h_i = 0$. Then $\mathrm{Cut}(s) = \frac{1}{2}(W_{\text{tot}} - E(s))$, where $W_{\text{tot}} = \sum_{(u,v)\in E} w_{uv}$.
   - **Zero-Field Gaussian SK**: $K_{ij} = -g_{ij} / \sqrt{N}$ ($i < j$), $g_{ij} \sim \mathcal{N}(0, 1)$, $h_i = 0$.
2. **Continuous Potential and Phase Dynamics**:
   $$V(\phi; t) = \frac{K_c(t)}{2} \sum_{i \neq j} \tilde{K}_{ij} \cos(\phi_i - \phi_j) + K_c(t) \sum_{i=0}^{N-1} \tilde{h}_i \cos\phi_i - \frac{K_s(t)}{2} \sum_{i=0}^{N-1} \cos(2\phi_i)$$
   $$\frac{d\phi_i}{dt} = +K_c(t) \sum_{j=0}^{N-1} \tilde{K}_{ij} \sin(\phi_i - \phi_j) + K_c(t) \tilde{h}_i \sin\phi_i - K_s(t) \sin(2\phi_i) + \sigma(t)\xi_i(t)$$
   where $\tilde{K} = K / J_{\text{scale}}$, $\tilde{h} = h / J_{\text{scale}}$, and $J_{\text{scale}} = \max\!\left(\sqrt{\frac{1}{N}\sum_i \left(\sum_j K_{ij}^2 + h_i^2\right)},\; \frac{1}{2}\|K\|_2,\; 10^{-12}\right)$ is computed deterministically via 32 fixed power-iteration steps starting from $\mathbf{1}/\sqrt{N}$.
3. **Frozen Universal Annealing Schedule**:
   - Step count $S = 800$, dimensionless step $\Delta t = 0.10$, Heun (RK2) predictor-corrector integration for Phase 1 ODE benchmarks:
     - $K_c(u) = 1.0 + 0.5 u$
     - $K_s(u) = 0.05 + 0.35 u^{1.5} + 0.65 \sin\!\big(\pi \min(u / 0.95, 1)\big) \sin^2\!\big(4\pi \min(u / 0.90, 1)\big)$
     - $\sigma(u) = 0.25 (1 - u)^{1.8}$
     for normalized progress $u = \text{step} / (S - 1) \in [0, 1]$.
   - Identical schedule parameters are applied across all problem sizes ($N = 12, 24, 100, 500, 1000$) and both problem classes (Max-Cut and SK).

## 3. Problem Instances and Frozen Generation Rules

1. **Existing 24-Spin Exact Benchmarks (Read-Only)**:
   - `maxcut24`: `../tensor-maxcut-qaoa-v1/graph.json` ($N=24, |E|=32, W_{\text{tot}}=67$, exact optimal cut `62`, $E_0 = -57$).
   - `sk24`: `../tensor-sk-qaoa-v1/couplings.npy` ($N=24$, 276 pairs, exact $E_0 = -16.548016339179107$).
2. **New Frozen Benchmarks (Single Recorded OS Draw, No Instance Selection)**:
   - `maxcut12` ($N=12, |E|=18$), `maxcut100` ($N=100, |E|=300$), `maxcut500` ($N=500, |E|=1500$), `maxcut1000` ($N=1000, |E|=3000$):
     Ring $(q, (q+1)\bmod N)$ plus distinct non-self undirected chords drawn sequentially from recorded little-endian `<u8` OS random bytes, integer weights $1 + (\text{val} \bmod 4) \in \{1, 2, 3, 4\}$.
   - `sk12` ($N=12$), `sk100` ($N=100$), `sk500` ($N=500$), `sk1000` ($N=1000$):
     Zero-field Gaussian SK from recorded `<u8` OS random bytes via Box–Muller transform matching `tensor-sk-qaoa-v1/sk.py`.
   - For $N=12$ (`maxcut12`, `sk12`), enumerate all $2^{12} = 4096$ configurations via two independent methods (bitwise XOR table and spin-matrix decomposition, tolerance $10^{-10}$).

## 4. Execution, Memory Accounting, Backend Swap, and Audit Rules

1. **Single-Thread BLAS & Isolated Process Peak RSS**:
   - Fix `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=1`.
   - Run each Phase 1 problem instance in its own fresh child process so `ru_maxrss` measures true per-instance process memory without dense oracle tables or prior runs (converting bytes on macOS Darwin vs KiB on Linux).
2. **Trajectories and Classical Baseline**:
   - Evaluate `batch = 128` independent OIM trajectories per problem seeded from `instances/oim-master-seed.bin` + problem ID; report batch prefixes `16, 64, 128`, best/mean/std energy (and cut), ground hits (for $N=12, 24$), runtime, and Peak RSS.
   - Run 128-start steepest 1-spin-flip Greedy (`SHA256("HexSolid-OIM-greedy-v1:{problem_id}:{seed}")`, matching `SK24` seed rule on `sk24`) in an isolated single-thread run under identical energy scoring.
3. **Seamless Backend Swap Verification (`PhaseODEBackend` $\to$ `AudioLoopbackBackend` $\to$ `RealAcousticBackend`)**:
   - All backends implement `PhaseTransportBackend` (`reset`, `measure_phasors`) returning unit complex phasors $z_{\text{obs}} \in \mathbb{C}^{\dots \times N}$.
   - Verify `IsingAnnealer` runs unmodified with `AudioLoopbackBackend` (48 kHz `irfft` multitone synthesis with Schroeder anti-clipping phase offsets, simulated room transfer $H(f)$ + AWGN channel, single-frame calibration $H_{\text{cal}}(f_i)$, and `rfft` phase demodulation — silent in-memory, zero speaker output) on `maxcut12` (`Musical12ToneMap` C5–B5) and `maxcut24`, `sk24`, `sk500` (`TopOctaveCombMap` 10–20 kHz).
   - Provide complete `RealAcousticBackend` (PortAudio / `sounddevice` speaker+mic calibration with pilot chirp cross-correlation alignment and $H(f)$ equalization, gated behind explicit `--enable-speaker`).
4. **Independent Read-Only Audit (`audit.py`)**:
   - Re-verify all external reference hashes, frozen source hashes, raw OS byte-to-instance replays, $N=12, 24$ exact ground witnesses, independent scalar/sparse energy recomputation for all saved spin witnesses across $N=12..1000$, analytical vs finite-difference gradient of $V(\phi)$, and FFT bin orthogonality / harmonic-free spacing. Exit non-zero if any check fails.

