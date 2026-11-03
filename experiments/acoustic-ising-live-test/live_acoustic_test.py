"""Live headphone-to-microphone diagnostic and RealAcousticBackend OIM test runner.

Imports the frozen acoustic-ising-v1 modules read-only without modifying any existing files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

for _k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "acoustic-ising-v1"
sys.path.insert(0, str(V1_DIR))

import numpy as np
import sounddevice as sd

from backends.real_acoustic import RealAcousticBackend
from frequency_map import FrequencyMap, Musical12ToneMap, TopOctaveCombMap
from problem import IsingProblem, MAXCUT24_DIR, SK24_DIR, save_json, spins_to_bitstring
from schedule import AnnealingSchedule
from solver import IsingAnnealer, derive_problem_rng


def dbfs(val: float) -> float:
    return 20.0 * math.log10(max(float(val), 1e-12))


def parse_dev(val: str | None) -> int | str | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return val


def print_devices(in_dev: int | str | None, out_dev: int | str | None) -> tuple[int, int]:
    apis = sd.query_hostapis()
    devs = sd.query_devices()
    def_in, def_out = sd.default.device
    resolved_in = def_in if in_dev is None else sd.query_devices(in_dev, "input")["index"]
    resolved_out = def_out if out_dev is None else sd.query_devices(out_dev, "output")["index"]

    print("=== Audio Devices ===")
    for idx, d in enumerate(devs):
        api_name = apis[d["hostapi"]]["name"]
        markers = []
        if idx == resolved_in:
            markers.append("ACTIVE_MIC")
        if idx == resolved_out:
            markers.append("ACTIVE_HP/SPK")
        tag = f" <-- [{', '.join(markers)}]" if markers else ""
        if markers or api_name == "Windows WASAPI":
            print(
                f"  [{idx:2d}] ({api_name}) {d['name']} "
                f"(in={d['max_input_channels']}, out={d['max_output_channels']}, sr={d['default_samplerate']:.0f}){tag}"
            )
    print(f"Selected Input : [{resolved_in}] {devs[resolved_in]['name']}")
    print(f"Selected Output: [{resolved_out}] {devs[resolved_out]['name']}")
    return int(resolved_in), int(resolved_out)


def probe_band(
    label: str,
    fmap: FrequencyMap,
    output_gain: float,
    in_dev: int | str | None,
    out_dev: int | str | None,
    margin_samples: int = 14400,
    n_repeats: int = 4,
) -> dict[str, object]:
    fs = fmap.sample_rate
    # 1. Measure ambient noise floor (0.25s)
    silence_tx = np.zeros((int(0.25 * fs), 1), dtype=np.float32)
    silence_rx = sd.playrec(
        silence_tx,
        samplerate=fs,
        channels=1,
        dtype="float32",
        device=(in_dev, out_dev),
        blocking=True,
    ).ravel().astype(np.float64)
    # Skip first 50ms transient
    steady_noise = silence_rx[int(0.05 * fs) :]
    noise_rms = float(np.sqrt(np.mean(steady_noise * steady_noise)))
    noise_peak = float(np.max(np.abs(steady_noise)))

    # 2. Build pilot + zero-phase Schroeder multitone packet identical to RealAcousticBackend
    backend = RealAcousticBackend(
        frequency_map=fmap,
        allow_speaker_output=True,
        output_gain=output_gain,
        cyclic_prefix_samples=960,
        record_margin_samples=margin_samples,
        cal_averages=n_repeats,
        input_device=in_dev,
        output_device=out_dev,
    )
    backend._sd = sd
    backend._pilot = backend._build_pilot_chirp()
    payload = backend._synthesize_single(np.zeros(fmap.n_spins, dtype=np.float64))
    gap = np.zeros(240, dtype=np.float64)
    tail = np.zeros(margin_samples, dtype=np.float64)
    tx = np.concatenate([backend._pilot, gap, payload, tail]).astype(np.float32)

    latencies = []
    window_peaks = []
    window_rmses = []
    raw_peaks = []
    clipped_samples = 0
    bin_spectra = []
    noise_bin_powers = []

    # Quiet reference bins near spin bins to measure in-band noise + distortion floor
    used_set = set(int(b) for b in fmap.bins)
    b_lo = max(10, int(fmap.bins[0]) - 20)
    b_hi = min(fmap.frame_samples // 2 - 5, int(fmap.bins[-1]) + 20)
    quiet_bins = np.array([b for b in range(b_lo, b_hi) if b not in used_set and (b - 1) not in used_set and (b + 1) not in used_set], dtype=np.int64)

    for _ in range(n_repeats):
        rx = sd.playrec(
            tx[:, None],
            samplerate=fs,
            channels=1,
            dtype="float32",
            device=(in_dev, out_dev),
            blocking=True,
        ).ravel().astype(np.float64)
        raw_peaks.append(float(np.max(np.abs(rx))))
        clipped_samples += int(np.sum(np.abs(rx) >= 0.98))

        n_fft = 1 << int(np.ceil(np.log2(len(rx) + len(backend._pilot))))
        corr = np.fft.irfft(np.fft.rfft(rx, n=n_fft) * np.conj(np.fft.rfft(backend._pilot, n=n_fft)), n=n_fft)
        onset = int(np.argmax(np.abs(corr[: margin_samples + len(backend._pilot)])))
        latencies.append(onset)

        start = onset + len(backend._pilot) + len(gap) + backend.cyclic_prefix_samples
        end = start + fmap.frame_samples
        if end <= len(rx):
            win = rx[start:end]
            window_peaks.append(float(np.max(np.abs(win))))
            window_rmses.append(float(np.sqrt(np.mean(win * win))))
            spec = np.fft.rfft(win)
            bin_spectra.append(spec[fmap.bins])
            if len(quiet_bins) > 0:
                noise_bin_powers.append(float(np.mean(np.abs(spec[quiet_bins]) ** 2)))

    spec_arr = np.stack(bin_spectra, axis=0)  # (R, N)
    mean_spec = np.mean(spec_arr, axis=0)
    sig_powers = np.abs(mean_spec) ** 2
    inband_noise_p = max(float(np.mean(noise_bin_powers)) if noise_bin_powers else 1e-24, 1e-24)
    snr_db_per_bin = 10.0 * np.log10(np.maximum(sig_powers, 1e-24) / inband_noise_p)

    # Phase repeatability relative to mean phasor across repeats
    rel_phasors = spec_arr / np.maximum(mean_spec[None, :], 1e-15)
    rel_angles_deg = np.rad2deg(np.angle(rel_phasors))
    phase_std_deg_per_bin = np.std(rel_angles_deg, axis=0)

    rx_peak_max = max(raw_peaks)
    win_rms_mean = float(np.mean(window_rmses))
    lat_ms = [1000.0 * l / fs for l in latencies]
    lat_spread = max(latencies) - min(latencies)

    if clipped_samples > 0 or rx_peak_max >= 0.92:
        level_status = "TOO_LOUD_CLIPPING"
    elif win_rms_mean < noise_rms * 2.0 or float(np.min(snr_db_per_bin)) < 8.0:
        level_status = "TOO_QUIET_LOW_SNR"
    elif rx_peak_max > 0.75:
        level_status = "SLIGHTLY_HOT_BUT_OK"
    elif rx_peak_max < 0.015 and float(np.min(snr_db_per_bin)) < 18.0:
        level_status = "SLIGHTLY_QUIET"
    else:
        level_status = "OPTIMAL"

    result = {
        "band": label,
        "f_range_hz": [float(fmap.frequencies_hz[0]), float(fmap.frequencies_hz[-1])],
        "output_gain": output_gain,
        "noise_floor_rms_dbfs": dbfs(noise_rms),
        "noise_floor_peak_dbfs": dbfs(noise_peak),
        "rx_raw_peak": rx_peak_max,
        "rx_raw_peak_dbfs": dbfs(rx_peak_max),
        "rx_payload_rms": win_rms_mean,
        "rx_payload_rms_dbfs": dbfs(win_rms_mean),
        "clipped_samples": clipped_samples,
        "latencies_samples": latencies,
        "latencies_ms": [round(x, 2) for x in lat_ms],
        "latency_jitter_samples": int(lat_spread),
        "snr_db_min": float(np.min(snr_db_per_bin)),
        "snr_db_median": float(np.median(snr_db_per_bin)),
        "snr_db_max": float(np.max(snr_db_per_bin)),
        "phase_jitter_deg_median": float(np.median(phase_std_deg_per_bin)),
        "phase_jitter_deg_max": float(np.max(phase_std_deg_per_bin)),
        "level_status": level_status,
    }
    print(f"\n--- Probe [{label}] ({fmap.frequencies_hz[0]:.1f} - {fmap.frequencies_hz[-1]:.1f} Hz, gain={output_gain}) ---")
    print(f"  Noise Floor   : RMS {result['noise_floor_rms_dbfs']:.1f} dBFS (peak {result['noise_floor_peak_dbfs']:.1f} dBFS)")
    print(f"  Received Level: Peak {rx_peak_max:.4f} ({result['rx_raw_peak_dbfs']:.1f} dBFS), Payload RMS {result['rx_payload_rms_dbfs']:.1f} dBFS, Clipped={clipped_samples}")
    print(f"  Latency       : {result['latencies_ms']} ms (samples={latencies}, spread={lat_spread} samples)")
    print(f"  Bin SNR (dB)  : min={result['snr_db_min']:.1f} dB, median={result['snr_db_median']:.1f} dB, max={result['snr_db_max']:.1f} dB")
    print(f"  Phase Stability: median={result['phase_jitter_deg_median']:.2f} deg, max={result['phase_jitter_deg_max']:.2f} deg")
    print(f"  Verdict       : {level_status}")
    return result


def load_problem_and_oracle(case_name: str) -> tuple[IsingProblem, dict[str, object]]:
    inst_dir = V1_DIR / "instances"
    if case_name == "maxcut12":
        prob = IsingProblem.from_maxcut_json(inst_dir / "maxcut12-graph.json", name="maxcut12")
        ora = json.loads((inst_dir / "maxcut12-oracle.json").read_text())
        return prob, {"ground_energy": float(ora["ground_energy"]), "optimal_cut": float(ora["optimal_cut"])}
    if case_name == "sk12":
        prob = IsingProblem.from_sk_npy(inst_dir / "sk12-couplings.npy", name="sk12")
        ora = json.loads((inst_dir / "sk12-oracle.json").read_text())
        return prob, {"ground_energy": float(ora["ground_energy"]), "optimal_cut": None}
    if case_name == "maxcut24":
        prob = IsingProblem.from_maxcut_json(MAXCUT24_DIR / "graph.json", name="maxcut24")
        ora = json.loads((MAXCUT24_DIR / "oracle.json").read_text())
        opt_cut = float(ora["optimal_cut"])
        w_tot = float(ora["total_weight"])
        return prob, {"ground_energy": w_tot - 2.0 * opt_cut, "optimal_cut": opt_cut}
    if case_name == "sk24":
        prob = IsingProblem.from_sk_npy(SK24_DIR / "couplings.npy", name="sk24")
        ora = json.loads((SK24_DIR / "oracle.json").read_text())
        return prob, {"ground_energy": float(ora["ground_energy"]), "optimal_cut": None}
    raise ValueError(f"Unsupported case: {case_name}")


class SyncedAcousticBackend(RealAcousticBackend):
    """
    Physical acoustic backend refined for close-coupled headphone-to-mic & dual-USB-clock hardware:
    1. Analytic (Hilbert) chirp correlation envelope eliminates 5.5 kHz carrier-fringe hopping from cup reflections.
    2. Three fixed phase-reference OFDM pilot bins track exact integer + sub-sample timing drift delta_tau
       between separate USB DAC and USB ADC crystals on every frame.
    3. Packs batch trajectories into a single contiguous multi-symbol burst per step to avoid repeated stream re-opens.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pilot_bins: np.ndarray | None = None
        self._pilot_cal: np.ndarray | None = None
        self.last_tau_samples: float = 0.0

    @property
    def name(self) -> str:
        fmap_name = self.fmap.name if self.fmap is not None else "AutoMap"
        return f"SyncedAcousticBackend({fmap_name})"

    def _choose_pilot_bins(self) -> np.ndarray:
        assert self.fmap is not None
        used = set(int(b) for b in self.fmap.bins)
        nyq = self.fmap.frame_samples // 2
        b1 = max(28, int(self.fmap.bins[0]) - 6)
        while any(abs(b1 - u) < 3 for u in used) or any(abs((b1 - 4) - u) < 3 for u in used):
            b1 -= 1
        b0 = b1 - 4  # delta_b = 4 -> unambiguous +/- M/8 = +/- 600 samples!
        b2 = min(nyq - 20, int(self.fmap.bins[-1]) + 6)
        while any(abs(b2 - u) < 3 for u in used):
            b2 += 1
        return np.array([b0, b1, b2], dtype=np.int64)

    def _synthesize_single(self, phi_1d: np.ndarray) -> np.ndarray:
        assert self.fmap is not None and self._pilot_bins is not None
        m = self.fmap.frame_samples
        spec = np.zeros(m // 2 + 1, dtype=np.complex128)
        theta = phi_1d + self.fmap.schroeder_phases
        n_tot = self.fmap.n_spins + len(self._pilot_bins)
        amp = (0.5 * m) / np.sqrt(max(n_tot, 1))
        spec[self.fmap.bins] = amp * (np.cos(theta) + 1j * np.sin(theta))
        spec[self._pilot_bins] = (1.8 * amp) * np.array([1.0 + 0j, -1.0 + 0j, 1.0 + 0j])
        raw = np.fft.irfft(spec, n=m)
        peak = max(float(np.max(np.abs(raw))), 1e-12)
        scaled = (self.output_gain / peak) * raw
        cp = self.cyclic_prefix_samples
        return np.concatenate([scaled[-cp:], scaled]) if cp > 0 else scaled

    def _locate_analytic_onset(self, rx: np.ndarray) -> int:
        assert self._pilot is not None
        n_fft = 1 << int(np.ceil(np.log2(len(rx) + len(self._pilot))))
        p_rfft = np.fft.rfft(self._pilot, n=n_fft)
        h_analytic = np.zeros(n_fft, dtype=np.complex128)
        h_analytic[0] = p_rfft[0]
        h_analytic[1 : len(p_rfft) - 1] = 2.0 * p_rfft[1:-1]
        h_analytic[len(p_rfft) - 1] = p_rfft[-1]
        env = np.abs(np.fft.ifft(np.fft.fft(rx, n=n_fft) * np.conj(h_analytic)))
        search_len = self.record_margin_samples + len(self._pilot)
        onset = int(np.argmax(env[:search_len]))
        self._latency_samples = onset
        return onset

    def _playrec_burst(self, frames_list: list[np.ndarray]) -> list[np.ndarray]:
        """Transmit [pilot | gap | sym_0 | sym_1 | ... | margin] in a single audio call."""
        assert self.fmap is not None and self._pilot is not None and self._sd is not None
        gap = np.zeros(240, dtype=np.float64)
        tail = np.zeros(self.record_margin_samples, dtype=np.float64)
        tx = np.concatenate([self._pilot, gap, *frames_list, tail]).astype(np.float32)
        rx = self._sd.playrec(
            tx[:, None],
            samplerate=self.fmap.sample_rate,
            channels=1,
            dtype="float32",
            device=(self.input_device, self.output_device),
            blocking=True,
        ).ravel().astype(np.float64)

        onset = self._locate_analytic_onset(rx)
        sym_len = self.cyclic_prefix_samples + self.fmap.frame_samples
        base = onset + len(self._pilot) + len(gap)
        windows: list[np.ndarray] = []
        for idx in range(len(frames_list)):
            start = base + idx * sym_len + self.cyclic_prefix_samples
            end = start + self.fmap.frame_samples
            if end > len(rx):
                raise RuntimeError(f"Acoustic latency ({onset} samples) exceeded record_margin_samples")
            windows.append(rx[start:end])
        return windows

    def _estimate_shift_samples(self, pilot_ratio: np.ndarray) -> float:
        assert self.fmap is not None and self._pilot_bins is not None
        m = float(self.fmap.frame_samples)
        db_coarse = float(self._pilot_bins[1] - self._pilot_bins[0])
        dphi_coarse = float(np.angle(pilot_ratio[1] * np.conj(pilot_ratio[0])))
        tau_coarse = dphi_coarse * m / (2.0 * np.pi * db_coarse)

        db_fine = float(self._pilot_bins[2] - self._pilot_bins[0])
        expected_fine = 2.0 * np.pi * db_fine * tau_coarse / m
        measured_fine = float(np.angle(pilot_ratio[2] * np.conj(pilot_ratio[0])))
        dphi_fine = expected_fine + float(np.angle(np.exp(1j * (measured_fine - expected_fine))))
        return dphi_fine * m / (2.0 * np.pi * db_fine)

    def calibrate(self) -> np.ndarray:
        assert self.fmap is not None and self._pilot_bins is not None
        zero_sym = self._synthesize_single(np.zeros(self.fmap.n_spins, dtype=np.float64))
        windows = self._playrec_burst([zero_sym for _ in range(self.cal_averages)])
        acc_spins = np.zeros(self.fmap.n_spins, dtype=np.complex128)
        acc_pilots = np.zeros(len(self._pilot_bins), dtype=np.complex128)
        p_ref = None
        for win in windows:
            spec = np.fft.rfft(win)
            if p_ref is None:
                p_ref = spec[self._pilot_bins]
                acc_pilots += p_ref
                acc_spins += spec[self.fmap.bins]
            else:
                p_cur = spec[self._pilot_bins]
                tau = self._estimate_shift_samples(p_cur / p_ref)
                acc_spins += spec[self.fmap.bins] * np.exp(-1j * 2.0 * np.pi * self.fmap.bins * tau / self.fmap.frame_samples)
                acc_pilots += p_cur * np.exp(-1j * 2.0 * np.pi * self._pilot_bins * tau / self.fmap.frame_samples)
        self._h_cal = acc_spins / float(self.cal_averages)
        self._pilot_cal = acc_pilots / float(self.cal_averages)
        return self._h_cal

    def reset(self, n_spins: int, rng: np.random.Generator) -> None:
        del rng
        self._sd = sd
        if self._explicit_fmap is not None:
            self.fmap = self._explicit_fmap
        else:
            self.fmap = Musical12ToneMap(n_spins) if n_spins <= 12 else TopOctaveCombMap(n_spins)
        self._pilot_bins = self._choose_pilot_bins()
        self._pilot = self._build_pilot_chirp()
        self.calls = 0
        self.calibrate()

    def _demod_window(self, win: np.ndarray) -> np.ndarray:
        assert self.fmap is not None and self._h_cal is not None and self._pilot_bins is not None and self._pilot_cal is not None
        spec = np.fft.rfft(win)
        tau = self._estimate_shift_samples(spec[self._pilot_bins] / self._pilot_cal)
        self.last_tau_samples = tau
        phase_corr = np.exp(-1j * 2.0 * np.pi * self.fmap.bins * tau / self.fmap.frame_samples)
        z_eq = (spec[self.fmap.bins] / self._h_cal) * phase_corr
        return z_eq / np.maximum(np.abs(z_eq), 1e-15)

    def measure_phasors(self, phi: np.ndarray) -> np.ndarray:
        if self.fmap is None or self._h_cal is None:
            raise RuntimeError("reset() must be called before measure_phasors()")
        self.calls += 1
        if phi.ndim == 1:
            wins = self._playrec_burst([self._synthesize_single(phi)])
            return self._demod_window(wins[0])
        syms = [self._synthesize_single(phi[b]) for b in range(phi.shape[0])]
        wins = self._playrec_burst(syms)
        return np.stack([self._demod_window(w) for w in wins], axis=0)


def run_live_solve(
    case_name: str,
    output_gain: float,
    n_steps: int,
    dt: float,
    batch_size: int,
    in_dev: int | str | None,
    out_dev: int | str | None,
    margin_samples: int = 14400,
    synced: bool = True,
    f_min_hz: float = 6000.0,
    f_max_hz: float = 11900.0,
) -> dict[str, object]:
    problem, oracle = load_problem_and_oracle(case_name)
    if problem.n_spins <= 12:
        fmap = Musical12ToneMap(problem.n_spins)
    else:
        fmap = TopOctaveCombMap(problem.n_spins, f_min_hz=f_min_hz, f_max_hz=f_max_hz)

    schedule = AnnealingSchedule(n_steps=n_steps, dt=dt)
    cls = SyncedAcousticBackend if synced else RealAcousticBackend
    backend = cls(
        frequency_map=fmap,
        allow_speaker_output=True,
        output_gain=output_gain,
        cyclic_prefix_samples=960,
        record_margin_samples=margin_samples,
        cal_averages=4,
        input_device=in_dev,
        output_device=out_dev,
    )
    annealer = IsingAnnealer(schedule=schedule, backend=backend, integrator="euler")

    master_seed = (V1_DIR / "instances" / "oim-master-seed.bin").read_bytes()
    rng = derive_problem_rng(master_seed, case_name, extra_tag="phase3_real_acoustic_live")

    print(f"\n=== Running {backend.name} on {case_name} (N={problem.n_spins}, batch={batch_size}, steps={n_steps}, gain={output_gain}) ===")
    res = annealer.solve(
        problem,
        rng=rng,
        batch_size=batch_size,
        prefix_sizes=(1, batch_size) if batch_size > 1 else (1,),
        record_trajectory_index=0,
    )

    ground_e = float(oracle["ground_energy"])
    hits = int(np.sum(np.abs(res.energies - ground_e) < 1e-8))
    reached_exact = abs(res.best_energy - ground_e) < 1e-8

    print(f"\n=== Live Acoustic Result: {case_name} ===")
    print(f"  Backend          : {res.backend_name} ({backend.calls} physical play/rec calls, latency={backend._latency_samples} samples)")
    if isinstance(backend, SyncedAcousticBackend):
        print(f"  Sub-sample Drift : {backend.last_tau_samples:+.4f} samples")
    print(f"  Elapsed          : {res.seconds:.2f} s")
    print(f"  Best Energy      : {res.best_energy:.6f} (Exact Ground: {ground_e:.6f}, Match={reached_exact}, Hits={hits}/{batch_size})")
    if problem.kind == "maxcut":
        print(f"  Best Cut         : {res.best_cut:.1f} / Optimal {oracle['optimal_cut']:.1f} (Mean Cut: {res.mean_cut:.2f})")
    print(f"  Best Bitstring   : {res.best_bitstring}")

    out_payload = {
        "case": case_name,
        "n_spins": problem.n_spins,
        "backend": res.backend_name,
        "output_gain": output_gain,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "dt": dt,
        "seconds": res.seconds,
        "backend_calls": backend.calls,
        "last_latency_samples": backend._latency_samples,
        "last_latency_ms": 1000.0 * backend._latency_samples / fmap.sample_rate,
        "best_energy": res.best_energy,
        "mean_energy": res.mean_energy,
        "exact_ground_energy": ground_e,
        "reached_exact_ground": reached_exact,
        "ground_hits": hits,
        "best_cut": res.best_cut,
        "optimal_cut": oracle["optimal_cut"],
        "best_bitstring": res.best_bitstring,
        "energies": res.energies,
    }
    tag = "synced" if synced else "raw"
    save_json(THIS_DIR / "results" / f"live-{case_name}-{tag}.json", out_payload)
    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "solve"), default="probe")
    parser.add_argument("--case", default="maxcut12")
    parser.add_argument("--gain", type=float, default=0.04)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--margin", type=int, default=9600)
    parser.add_argument("--raw-backend", action="store_true")
    parser.add_argument("--f-min", type=float, default=6000.0)
    parser.add_argument("--f-max", type=float, default=11900.0)
    parser.add_argument("--in-dev", type=str, default=None)
    parser.add_argument("--out-dev", type=str, default=None)
    args = parser.parse_args()

    in_dev = parse_dev(args.in_dev)
    out_dev = parse_dev(args.out_dev)
    print_devices(in_dev, out_dev)

    if args.mode == "probe":
        p12 = probe_band("Musical12Tone (C5..B5)", Musical12ToneMap(12), args.gain, in_dev, out_dev, args.margin)
        p24_6k = probe_band("HarmonicFreeComb (6..11.9kHz)", TopOctaveCombMap(24, f_min_hz=6000.0, f_max_hz=11900.0), args.gain, in_dev, out_dev, args.margin)
        p24_10k = probe_band("TopOctaveComb (10..20kHz)", TopOctaveCombMap(24), args.gain, in_dev, out_dev, args.margin)
        save_json(THIS_DIR / "results" / "probe.json", {"musical12": p12, "comb24_6k": p24_6k, "comb24_10k": p24_10k})
    else:
        run_live_solve(
            case_name=args.case,
            output_gain=args.gain,
            n_steps=args.steps,
            dt=args.dt,
            batch_size=args.batch,
            in_dev=in_dev,
            out_dev=out_dev,
            margin_samples=args.margin,
            synced=not args.raw_backend,
            f_min_hz=args.f_min,
            f_max_hz=args.f_max,
        )


if __name__ == "__main__":
    main()

