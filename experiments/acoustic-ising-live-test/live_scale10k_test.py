"""10,000-spin (scale10k) live headphone-to-mic acoustic OIM test & comparison against simulation summary.json.

Leaves all frozen files in acoustic-ising-v1 100% untouched.
"""
from __future__ import annotations

import argparse
import cmath
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
from scipy.interpolate import CubicSpline
import sounddevice as sd

from audit_scale10k import independent_seeded_maxcut_score, stream_sk_energy_o_n_memory
from backends.base import PhaseTransportBackend
from backends.ode_backend import PhaseODEBackend
from backends.audio_loopback import AudioLoopbackBackend
from frequency_map import FrequencyMap, TopOctaveCombMap
from prepare_scale10k import build_seeded_maxcut_problem, build_streamed_sk_problem
from problem import IsingProblem, save_json
from schedule import AnnealingSchedule
from solver import IsingAnnealer, derive_problem_rng


def cmath_exp(z: complex) -> complex:
    return cmath.exp(z)


def schroeder_pilot_phases(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return np.mod(np.pi * k * (k + 1.0) / max(n, 1), 2.0 * np.pi)


def fixed_reference_phases(n_spins: int) -> np.ndarray:
    """
    Deterministic pseudo-random reference phases in [0, 2*pi).
    Unlike quadratic Schroeder phases (which turn phi=0 into a slow chirp while random spins are stationary noise),
    pseudo-random reference phases give EVERY frame (calibration AND arbitrary spin states) identical stationary
    spectrum and crest factor (~4.0), preventing any USB mic DSP/AGC or transducer thermal shift.
    """
    rng = np.random.default_rng(20261103)
    return rng.uniform(0.0, 2.0 * np.pi, size=n_spins)


class DualChirp10kAcousticBackend(PhaseTransportBackend):
    """
    Physical acoustic backend capable of multiplexing 10,000 spins simultaneously across independent USB clocks:
    - Brackets each M-sample multitone frame with high-SNR start & end wideband chirps (2.0-9.0 kHz, 40 ms)
      with strict FFT bandpass filtering to measure exact DAC-vs-ADC Sampling Frequency Offset (SFO in ppm)
      to < 0.2 ppm and resample each frame via CubicSpline.
    - Uses 32 OFDM pilot bins bracketing the spin comb to lock residual sub-sample timing to < 0.002 samples.
    """

    def __init__(
        self,
        frequency_map: FrequencyMap,
        allow_speaker_output: bool = False,
        output_gain: float = 0.08,
        cyclic_prefix_samples: int = 1920,
        record_margin_samples: int = 9600,
        cal_averages: int = 3,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        same_clock: bool = False,
        out_channel: str = "lr",
    ) -> None:
        if not allow_speaker_output:
            raise PermissionError("Pass allow_speaker_output=True to emit physical audio.")
        if out_channel not in ("lr", "l", "r"):
            raise ValueError(f"Unsupported out_channel: {out_channel}")
        self.fmap = frequency_map
        self.output_gain = float(output_gain)
        self.cyclic_prefix_samples = int(cyclic_prefix_samples)
        self.same_clock = bool(same_clock)
        self.out_channel = out_channel
        self.record_margin_samples = int(record_margin_samples)
        self.cal_averages = int(cal_averages)
        self.input_device = input_device
        self.output_device = output_device

        self._ref_phases = fixed_reference_phases(self.fmap.n_spins)
        self._pilot_f0 = 2000.0
        self._pilot_f1 = 9000.0
        self._pilot = self._build_pilot_chirp()
        self._rx_pilot_template: np.ndarray | None = None
        self._template_pre_samples: int = 0
        self._cal_alpha: float | None = None
        self._pilot_bins = self._choose_pilot_bins()
        self._eq_weights: np.ndarray = np.ones(self.fmap.n_spins, dtype=np.float64)
        self._h_cal: np.ndarray | None = None
        self._pilot_cal: np.ndarray | None = None
        self.calls: int = 0
        self.last_sfo_ppm: float = 0.0
        self.last_tau_samples: float = 0.0
        self.last_rx_peak: float = 0.0
        self.cal_snr_median_db: float = 0.0
        self.cal_snr_min_db: float = 0.0
        self.cal_snr_p05_db: float = 0.0
        self.cal_phase_std_deg: float = 0.0

    @property
    def name(self) -> str:
        return f"DualChirp10kAcousticBackend({self.fmap.name}, M={self.fmap.frame_samples})"

    def _build_pilot_chirp(self) -> np.ndarray:
        """40 ms Hann-windowed wideband chirp (2.0 kHz -> 9.0 kHz) for < 0.02-sample timing lock."""
        fs = self.fmap.sample_rate
        n_p = int(0.040 * fs)  # 1920 samples
        t = np.arange(n_p, dtype=np.float64) / fs
        f0, f1 = self._pilot_f0, self._pilot_f1
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t * t) / (n_p / fs))
        chirp_amp = min(0.12, 0.65 * self.output_gain)
        return chirp_amp * np.hanning(n_p) * np.cos(phase)

    def _bandpass_pilot(self, x: np.ndarray) -> np.ndarray:
        """Strict zero-phase FFT bandpass [1800 Hz, 9200 Hz] so 50/60 Hz hum & high-band reverb cannot shift peaks."""
        n = len(x)
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fmap.sample_rate)
        mask = (freqs >= (self._pilot_f0 - 200.0)) & (freqs <= (self._pilot_f1 + 200.0))
        spec[~mask] = 0.0
        return np.fft.irfft(spec, n=n)

    def _choose_pilot_bins(self) -> np.ndarray:
        """Select 16 lower EVEN pilot bins and 16 upper EVEN pilot bins cleanly outside the 10,000-spin band."""
        used = set(int(b) for b in self.fmap.bins)
        nyq = self.fmap.frame_samples // 2
        low_pilots: list[int] = []
        b = (int(self.fmap.bins[0]) - 24) & ~1  # even bin so M//2 half-frame FFT is exact integer bin!
        while len(low_pilots) < 16 and b > 80:
            if all(abs(b - u) >= 8 for u in used):
                low_pilots.append(b)
                b -= 8  # even step = 8 bins
            else:
                b -= 2
        low_pilots.reverse()

        high_pilots: list[int] = []
        b = (int(self.fmap.bins[-1]) + 24) & ~1
        while len(high_pilots) < 16 and b < nyq - 80:
            if all(abs(b - u) >= 8 for u in used):
                high_pilots.append(b)
                b += 8
            else:
                b += 2
        return np.array(low_pilots + high_pilots, dtype=np.int64)

    def _synthesize_symbol(self, phi_1d: np.ndarray) -> np.ndarray:
        m = self.fmap.frame_samples
        spec = np.zeros(m // 2 + 1, dtype=np.complex128)
        theta = phi_1d + self._ref_phases
        w = self._eq_weights
        w_rms = float(np.sqrt(np.mean(w * w)))
        n_tot = self.fmap.n_spins + len(self._pilot_bins)
        amp = (0.5 * m) / (np.sqrt(max(n_tot, 1)) * max(w_rms, 1e-12))
        spec[self.fmap.bins] = (amp * w) * (np.cos(theta) + 1j * np.sin(theta))
        p_phases = schroeder_pilot_phases(len(self._pilot_bins))
        spec[self._pilot_bins] = (5.0 * amp) * (np.cos(p_phases) + 1j * np.sin(p_phases))
        raw = np.fft.irfft(spec, n=m)
        # Fixed gain factor (deterministic across all frames; soft clip only far in the tail so channel stays linear)
        scaled = (self.output_gain / 3.8) * raw
        scaled = np.clip(scaled, -0.85, 0.85)
        cp = self.cyclic_prefix_samples
        return np.concatenate([scaled[-cp:], scaled]) if cp > 0 else scaled

    def _locate_template_peak(self, corr_r: np.ndarray, search_lo: int, search_hi: int) -> float:
        sub = corr_r[search_lo:search_hi]
        idx = int(np.argmax(sub)) + search_lo
        lo = max(0, idx - 5)
        hi = min(len(corr_r), idx + 6)
        xs = np.arange(lo, hi, dtype=np.float64)
        cs = CubicSpline(xs, corr_r[lo:hi], bc_type="natural")
        grid = np.linspace(float(idx) - 1.5, float(idx) + 1.5, 601)
        return float(grid[int(np.argmax(cs(grid)))])

    def _analytic_envelope(self, c_rfft: np.ndarray, n_fft: int) -> np.ndarray:
        """Magnitude of complex analytic cross-correlation (eliminates 5.5 kHz carrier fringes completely!)."""
        full = np.zeros(n_fft, dtype=np.complex128)
        full[0] = c_rfft[0]
        if n_fft % 2 == 0:
            full[1 : n_fft // 2] = 2.0 * c_rfft[1 : n_fft // 2]
            full[n_fft // 2] = c_rfft[n_fft // 2]
        else:
            full[1 : (n_fft + 1) // 2] = 2.0 * c_rfft[1:]
        return np.abs(np.fft.ifft(full))

    def _refine_alpha_from_pilots(self, win: np.ndarray) -> float:
        """
        Measure residual clock drift delta_alpha directly from the 32 continuous even-bin pilot tones
        between first half win[:M//2] and second half win[M//2:]!
        """
        m = self.fmap.frame_samples
        half = m // 2
        xa = np.fft.rfft(win[:half])
        xb = np.fft.rfft(win[half:])
        q = self._pilot_bins // 2
        ratios = xb[q] * np.conj(xa[q])
        u = ratios / np.maximum(np.abs(ratios), 1e-15)
        # If sample_coords step alpha_0 = alpha_true - d_alpha is slightly too small,
        # observed pilot frequency in win is b_p * (alpha_0 / alpha_true) = b_p * (1 - d_alpha),
        # so phase in second half (n + M/2) shifts by -pi * b_p * d_alpha -> d_alpha = -angle / (pi * b_p).
        b_f = self._pilot_bins.astype(np.float64)
        # Weighted circular linear fit (unambiguous within +/- 45 ppm)
        d_alphas = -np.angle(u) / (np.pi * b_f)
        return float(np.median(d_alphas))

    def _playrec_and_resample(self, symbol_with_cp: np.ndarray) -> np.ndarray:
        gap1 = np.zeros(1440, dtype=np.float64)  # 30 ms guard after start chirp so chirp reverb dies before CP
        tail = np.zeros(self.record_margin_samples, dtype=np.float64)
        if self.same_clock:
            tx = np.concatenate([self._pilot, gap1, symbol_with_cp, tail]).astype(np.float32)
            d_tx = 0
        else:
            gap2 = np.zeros(2400, dtype=np.float64)  # 50 ms quiet guard so symbol reverb dies before end chirp
            tx = np.concatenate([self._pilot, gap1, symbol_with_cp, gap2, self._pilot, tail]).astype(np.float32)
            d_tx = len(self._pilot) + len(gap1) + len(symbol_with_cp) + len(gap2)

        if self.out_channel == "l":
            tx_out = np.column_stack([tx, np.zeros_like(tx)])
        elif self.out_channel == "r":
            tx_out = np.column_stack([np.zeros_like(tx), tx])
        else:
            tx_out = np.column_stack([tx, tx])

        rx = sd.playrec(
            tx_out,
            samplerate=self.fmap.sample_rate,
            channels=1,
            dtype="float32",
            device=(self.input_device, self.output_device),
            blocking=True,
        ).ravel().astype(np.float64)
        self.last_rx_peak = float(np.max(np.abs(rx)))

        rx_bp = self._bandpass_pilot(rx)

        if self._rx_pilot_template is None:
            n_fft0 = 1 << int(np.ceil(np.log2(len(rx_bp) + len(self._pilot))))
            env0 = self._analytic_envelope(
                np.fft.rfft(rx_bp, n=n_fft0) * np.conj(np.fft.rfft(self._pilot, n=n_fft0)), n_fft0
            )
            coarse0 = int(np.argmax(env0[: self.record_margin_samples + len(self._pilot)]))
            t_start = max(0, coarse0 - 48)
            t_end = min(len(rx_bp), coarse0 + len(self._pilot) + 240)
            self._rx_pilot_template = rx_bp[t_start:t_end].copy()
            self._template_pre_samples = coarse0 - t_start

        tmpl = self._rx_pilot_template
        n_fft = 1 << int(np.ceil(np.log2(len(rx_bp) + len(tmpl))))
        env = self._analytic_envelope(
            np.fft.rfft(rx_bp, n=n_fft) * np.conj(np.fft.rfft(tmpl, n=n_fft)), n_fft
        )

        t1_tmpl = self._locate_template_peak(env, 0, self.record_margin_samples + len(tmpl))
        t1_onset = t1_tmpl + float(self._template_pre_samples)

        m = self.fmap.frame_samples
        offset_tx = len(self._pilot) + len(gap1) + self.cyclic_prefix_samples

        if self.same_clock:
            # Shared DAC/ADC crystal clock -> alpha === 1.000000 exact! Zero spline distortion;
            # slice integer samples directly and let the 32 OFDM pilots cancel fractional sample offset exactly!
            self.last_sfo_ppm = 0.0
            s0 = max(0, int(round(t1_onset)) + offset_tx)
            return rx[s0 : s0 + m]

        expected_t2 = int(round(t1_tmpl + d_tx))
        t2_tmpl = self._locate_template_peak(
            env,
            max(0, expected_t2 - 200),
            min(len(rx_bp), expected_t2 + 200),
        )
        alpha = (t2_tmpl - t1_tmpl) / float(d_tx)

        i_lo = max(0, int(math.floor(t1_onset + alpha * offset_tx)) - 8)
        i_hi = min(len(rx), int(math.ceil(t1_onset + alpha * (offset_tx + m))) + 9)
        cs = CubicSpline(np.arange(i_lo, i_hi, dtype=np.float64), rx[i_lo:i_hi], bc_type="natural")

        n_idx = offset_tx + np.arange(m, dtype=np.float64)
        win = cs(t1_onset + alpha * n_idx)
        self.last_sfo_ppm = (alpha - 1.0) * 1e6
        return win

    def _estimate_shift_samples(self, pilot_ratio: np.ndarray) -> float:
        """
        Estimate residual timing offset tau from the 32 pilot subcarriers!
        Step 1: Coarse tau from adjacent 7-bin pilot pairs.
        Step 2: Fine tau across the full 10,000-spin span (lower 16 vs upper 16 pilots).
        """
        m = float(self.fmap.frame_samples)
        bins_p = self._pilot_bins.astype(np.float64)
        u = pilot_ratio / np.maximum(np.abs(pilot_ratio), 1e-15)
        adj_diffs = []
        step_bins = None
        for idx in list(range(0, 15)) + list(range(16, 31)):
            dk = bins_p[idx + 1] - bins_p[idx]
            if step_bins is None:
                step_bins = dk
            if abs(dk - step_bins) < 1e-6:
                adj_diffs.append(u[idx + 1] * np.conj(u[idx]))
        assert step_bins is not None
        mean_adj = np.mean(adj_diffs)
        tau_coarse = float(np.angle(mean_adj)) * m / (2.0 * math.pi * step_bins)

        rem = u * np.exp(-1j * 2.0 * np.pi * bins_p * tau_coarse / m)
        low_mean = np.mean(rem[:16])
        high_mean = np.mean(rem[16:])
        k_low = float(np.mean(bins_p[:16]))
        k_high = float(np.mean(bins_p[16:]))
        dphi_lh = float(np.angle(high_mean * np.conj(low_mean)))
        tau_fine = tau_coarse + dphi_lh * m / (2.0 * math.pi * (k_high - k_low))
        return tau_fine

    def calibrate(self) -> np.ndarray:
        self._eq_weights = np.ones(self.fmap.n_spins, dtype=np.float64)
        zero_sym = self._synthesize_symbol(np.zeros(self.fmap.n_spins, dtype=np.float64))
        aligned_spins = []
        acc_pilots = np.zeros(len(self._pilot_bins), dtype=np.complex128)
        noise_powers = []
        alphas = []

        used = set(int(b) for b in self.fmap.bins) | set(int(b) for b in self._pilot_bins)
        # Measure in-band quiet floor using unused bins inside or flanking the comb!
        inband_quiet = [
            b for b in range(int(self.fmap.bins[0]), int(self.fmap.bins[-1])) if b not in used
        ]
        if len(inband_quiet) >= 100:
            quiet_bins = np.array(inband_quiet[:: max(1, len(inband_quiet) // 2000)], dtype=np.int64)
        else:
            q_lo = max(40, int(self.fmap.bins[0]) - 300)
            q_hi = int(self.fmap.bins[0]) - 40
            quiet_bins = np.array([b for b in range(q_lo, q_hi) if b not in used], dtype=np.int64)

        p_ref = None
        for _ in range(self.cal_averages):
            win = self._playrec_and_resample(zero_sym)
            alphas.append(1.0 + self.last_sfo_ppm * 1e-6)
            spec = np.fft.rfft(win)
            if len(quiet_bins) > 0:
                noise_powers.append(float(np.median(np.abs(spec[quiet_bins]) ** 2)))
            if p_ref is None:
                p_ref = spec[self._pilot_bins]
                acc_pilots += p_ref
                aligned_spins.append(spec[self.fmap.bins])
            else:
                p_cur = spec[self._pilot_bins]
                tau = self._estimate_shift_samples(p_cur / p_ref)
                aligned_spins.append(
                    spec[self.fmap.bins] * np.exp(-1j * 2.0 * np.pi * self.fmap.bins * tau / self.fmap.frame_samples)
                )
                acc_pilots += p_cur * np.exp(-1j * 2.0 * np.pi * self._pilot_bins * tau / self.fmap.frame_samples)

        self._cal_alpha = float(np.median(alphas))
        arr = np.stack(aligned_spins, axis=0)
        h_cal = np.mean(arr, axis=0)
        self._h_cal = h_cal
        self._pilot_cal = acc_pilots / float(self.cal_averages)

        sig_p = np.abs(h_cal) ** 2
        n_p = max(float(np.mean(noise_powers)) if noise_powers else 1e-24, 1e-24)
        snr_bins = 10.0 * np.log10(np.maximum(sig_p, 1e-24) / n_p)
        self.cal_snr_median_db = float(np.median(snr_bins))
        self.cal_snr_p05_db = float(np.percentile(snr_bins, 5.0))
        self.cal_snr_min_db = float(np.min(snr_bins))
        if len(arr) > 1:
            rel_deg = np.rad2deg(np.angle(arr / np.maximum(h_cal[None, :], 1e-15)))
            self.cal_phase_std_deg = float(np.median(np.std(rel_deg, axis=0)))
        return h_cal

    def reset(self, n_spins: int, rng: np.random.Generator) -> None:
        del rng
        if n_spins != self.fmap.n_spins:
            raise ValueError(f"Expected {self.fmap.n_spins} spins, got {n_spins}")
        self._rx_pilot_template = None
        self._cal_alpha = None
        self.calls = 0
        self.calibrate()

    def _demod_one(self, phi_1d: np.ndarray) -> np.ndarray:
        assert self._h_cal is not None and self._pilot_cal is not None
        win = self._playrec_and_resample(self._synthesize_symbol(phi_1d))
        spec = np.fft.rfft(win)
        tau = self._estimate_shift_samples(spec[self._pilot_bins] / self._pilot_cal)
        self.last_tau_samples = tau
        corr = np.exp(-1j * 2.0 * np.pi * self.fmap.bins * tau / self.fmap.frame_samples)
        z_eq = (spec[self.fmap.bins] / self._h_cal) * corr
        return z_eq / np.maximum(np.abs(z_eq), 1e-15)

    def measure_phasors(self, phi: np.ndarray) -> np.ndarray:
        self.calls += 1
        out = self._demod_one(phi) if phi.ndim == 1 else np.stack([self._demod_one(phi[b]) for b in range(phi.shape[0])], axis=0)
        if self.calls == 1 or self.calls % 10 == 0:
            print(
                f"  [Acoustic 10k Frame {self.calls:3d}] rx_peak={self.last_rx_peak:.4f}, "
                f"SFO={self.last_sfo_ppm:+.2f} ppm, tau={self.last_tau_samples:+.4f} samp",
                flush=True,
            )
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe10k", "solve10k", "sim_check"), default="probe10k")
    parser.add_argument("--case", choices=("maxcut10000", "sk10000"), default="maxcut10000")
    parser.add_argument("--gain", type=float, default=0.06)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--dt", type=float, default=0.55)
    parser.add_argument("--frame-samples", type=int, default=96000)
    parser.add_argument("--f-min", type=float, default=5500.0)
    parser.add_argument("--f-max", type=float, default=11000.0)
    parser.add_argument("--in-dev", type=int, default=15)
    parser.add_argument("--out-dev", type=int, default=12)
    parser.add_argument("--same-clock", action="store_true")
    parser.add_argument("--out-ch", choices=("lr", "l", "r"), default="lr")
    args = parser.parse_args()

    seeds_doc = json.loads((V1_DIR / "instances_scale10k" / "seeds.json").read_text())
    sim_summary = json.loads((V1_DIR / "results_scale10k" / "summary.json").read_text())
    master_seed = bytes.fromhex(str(seeds_doc["solver_master_seed_hex"]))

    if args.mode == "sim_check":
        for cname in ("maxcut10000", "sk10000"):
            meta = seeds_doc["instances"][cname]
            prob = (
                build_seeded_maxcut_problem(10000, int(meta["n_edges"]), str(meta["seed_hex"]), name=cname)
                if meta["kind"] == "maxcut"
                else build_streamed_sk_problem(10000, str(meta["seed_hex"]), name=cname)
            )
            grid = [(60, 0.50), (70, 0.50), (80, 0.50), (85, 0.48), (90, 0.46), (100, 0.45)] if cname == "maxcut10000" else [(50, 0.30), (65, 0.28)]
            for s, dt in grid:
                rng = derive_problem_rng(master_seed, cname, extra_tag="scale10k_loopback_maxcut10000_comb" if cname == "maxcut10000" else "scale10k_ode")
                r = IsingAnnealer(AnnealingSchedule(n_steps=s, dt=dt), PhaseODEBackend(), "euler").solve(prob, rng, batch_size=1, prefix_sizes=(1,))
                metric = f"cut={r.best_cut:.1f}" if cname == "maxcut10000" else f"E={r.best_energy:.2f}"
                print(f"{cname} ODE (steps={s}, dt={dt}) -> {metric}")
        return

    fmap = TopOctaveCombMap(
        n_spins=10000,
        sample_rate=48000,
        frame_samples=args.frame_samples,
        f_min_hz=args.f_min,
        f_max_hz=args.f_max,
    )

    if args.mode == "probe10k":
        backend = DualChirp10kAcousticBackend(
            frequency_map=fmap,
            allow_speaker_output=True,
            output_gain=args.gain,
            cal_averages=3,
            input_device=args.in_dev,
            output_device=args.out_dev,
            same_clock=args.same_clock,
            out_channel=args.out_ch,
        )
        backend.reset(10000, np.random.default_rng(0))
        # Test random binary spin readout accuracy across all 10,000 spins in a single physical frame!
        rng_test = np.random.default_rng(42)
        true_bits = rng_test.integers(0, 2, size=10000)
        phi_tx = np.where(true_bits == 0, 0.0, np.pi)
        z_rx = backend.measure_phasors(phi_tx)
        rx_bits = np.where(z_rx.real >= 0.0, 0, 1)
        bit_acc = float(np.mean(rx_bits == true_bits))
        phase_err_deg = np.rad2deg(np.angle(z_rx * np.exp(-1j * phi_tx)))

        print(f"=== 10,000-Spin Physical Acoustic Link Probe ({args.f_min:.0f}-{args.f_max:.0f} Hz, M={args.frame_samples}, gain={args.gain}, ch={args.out_ch.upper()}) ===")
        print(f"  Harmonic-Free Check  : {fmap.validate()['passed']} (df={fmap.bin_hz:.3f} Hz, T_frame={fmap.frame_duration_seconds:.2f} s)")
        print(f"  Received Peak        : {backend.last_rx_peak:.4f} ({20*math.log10(max(backend.last_rx_peak,1e-12)):.1f} dBFS)")
        print(f"  Measured USB SFO     : {backend.last_sfo_ppm:+.2f} ppm (residual tau={backend.last_tau_samples:+.4f} samples)")
        print(f"  10,000-Bin SNR       : min={backend.cal_snr_min_db:.1f} dB, p05={backend.cal_snr_p05_db:.1f} dB, median={backend.cal_snr_median_db:.1f} dB")
        print(f"  Cal Phase Repeat std : median={backend.cal_phase_std_deg:.2f} deg")
        print(f"  10,000-Spin BPSK Acc : {bit_acc*100:.2f}% ({int(np.sum(rx_bits == true_bits))}/10000 spins correct in 1 frame, phase_std={float(np.std(phase_err_deg)):.2f} deg)")
        return

    # Live 10,000-spin OIM optimization + direct comparison with same-schedule ODE/Loopback & summary.json!
    meta = seeds_doc["instances"][args.case]
    if meta["kind"] == "maxcut":
        problem = build_seeded_maxcut_problem(10000, int(meta["n_edges"]), str(meta["seed_hex"]), name=args.case)
        extra_tag = "scale10k_loopback_maxcut10000_comb"
    else:
        problem = build_streamed_sk_problem(10000, str(meta["seed_hex"]), name=args.case)
        extra_tag = "scale10k_ode"

    schedule = AnnealingSchedule(n_steps=args.steps, dt=args.dt)

    # 1. Exact matched-schedule simulation baselines (ODE & virtual AudioLoopback with identical seed & steps)
    rng_ode = derive_problem_rng(master_seed, args.case, extra_tag=extra_tag)
    ode_same = IsingAnnealer(schedule, PhaseODEBackend(), "euler").solve(problem, rng_ode, batch_size=1, prefix_sizes=(1,))

    rng_lb = derive_problem_rng(master_seed, args.case, extra_tag=extra_tag)
    lb_same = IsingAnnealer(
        schedule,
        AudioLoopbackBackend(frequency_map=fmap, cyclic_prefix_samples=480, snr_db=42.0, cal_averages=2),
        "euler",
    ).solve(problem, rng_lb, batch_size=1, prefix_sizes=(1,))

    # 2. Real physical headphone -> air -> microphone 10,000-spin OIM solve!
    backend = DualChirp10kAcousticBackend(
        frequency_map=fmap,
        allow_speaker_output=True,
        output_gain=args.gain,
        cal_averages=2,
        input_device=args.in_dev,
        output_device=args.out_dev,
        same_clock=args.same_clock,
        out_channel=args.out_ch,
    )
    rng_real = derive_problem_rng(master_seed, args.case, extra_tag=extra_tag)
    print(
        f"\n=== Starting Physical 10,000-Spin Acoustic OIM: {args.case} "
        f"(steps={args.steps}, dt={args.dt}, gain={args.gain}, ch={args.out_ch.upper()}) ===",
        flush=True,
    )
    res_real = IsingAnnealer(schedule, backend, "euler").solve(problem, rng_real, batch_size=1, prefix_sizes=(1,))

    # 3. Independent streaming audit (exact functions from audit_scale10k.py!)
    if meta["kind"] == "maxcut":
        audit_e, audit_cut = independent_seeded_maxcut_score(
            10000, int(meta["n_edges"]), str(meta["seed_hex"]), res_real.best_bitstring
        )
        audit_ok = abs(audit_e - res_real.best_energy) < 1e-8 and abs(audit_cut - float(res_real.best_cut)) < 1e-8
    else:
        audit_e = stream_sk_energy_o_n_memory(10000, str(meta["seed_hex"]), res_real.best_bitstring)
        audit_cut = None
        audit_ok = abs(audit_e - res_real.best_energy) < 1e-7

    # 4. Pull frozen 800-step benchmark references from results_scale10k/summary.json
    p1_ref = next(r for r in sim_summary["phase1_ode_scale10k"] if r["case"] == args.case)
    p2_ref = next((r for r in sim_summary["phase2_loopback_scale10k"] if r["case"] == args.case), None)

    hamming_agree_ode = float(np.mean(np.array(list(res_real.best_bitstring)) == np.array(list(ode_same.best_bitstring))))
    hamming_agree_ode = max(hamming_agree_ode, 1.0 - hamming_agree_ode)  # Z2 spin-flip symmetry

    out = {
        "case": args.case,
        "n_spins": 10000,
        "schedule": {"n_steps": args.steps, "dt": args.dt, "integrator": "euler"},
        "acoustic_hardware": {
            "backend": backend.name,
            "f_range_hz": [args.f_min, args.f_max],
            "frame_samples": args.frame_samples,
            "frame_duration_seconds": fmap.frame_duration_seconds,
            "output_gain": args.gain,
            "output_channel": args.out_ch,
            "rx_peak": backend.last_rx_peak,
            "measured_usb_sfo_ppm": backend.last_sfo_ppm,
            "cal_snr_min_db": backend.cal_snr_min_db,
            "cal_snr_p05_db": backend.cal_snr_p05_db,
            "cal_snr_median_db": backend.cal_snr_median_db,
            "cal_phase_std_deg": backend.cal_phase_std_deg,
            "wall_seconds": res_real.seconds,
            "backend_calls": backend.calls,
        },
        "real_acoustic_10k": {
            "best_energy": res_real.best_energy,
            "best_cut": res_real.best_cut,
            "independent_audit_passed": audit_ok,
            "independent_audit_energy": audit_e,
            "independent_audit_cut": audit_cut,
            "spin_agreement_with_matched_ode": hamming_agree_ode,
        },
        "matched_step_simulation": {
            "ode_euler_same_seed": {
                "best_energy": ode_same.best_energy,
                "best_cut": ode_same.best_cut,
            },
            "virtual_loopback_same_seed": {
                "best_energy": lb_same.best_energy,
                "best_cut": lb_same.best_cut,
            },
        },
        "frozen_summary_800step_reference": {
            "greedy_baseline": {
                "n_starts": p1_ref["greedy"]["n_starts"],
                "best_energy": p1_ref["greedy"]["best_energy"],
                "mean_energy": p1_ref["greedy"]["mean_energy"],
                "best_cut": p1_ref["greedy"].get("best_cut"),
                "mean_cut": p1_ref["greedy"].get("mean_cut"),
            },
            "phase1_ode_800step": {
                "batch_size": p1_ref["oim"]["batch_size"],
                "best_energy": p1_ref["oim"]["best_energy"],
                "mean_energy": p1_ref["oim"]["mean_energy"],
                "best_cut": p1_ref["oim"].get("best_cut"),
                "mean_cut": p1_ref["oim"].get("mean_cut"),
            },
            "phase2_loopback_800step": (
                {
                    "batch_size": p2_ref["batch_size"],
                    "best_energy": p2_ref["best_energy"],
                    "mean_energy": p2_ref["mean_energy"],
                    "best_cut": p2_ref.get("best_cut"),
                    "mean_cut": p2_ref.get("mean_cut"),
                }
                if p2_ref is not None
                else None
            ),
        },
        "best_bitstring": res_real.best_bitstring,
    }
    suffix = "-sameclock" if args.same_clock else ""
    ch_suffix = f"-{args.out_ch}" if args.out_ch != "lr" else ""
    out_file = THIS_DIR / "results" / f"live10k-{args.case}{suffix}{ch_suffix}.json"
    save_json(out_file, out)
    print(json.dumps({k: v for k, v in out.items() if k != "best_bitstring"}, indent=2))


if __name__ == "__main__":
    main()
