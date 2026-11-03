"""Dimensionless annealing schedule profiles for Oscillator Ising Machines."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AnnealingSchedule:
    """
    Universal dimensionless OIM schedule (Wang & Roychowdhury sub-harmonic injection locking cycles).
    Progress u = step / max(n_steps - 1, 1) in [0, 1].
    """

    n_steps: int = 800
    dt: float = 0.10
    kc_start: float = 1.0
    kc_end: float = 1.5
    ks_base_start: float = 0.05
    ks_base_end: float = 0.40
    ks_cycle_amp: float = 0.65
    ks_cycles: float = 4.0
    sigma_start: float = 0.25
    sigma_exponent: float = 1.8

    def progress(self, step: int) -> float:
        if self.n_steps <= 1:
            return 1.0
        return min(max(float(step) / float(self.n_steps - 1), 0.0), 1.0)

    def kc(self, u: float) -> float:
        return self.kc_start + (self.kc_end - self.kc_start) * u

    def ks(self, u: float) -> float:
        osc = math.sin(math.pi * self.ks_cycles * min(u / 0.90, 1.0)) ** 2
        base = self.ks_base_start + (self.ks_base_end - self.ks_base_start) * (u ** 1.5)
        amp = self.ks_cycle_amp * math.sin(math.pi * min(u / 0.95, 1.0))
        return base + amp * osc

    def sigma(self, u: float) -> float:
        if u >= 1.0:
            return 0.0
        return self.sigma_start * ((1.0 - u) ** self.sigma_exponent)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_steps": self.n_steps,
            "dt": self.dt,
            "kc_start": self.kc_start,
            "kc_end": self.kc_end,
            "ks_base_start": self.ks_base_start,
            "ks_base_end": self.ks_base_end,
            "ks_cycle_amp": self.ks_cycle_amp,
            "ks_cycles": self.ks_cycles,
            "sigma_start": self.sigma_start,
            "sigma_exponent": self.sigma_exponent,
        }

