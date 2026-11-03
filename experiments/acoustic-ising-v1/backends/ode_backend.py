"""Phase 1 backend: pure in-memory phase representation without carrier waveform synthesis."""
from __future__ import annotations

import numpy as np
from .base import PhaseTransportBackend


class PhaseODEBackend(PhaseTransportBackend):
    """Direct noiseless unit-phasor readout z = cos(phi) + 1j * sin(phi) with O(B*N) memory."""

    def __init__(self) -> None:
        self._n_spins: int = 0
        self.calls: int = 0

    @property
    def name(self) -> str:
        return "PhaseODEBackend"

    def reset(self, n_spins: int, rng: np.random.Generator) -> None:
        del rng
        self._n_spins = int(n_spins)
        self.calls = 0

    def measure_phasors(self, phi: np.ndarray) -> np.ndarray:
        if phi.shape[-1] != self._n_spins:
            raise ValueError(f"Expected last axis {self._n_spins}, got {phi.shape}")
        self.calls += 1
        return np.cos(phi) + 1j * np.sin(phi)

