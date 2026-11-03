"""Abstract transport interface separating Ising annealing control from numerical/physical phase propagation."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class PhaseTransportBackend(ABC):
    """
    Common boundary between IsingAnnealer and physical/numerical phase propagation.
    All backends accept phase arrays phi of shape (N,) or (B, N) in radians and return
    unit-modulus complex phasors z_obs = exp(1j * phi_obs) of identical shape.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""

    @abstractmethod
    def reset(self, n_spins: int, rng: np.random.Generator) -> None:
        """Initialize or calibrate the backend for an N-spin run without using any problem oracle."""

    @abstractmethod
    def measure_phasors(self, phi: np.ndarray) -> np.ndarray:
        """Propagate/observe current phases phi and return unit complex phasors exp(1j * phi_obs)."""

