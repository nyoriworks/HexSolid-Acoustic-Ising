"""Pluggable phase transport backends for the Oscillator Ising Machine."""
from .base import PhaseTransportBackend
from .ode_backend import PhaseODEBackend
from .audio_loopback import AudioLoopbackBackend
from .real_acoustic import RealAcousticBackend

__all__ = [
    "PhaseTransportBackend",
    "PhaseODEBackend",
    "AudioLoopbackBackend",
    "RealAcousticBackend",
]

