"""Data-driven analysis: what sampled, possibly noisy trajectories can tell about the
oscillator. E1, ``estimate_cycle``, is the limit cycle from trajectory tails."""

from .cycle import (
    CycleEstimate as CycleEstimate,
    density_coefficients as density_coefficients,
    estimate_cycle as estimate_cycle,
    orientation as orientation,
    phase_from_protophase as phase_from_protophase,
    protophase as protophase,
)
