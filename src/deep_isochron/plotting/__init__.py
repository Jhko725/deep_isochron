"""Plotting: the project's matplotlib style, a trajectory plot, and the phase-field and
isochron figures of a reference and a trained model."""

from .isochrons import (
    HairpinMatch as HairpinMatch,
    PhaseComparison as PhaseComparison,
    plot_isochron_hairpins as plot_isochron_hairpins,
    plot_isochron_match as plot_isochron_match,
    plot_phase_comparison as plot_phase_comparison,
    plot_phase_field as plot_phase_field,
)
from .style import (
    colored_curve as colored_curve,
    configure_rcparams as configure_rcparams,
)
from .trajectories import plot_source_trajectories as plot_source_trajectories
