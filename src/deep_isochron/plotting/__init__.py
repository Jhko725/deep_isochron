"""Plotting: the project's matplotlib style, a trajectory plot, the phase-field and
isochron figures of a reference and a trained model, and the deformation figures of a
planar map."""

from .deformation import (
    animate_deformation as animate_deformation,
    anisotropy_scale as anisotropy_scale,
    area_scale as area_scale,
    FillScale as FillScale,
    plot_deformation as plot_deformation,
    plot_deformation_strip as plot_deformation_strip,
    SheetStyle as SheetStyle,
)
from .isochrons import (
    HairpinMatch as HairpinMatch,
    PhaseComparison as PhaseComparison,
    plot_isochron_hairpins as plot_isochron_hairpins,
    plot_isochron_match as plot_isochron_match,
    plot_phase_comparison as plot_phase_comparison,
    plot_phase_field as plot_phase_field,
)
from .sheet import (
    build_sheet as build_sheet,
    cell_values as cell_values,
    CellValues as CellValues,
    Sheet as Sheet,
    sheet_limits as sheet_limits,
    stage_labels as stage_labels,
    stage_sheets as stage_sheets,
)
from .style import (
    colored_curve as colored_curve,
    configure_rcparams as configure_rcparams,
)
from .trajectories import plot_source_trajectories as plot_source_trajectories
