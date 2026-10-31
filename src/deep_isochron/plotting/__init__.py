"""Plotting: the project's matplotlib style, a color scale utility, a trajectory plot,
the phase-field and isochron figures of a reference and a trained model, and the
deformation figures of a planar map."""

from .deformation import (
    animate_deformation as animate_deformation,
    ANISOTROPY as ANISOTROPY,
    AREA as AREA,
    cell_jacobians as cell_jacobians,
    CellMetric as CellMetric,
    draw_sheet as draw_sheet,
    plot_deformation as plot_deformation,
    plot_deformation_strip as plot_deformation_strip,
    Sheet as Sheet,
    sheet_limits as sheet_limits,
    SheetConfig as SheetConfig,
    SheetStyle as SheetStyle,
    stage_labels as stage_labels,
    stage_maps as stage_maps,
    stage_sheets as stage_sheets,
)
from .isochrons import (
    HairpinMatch as HairpinMatch,
    PhaseComparison as PhaseComparison,
    plot_isochron_hairpins as plot_isochron_hairpins,
    plot_isochron_match as plot_isochron_match,
    plot_phase_comparison as plot_phase_comparison,
    plot_phase_field as plot_phase_field,
)
from .scales import clipped_norm as clipped_norm
from .style import (
    add_colorbar as add_colorbar,
    colored_curve as colored_curve,
    configure_rcparams as configure_rcparams,
    set_limits as set_limits,
)
from .trajectories import plot_source_trajectories as plot_source_trajectories
