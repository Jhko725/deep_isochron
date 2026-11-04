"""Plotting: the project's matplotlib style, a color scale utility, a trajectory plot,
the phase-field and isochron figures of a reference and a trained model, the
deformation figures of a planar map, the animations of a normal form's oscillator for
a talk, the conjugacy schematic, the coupling-warp figure of a spline coupling layer
beside an affine one, and the network figure of the invertible network as a chain of
linear and coupling layers."""

from .coupling_warp import (
    draw_grid_sheet as draw_grid_sheet,
    draw_identity_warp as draw_identity_warp,
    draw_warp_curve as draw_warp_curve,
    plot_coupling_warps as plot_coupling_warps,
    schematic_affine_coupling as schematic_affine_coupling,
    schematic_sheet as schematic_sheet,
    schematic_spline_coupling as schematic_spline_coupling,
    schematic_trajectories as schematic_trajectories,
    WarpLabels as WarpLabels,
    WarpStyle as WarpStyle,
)
from .deformation import (
    animate_deformation as animate_deformation,
    ANISOTROPY as ANISOTROPY,
    AREA as AREA,
    cell_jacobians as cell_jacobians,
    CellMetric as CellMetric,
    draw_sheet as draw_sheet,
    draw_trajectories as draw_trajectories,
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
from .inn_stack import (
    plot_inn_stack as plot_inn_stack,
    schematic_linear as schematic_linear,
    StackLabels as StackLabels,
    StackStyle as StackStyle,
)
from .isochrons import (
    HairpinMatch as HairpinMatch,
    PhaseComparison as PhaseComparison,
    plot_isochron_hairpins as plot_isochron_hairpins,
    plot_isochron_match as plot_isochron_match,
    plot_phase_comparison as plot_phase_comparison,
    plot_phase_field as plot_phase_field,
)
from .phase_portrait import (
    animate_isochron_sync as animate_isochron_sync,
    animate_kick as animate_kick,
    animate_phase_field as animate_phase_field,
    PortraitStyle as PortraitStyle,
)
from .scales import clipped_norm as clipped_norm
from .schematic import (
    draw_data as draw_data,
    draw_latent_initial as draw_latent_initial,
    draw_latent_trajectories as draw_latent_trajectories,
    draw_prediction as draw_prediction,
    SchematicLabels as SchematicLabels,
    SchematicStyle as SchematicStyle,
)
from .style import (
    add_colorbar as add_colorbar,
    colored_curve as colored_curve,
    configure_rcparams as configure_rcparams,
    curve_colors as curve_colors,
    set_limits as set_limits,
)
from .trajectories import (
    cycle_curve as cycle_curve,
    plot_source_trajectories as plot_source_trajectories,
)
