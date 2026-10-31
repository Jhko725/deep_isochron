"""Deformation figures: how a map of the plane, such as a trained bijection, stretches
a mesh, small circles and a batch of trajectories, layer by layer. The subpackage has
three layers. The sheet (`sheet.py`) holds the points of a figure and maps them through
each stage of the bijection; a metric (`metrics.py`) gives each mesh cell the value that
its fill shows, from the map at the cell's center; the figures draw the stages with that
fill.
"""

from .figures import (
    animate_deformation as animate_deformation,
    draw_sheet as draw_sheet,
    plot_deformation as plot_deformation,
    plot_deformation_strip as plot_deformation_strip,
    SheetStyle as SheetStyle,
)
from .metrics import (
    ANISOTROPY as ANISOTROPY,
    AREA as AREA,
    CellMetric as CellMetric,
)
from .sheet import (
    cell_jacobians as cell_jacobians,
    Sheet as Sheet,
    sheet_limits as sheet_limits,
    SheetConfig as SheetConfig,
    stage_labels as stage_labels,
    stage_maps as stage_maps,
    stage_sheets as stage_sheets,
)
