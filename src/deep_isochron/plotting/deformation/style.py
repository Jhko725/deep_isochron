"""The drawing constants of the deformation figures: `SheetStyle`, which every figure
takes. A style is a frozen dataclass of Python values, changed one field at a time
with `copy.replace`; `curve_colors` is one matplotlib color, a sequence of colors or a
`ListedColormap`, which `plotting.style.curve_colors` turns into one color per
trajectory.

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from matplotlib.colors import ListedColormap
from matplotlib.typing import ColorType


@dataclass(frozen=True)
class SheetStyle:
    """The drawing constants of a deformation figure; `copy.replace` changes one.

    **Attributes:**

    - `fill_alpha`: The weight of the colormap blended over white in the fill.
    - `mesh_color`, `mesh_width`: The mesh polylines.
    - `circle_color`, `circle_width`: The circles.
    - `curve_width`: The trajectories.
    - `halo_width`: The white stroke under each trajectory, wider than `curve_width`.
    - `curve_colors`: The trajectories' colors: one matplotlib color for all of them, a
        sequence of colors, or a `ListedColormap`, whose colors are cycled in order.
    - `marker_size`: The triangle at the start of each trajectory.
    - `color_limit_quantile`: The quantile of the pooled |values| (or of the values,
        for a metric that is not symmetric) that sets the color limits; values beyond
        it are clipped.
    - `figsize_pair`: The size in inches of the domain | image pair.
    - `figsize_frame`: The size in inches of a flipbook frame.
    - `strip_panel_width`: The width in inches of each panel of a strip.
    - `strip_height`: The height in inches of a strip.
    - `colorbar_shrink`: The colorbar's length as a fraction of the panels' height.
    - `colorbar_aspect`: The ratio of the colorbar's length to its width.
    - `colorbar_pad`: The gap between the panels and the colorbar, as a fraction of
        the panels' width.
    - `colorbar_tick_fontsize`: The font size of the colorbar's tick labels, in points.
    - `colorbar_label_above`: Whether the metric's label is the colorbar's title,
        horizontal above the bar; otherwise it is the rotated label at the bar's side.
    """

    fill_alpha: float = 0.5
    mesh_color: str = "0.65"
    mesh_width: float = 0.4
    circle_color: str = "black"
    circle_width: float = 0.8
    curve_width: float = 1.6
    halo_width: float = 3.4
    curve_colors: ColorType | Sequence[ColorType] | ListedColormap = "0.1"
    marker_size: float = 7.0
    color_limit_quantile: float = 0.98
    figsize_pair: tuple[float, float] = (9.0, 4.0)
    figsize_frame: tuple[float, float] = (8.0, 4.5)
    strip_panel_width: float = 3.2
    strip_height: float = 3.8
    colorbar_shrink: float = 0.7
    colorbar_aspect: float = 25.0
    colorbar_pad: float = 0.02
    colorbar_tick_fontsize: float = 9.0
    colorbar_label_above: bool = True
