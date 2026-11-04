"""The panels of the conjugacy schematic, the method figure that draws the training
pipeline as a commutative square: its strings, its drawing constants and one drawer
per panel.

`SchematicLabels` holds every string of the figure as raw mathtext, and
`SchematicStyle` holds the colors, line widths, font sizes and geometry, with the
labels as one field; both are frozen dataclasses changed with `copy.replace`. The four
drawers are pure matplotlib over NumPy points: `draw_data` draws the data windows with
filled start markers, `draw_latent_initial` the latent initial conditions as open
markers, `draw_latent_trajectories` the normal form's trajectories dashed from them,
and `draw_prediction` the model's predictions dashed over the faded data. Each drawer
also marks the equilibrium of its panel, and none sets limits or hides the axes. The
figure function that samples the model and assembles the panels is to come.

The design and its decisions are in `docs/changes/2026-11-04-conjugacy-schematic.md`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import colorcet as cc
import matplotlib.patheffects as pe
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.typing import ColorType

from .style import curve_colors


@dataclass(frozen=True)
class SchematicLabels:
    r"""The strings of the conjugacy schematic, as raw mathtext; `copy.replace` changes
    one.

    **Attributes:**

    - `data_title`: The title of the data panel, top left.
    - `prediction_title`: The title of the prediction panel, top right.
    - `latent_initial_title`: The title of the latent initial-condition panel, bottom
        left.
    - `latent_trajectories_title`: The title of the latent trajectory panel, bottom
        right.
    - `map`: The name in the block of the bijection, on the left between the rows.
    - `map_formula`: The formula under `map`.
    - `inverse`: The name in the block of the inverse, on the right between the rows.
    - `inverse_formula`: The formula under `inverse`.
    - `latent`: The name in the normal-form block, between the latent panels.
    - `latent_formula`: The formula under `latent`.
    - `flow`: The label of the arrow of the true flow, between the data panels.
    - `loss`: The loss, under the arrow of the true flow.
    - `shared`: The note that joins the blocks of the bijection and its inverse.
    - `legend`: The line under the figure that explains the line styles and markers.
    """

    data_title: str = r"Data $x(t)$"
    prediction_title: str = r"Prediction $\hat{x}(t)$ vs data"
    latent_initial_title: str = r"Latent initial conditions $z_0$"
    latent_trajectories_title: str = r"Latent trajectories $z(t)$"
    map: str = r"INN $h$"
    map_formula: str = r"$x_0 \mapsto z_0 = h(x_0)$"
    inverse: str = r"INN $h^{-1}$"
    inverse_formula: str = r"$z(t) \mapsto \hat{x}(t)$"
    latent: str = "Normal form"
    latent_formula: str = r"ODE solve, $\theta = (a, b, \omega)$"
    flow: str = r"true flow $\varphi^t$ (unknown)"
    loss: str = r"$\mathcal{L} = \sum_k \|\hat{x}(t_k) - x(t_k)\|^2$"
    shared: str = "shared weights"
    legend: str = r"solid = data, dashed = model; filled dot = $x_0$, open dot = $z_0$"


@dataclass(frozen=True)
class SchematicStyle:
    """The drawing constants of the conjugacy schematic; `copy.replace` changes one.

    **Attributes:**

    - `labels`: The strings of the figure.
    - `curve_colors`: The colors of the windows: one matplotlib color for all of them,
        a sequence of colors, or a `ListedColormap`, whose colors are cycled in order.
    - `curve_width`: The width of the data, latent and predicted curves, in points.
    - `halo_width`: The width of the white stroke under each curve, in points, wider
        than `curve_width`.
    - `dash`: The dash pattern `(on, off)` of the model's curves, in multiples of the
        line width.
    - `data_alpha`: The opacity of the data drawn under the predictions.
    - `marker_size`: The diameter of the start markers, in points.
    - `marker_edge_width`: The edge width of the start markers, in points.
    - `cycle_color`: The color of the limit cycle, the latent circle and the
        equilibrium markers.
    - `cycle_width`: The width of the limit cycle and the latent circle, in points.
    - `equilibrium_size`: The diameter of the open equilibrium marker, in points.
    - `margin`: The fraction of a panel box's span added on each side.
    - `panel_title_size`: The font size of the panel titles, in points.
    - `block_title_size`: The font size of the block names, in points.
    - `block_formula_size`: The font size of the block formulas, in points.
    - `annotation_size`: The font size of the flow label, the loss and the
        shared-weights note, in points.
    - `legend_size`: The font size of the legend line, in points.
    - `figsize`: The size of the figure, in inches.
    - `panel_edge`: The edge of each square panel, in inches.
    - `figure_margin`: The margin around the panels, in inches.
    - `block_size`: The width and height of the blocks of the bijection and its
        inverse, in inches.
    - `latent_block_width`: The width of the normal-form block, in inches.
    - `arrow_gap`: The gap between an arrow's end and the panel or block it points to,
        in inches.
    - `loss_offset`: The distance of the loss below the arrow of the true flow, in
        inches.

    **Raises:**

    `ValueError` naming the field if `margin` is negative, a width or size is not
    positive, or `data_alpha` is outside `[0, 1]`.
    """

    labels: SchematicLabels = SchematicLabels()
    # A factory, since a colormap defines `__eq__` and so is unhashable to dataclasses.
    curve_colors: ColorType | Sequence[ColorType] | ListedColormap = field(
        default_factory=lambda: cc.cm.glasbey_dark
    )
    curve_width: float = 2.2
    halo_width: float = 4.0
    dash: tuple[float, float] = (3.0, 1.5)
    data_alpha: float = 0.3
    marker_size: float = 7.0
    marker_edge_width: float = 1.8
    cycle_color: str = "0.6"
    cycle_width: float = 1.0
    equilibrium_size: float = 3.0
    margin: float = 0.08
    panel_title_size: float = 10.0
    block_title_size: float = 11.0
    block_formula_size: float = 8.0
    annotation_size: float = 9.0
    legend_size: float = 8.0
    figsize: tuple[float, float] = (7.4, 6.2)
    panel_edge: float = 2.3
    figure_margin: float = 0.35
    block_size: tuple[float, float] = (1.3, 0.56)
    latent_block_width: float = 1.5
    arrow_gap: float = 0.04
    loss_offset: float = 0.15

    def __post_init__(self) -> None:
        """Check every validated field's range."""
        if self.margin < 0:
            raise ValueError(f"margin must not be negative, got {self.margin}.")
        if not 0 <= self.data_alpha <= 1:
            raise ValueError(f"data_alpha must be in [0, 1], got {self.data_alpha}.")
        for name in (
            "curve_width",
            "halo_width",
            "marker_size",
            "marker_edge_width",
            "cycle_width",
            "equilibrium_size",
            "panel_title_size",
            "block_title_size",
            "block_formula_size",
            "annotation_size",
            "legend_size",
            "figsize",
            "panel_edge",
            "block_size",
            "latent_block_width",
        ):
            value = getattr(self, name)
            if min(value if isinstance(value, tuple) else (value,)) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


def _curves(
    ax: Axes,
    points: Float[np.ndarray, "q time 2"],
    colors: list,
    style: SchematicStyle,
    *,
    dashed: bool,
    alpha: float,
    halo: bool,
) -> None:
    """Draw each window of `points` as a curve in its color of `colors`.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `points`: The windows.
    - `colors`: One color per window.
    - `style`: The drawing constants.
    - `dashed`: Whether the curves take `style.dash`, the line style of the model's
        curves; otherwise they are solid.
    - `alpha`: The opacity of the curves.
    - `halo`: Whether a white stroke of `style.halo_width` is drawn under each curve.
    """
    linestyle = (0, style.dash) if dashed else "-"
    path_effects = (
        [pe.withStroke(linewidth=style.halo_width, foreground="white")] if halo else []
    )
    for curve, color in zip(points, colors, strict=True):
        ax.plot(
            *curve.T,
            color=color,
            linewidth=style.curve_width,
            linestyle=linestyle,
            alpha=alpha,
            path_effects=path_effects,
            zorder=3,
        )


def _start_markers(
    ax: Axes,
    points: Float[np.ndarray, "q 2"],
    colors: list,
    style: SchematicStyle,
    *,
    filled: bool,
) -> None:
    """Draw a circle at each of `points`, edged in its color of `colors`.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `points`: The start points, one per window.
    - `colors`: One color per point.
    - `style`: The drawing constants.
    - `filled`: Whether a circle is filled with its color, as an initial condition in
        the data space; otherwise it is filled white, as one in the latent space.
    """
    for point, color in zip(points, colors, strict=True):
        ax.plot(
            *point,
            marker="o",
            markersize=style.marker_size,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=style.marker_edge_width,
            linestyle="none",
            zorder=5,
        )


def _cycle_and_equilibrium(
    ax: Axes,
    cycle_points: Float[np.ndarray, "n 2"],
    equilibrium: Float[np.ndarray, " 2"],
    style: SchematicStyle,
) -> None:
    """Draw the closed polyline `cycle_points` and an open circle at `equilibrium`,
    both in `style.cycle_color`."""
    ax.plot(
        *cycle_points.T,
        color=style.cycle_color,
        linewidth=style.cycle_width,
        zorder=1,
    )
    ax.plot(
        *equilibrium,
        marker="o",
        markersize=style.equilibrium_size,
        markerfacecolor="white",
        markeredgecolor=style.cycle_color,
        linestyle="none",
        zorder=4,
    )


def draw_data(
    ax: Axes,
    data: Float[np.ndarray, "q time 2"],
    cycle_points: Float[np.ndarray, "n 2"],
    equilibrium: Float[np.ndarray, " 2"],
    style: SchematicStyle = SchematicStyle(),
) -> None:
    """Draw the data panel on `ax`: the limit cycle, the source's equilibrium, each
    window as a solid curve with a white halo, and a filled circle at its start.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `data`: The windows, each colored by `curve_colors(style.curve_colors, q)`.
    - `cycle_points`: The closed limit cycle, such as the result of `cycle_curve`.
    - `equilibrium`: The source's equilibrium, the center of the cycle.
    - `style`: The drawing constants.
    """
    colors = curve_colors(style.curve_colors, len(data))
    _cycle_and_equilibrium(ax, cycle_points, equilibrium, style)
    _curves(ax, data, colors, style, dashed=False, alpha=1.0, halo=True)
    _start_markers(ax, data[:, 0], colors, style, filled=True)


def draw_latent_initial(
    ax: Axes,
    z0s: Float[np.ndarray, "q 2"],
    circle_points: Float[np.ndarray, "n 2"],
    style: SchematicStyle = SchematicStyle(),
) -> None:
    """Draw the latent initial-condition panel on `ax`: the normal form's cycle, its
    equilibrium at the origin, and an open circle at each initial condition.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `z0s`: The latent initial conditions, one per window, each colored by
        `curve_colors(style.curve_colors, q)`.
    - `circle_points`: The normal form's closed limit cycle.
    - `style`: The drawing constants.
    """
    colors = curve_colors(style.curve_colors, len(z0s))
    _cycle_and_equilibrium(ax, circle_points, np.zeros(2), style)
    _start_markers(ax, z0s, colors, style, filled=False)


def draw_latent_trajectories(
    ax: Axes,
    latents: Float[np.ndarray, "q time 2"],
    circle_points: Float[np.ndarray, "n 2"],
    style: SchematicStyle = SchematicStyle(),
) -> None:
    """Draw the latent trajectory panel on `ax`: the normal form's cycle, its
    equilibrium at the origin, each trajectory dashed with a white halo, and an open
    circle at its start.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `latents`: The normal form's trajectories, one per window, each colored by
        `curve_colors(style.curve_colors, q)`.
    - `circle_points`: The normal form's closed limit cycle.
    - `style`: The drawing constants.
    """
    colors = curve_colors(style.curve_colors, len(latents))
    _cycle_and_equilibrium(ax, circle_points, np.zeros(2), style)
    _curves(ax, latents, colors, style, dashed=True, alpha=1.0, halo=True)
    _start_markers(ax, latents[:, 0], colors, style, filled=False)


def draw_prediction(
    ax: Axes,
    data: Float[np.ndarray, "q time 2"],
    predictions: Float[np.ndarray, "q time 2"],
    cycle_points: Float[np.ndarray, "n 2"],
    equilibrium: Float[np.ndarray, " 2"],
    style: SchematicStyle = SchematicStyle(),
) -> None:
    """Draw the prediction panel on `ax`: the limit cycle, the source's equilibrium,
    each window of data solid at `style.data_alpha` without a halo, its prediction
    dashed with a white halo, and a filled circle at the window's start.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `data`: The windows, each colored by `curve_colors(style.curve_colors, q)`.
    - `predictions`: The model's prediction of each window, at the same times.
    - `cycle_points`: The closed limit cycle, such as the result of `cycle_curve`.
    - `equilibrium`: The source's equilibrium, the center of the cycle.
    - `style`: The drawing constants.
    """
    colors = curve_colors(style.curve_colors, len(data))
    _cycle_and_equilibrium(ax, cycle_points, equilibrium, style)
    _curves(ax, data, colors, style, dashed=False, alpha=style.data_alpha, halo=False)
    _curves(ax, predictions, colors, style, dashed=True, alpha=1.0, halo=True)
    _start_markers(ax, data[:, 0], colors, style, filled=True)
