"""The deformation figures: a domain | image pair, a strip of stages and a flipbook of a
planar bijection, each drawn from the bijection's layers and a batch of trajectories.

`plot_deformation`, `plot_deformation_strip` and `animate_deformation` take the layers
and the trajectories and default the rest: the box from the trajectories, the `AREA`
metric, a `SheetConfig` and a `SheetStyle`. Each builds the `Sheet`, maps it through the
stages, evaluates the metric from each stage's composed map at the cell centers and
pools the drawn stages into one `clipped_norm`, so every panel of a figure shares one
colorbar. `draw_sheet` draws one sheet on an axes, for a custom figure or a metric
defined outside the package; a `SheetStyle` holds the drawing constants.

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import colorcet as cc
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from ...model.invertible import AbstractBijection
from ..scales import clipped_norm
from ..style import add_colorbar, set_limits
from .metrics import AREA, CellMetric
from .sheet import (
    Sheet,
    sheet_limits,
    SheetConfig,
    stage_labels,
    stage_maps,
    stage_sheets,
)


@dataclass(frozen=True)
class SheetStyle:
    """The drawing constants of a deformation figure; `copy.replace` changes one.

    **Attributes:**

    - `fill_alpha`: The weight of the colormap blended over white in the fill.
    - `mesh_color`, `mesh_width`: The mesh polylines.
    - `circle_color`, `circle_width`: The circles.
    - `curve_width`: The trajectories.
    - `halo_width`: The white stroke under each trajectory, wider than `curve_width`.
    - `curve_colors`: The trajectories' colors, in turn, repeating when there are more
        trajectories than colors.
    - `marker_size`: The triangle at the start of each trajectory.
    - `color_limit_quantile`: The quantile of the pooled |values| (or of the values,
        for a metric that is not symmetric) that sets the color limits; values beyond
        it are clipped.
    - `figsize_pair`: The size in inches of the domain | image pair.
    - `figsize_frame`: The size in inches of a flipbook frame.
    - `strip_panel_width`: The width in inches of each panel of a strip.
    - `strip_height`: The height in inches of a strip.
    """

    fill_alpha: float = 0.25
    mesh_color: str = "0.65"
    mesh_width: float = 0.4
    circle_color: str = "black"
    circle_width: float = 0.8
    curve_width: float = 1.6
    halo_width: float = 3.4
    curve_colors: tuple[str, ...] = tuple(cc.glasbey_dark)
    marker_size: float = 7.0
    color_limit_quantile: float = 0.98
    figsize_pair: tuple[float, float] = (9.0, 4.0)
    figsize_frame: tuple[float, float] = (8.0, 4.5)
    strip_panel_width: float = 3.2
    strip_height: float = 3.8


def draw_sheet(
    ax: Axes,
    sheet: Sheet,
    values: Float[np.ndarray, "c c"],
    norm: Normalize,
    metric: CellMetric,
    style: SheetStyle = SheetStyle(),
) -> QuadMesh:
    """Draw `sheet` on `ax` with its cells filled by `values` in the colormap of
    `metric`; every deformation figure calls it.

    From the bottom up: the fill, the mesh polylines, the circles, the extra curves
    (one gray line collection, without a halo), the trajectories with a white halo,
    and a black triangle at the start of each trajectory, pointing along its first
    segment in this panel's coordinates. The fill colors the sub-quads of
    `sheet.fill_nodes`, each cell's value repeated over its sub-quads.

    The fill is drawn opaque, with the colormap blended over white as
    `1 − fill_alpha (1 − rgb)`, rather than with a real alpha: under a real alpha the
    pixels on the edges shared by two sub-quads are painted twice and show as darker
    seams.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `sheet`: The sheet, in the domain or in the image of a map.
    - `values`: The value of each mesh cell, indexed `[iy, ix]`, such as
        `np.asarray(metric.fn(stage_maps(layers)[k], sheet.centers))`.
    - `norm`: The color range of `values`, such as a `clipped_norm`.
    - `metric`: The metric that `values` come from; its `cmap` colors the fill.
    - `style`: The drawing constants.

    **Returns:**

    The fill, for a colorbar.
    """
    subdivisions = sheet.config.subdivisions
    rgb = metric.cmap(np.linspace(0.0, 1.0, 256))[:, :3]
    blended = ListedColormap(1.0 - style.fill_alpha * (1.0 - rgb))
    fine = np.repeat(np.repeat(values, subdivisions, axis=0), subdivisions, axis=1)
    fill_nodes = np.asarray(sheet.fill_nodes)
    fill = ax.pcolormesh(
        fill_nodes[..., 0],
        fill_nodes[..., 1],
        fine,
        shading="flat",
        cmap=blended,
        norm=norm,
        edgecolors="none",
        antialiased=False,
        zorder=0,
    )
    ax.add_collection(
        LineCollection(
            list(np.asarray(sheet.lines)),
            colors=style.mesh_color,
            linewidths=style.mesh_width,
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(
            list(np.asarray(sheet.circles)),
            colors=style.circle_color,
            linewidths=style.circle_width,
            zorder=2,
        )
    )
    if sheet.curves:
        ax.add_collection(
            LineCollection(
                [np.asarray(curve) for curve in sheet.curves],
                colors="0.3",
                linewidths=0.9,
                zorder=2.5,
            )
        )
    halo = [pe.withStroke(linewidth=style.halo_width, foreground="white")]
    for i, trajectory in enumerate(np.asarray(sheet.trajectories)):
        ax.plot(
            *trajectory.T,
            color=style.curve_colors[i % len(style.curve_colors)],
            linewidth=style.curve_width,
            path_effects=halo,
            zorder=3,
        )
        dx, dy = trajectory[1] - trajectory[0]
        # "^" points up; rotating it by the segment's angle − 90° points it along.
        turn = Affine2D().rotate_deg(np.degrees(np.arctan2(dy, dx)) - 90.0)
        ax.plot(
            *trajectory[0],
            marker=MarkerStyle("^").transformed(turn),
            markersize=style.marker_size,
            color="black",
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=5,
        )
    return fill


def _layers(
    transforms: AbstractBijection | Sequence[AbstractBijection],
) -> tuple[AbstractBijection, ...]:
    """The layers of `transforms`: a single bijection is one layer."""
    if isinstance(transforms, AbstractBijection):
        return (transforms,)
    return tuple(transforms)


def _stages(
    layers: tuple[AbstractBijection, ...],
    trajectories: Float[np.ndarray, "t q 2"] | Float[Array, "t q 2"],
    stages: Sequence[int],
    *,
    metric: CellMetric,
    inverse: bool,
    box: Sequence[float] | None,
    curves: Sequence[Float[np.ndarray, "_ 2"]],
    config: SheetConfig,
    style: SheetStyle,
) -> tuple[
    tuple[Sheet, ...], tuple[Float[np.ndarray, "c c"], ...], tuple[str, ...], Normalize
]:
    """Build the sheet, map it through the stages of `layers`, and evaluate the metric
    at each stage in `stages` and the color range pooled over them.

    With `box` `None`, the box is the bounding box of `trajectories` padded by
    `config.box_pad` of its extent on each side. The value of stage `k` is the metric
    of the composed map `stage_maps(layers, inverse)[k]` at the cell centers.

    **Returns:**

    The sheet, the cell values and the `stage_labels` entry of each stage in `stages`,
    and their `clipped_norm` at `style.color_limit_quantile`.

    **Raises:**

    `ValueError` if `stages` is empty or names a stage outside `0, …, len(layers)`.
    """
    outside = [k for k in stages if not 0 <= k <= len(layers)]
    if not stages or outside:
        raise ValueError(
            f"stages must be a non-empty selection of 0, …, {len(layers)}, got "
            f"{tuple(stages)}; out of range: {outside}."
        )
    if box is None:
        points = np.asarray(trajectories).reshape(-1, 2)
        lo, hi = points.min(axis=0), points.max(axis=0)
        margin = config.box_pad * (hi - lo)
        (x_lo, y_lo), (x_hi, y_hi) = lo - margin, hi + margin
        box = (float(x_lo), float(y_lo), float(x_hi), float(y_hi))
    sheet = Sheet(box, trajectories, curves=curves, config=config)
    sheets = stage_sheets(layers, sheet, inverse=inverse)
    maps = stage_maps(layers, inverse=inverse)
    labels = stage_labels(layers, inverse=inverse)
    values = tuple(np.asarray(metric.fn(maps[k], sheet.centers)) for k in stages)
    norm = clipped_norm(values, style.color_limit_quantile, metric.symmetric)
    return (
        tuple(sheets[k] for k in stages),
        values,
        tuple(labels[k] for k in stages),
        norm,
    )


def _colorbar(
    fig: plt.Figure,
    axes: Sequence[Axes],
    fill: QuadMesh,
    norm: Normalize,
    metric: CellMetric,
) -> None:
    """Add one colorbar of `fill` at the right of `axes`, ticked at the logarithms of
    the metric's `tick_values` that lie inside `norm`: labeled `×v` for a symmetric
    metric's factors and `v` otherwise. Raises `ValueError` if `norm` lacks an end."""
    lo, hi = norm.vmin, norm.vmax
    if lo is None or hi is None:
        raise ValueError(f"norm must have both ends set, got ({lo}, {hi}).")
    shown = [v for v in metric.tick_values if lo <= np.log(v) <= hi]
    add_colorbar(
        fig,
        axes,
        fill,
        label=metric.label,
        ticks=[float(np.log(v)) for v in shown],
        tick_labels=[f"×{v:g}" if metric.symmetric else f"{v:g}" for v in shown],
    )


def plot_deformation(
    transforms: AbstractBijection | Sequence[AbstractBijection],
    trajectories: Float[np.ndarray, "t q 2"] | Float[Array, "t q 2"],
    *,
    metric: CellMetric = AREA,
    inverse: bool = False,
    box: Sequence[float] | None = None,
    curves: Sequence[Float[np.ndarray, "_ 2"]] = (),
    config: SheetConfig = SheetConfig(),
    style: SheetStyle = SheetStyle(),
) -> plt.Figure:
    """Draw the domain | image pair of a bijection: the sheet on the left, its image
    under every layer on the right, both filled by the metric of the whole map.

    The domain panel spans the box and the image panel the image's `sheet_limits`, both
    at equal aspect. The panels are titled `"domain"` and `"image"`, with axes $x$, $y$
    and $u$, $v$. One colorbar at the right shows the metric.

    **Arguments:**

    - `transforms`: The bijection, or its layers in the order a forward pass applies
        them, such as the `transforms` of a `SequentialINN`.
    - `trajectories`: The trajectories to draw, each a polyline of `q` samples.
    - `metric`: The value that fills each cell.
    - `inverse`: Whether to draw the inverse map, last layer first.
    - `box`: The box `(x_lo, y_lo, x_hi, y_hi)` of the domain; if `None`, the
        trajectories' bounding box padded by `config.box_pad`.
    - `curves`: Extra polylines to draw, such as isochrons.
    - `config`: The resolution of the sheet.
    - `style`: The drawing constants.

    **Returns:**

    The figure: its axes are the domain panel, the image panel and the colorbar.
    """
    layers = _layers(transforms)
    (reference, deformed), (_, values), _, norm = _stages(
        layers,
        trajectories,
        (0, len(layers)),
        metric=metric,
        inverse=inverse,
        box=box,
        curves=curves,
        config=config,
        style=style,
    )
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_pair, layout="constrained")
    domain, image = axes
    fill = draw_sheet(domain, reference, values, norm, metric, style)
    draw_sheet(image, deformed, values, norm, metric, style)
    set_limits(domain, sheet_limits([reference], pad=0.0))
    set_limits(image, sheet_limits([deformed]))
    for ax, title, (xlabel, ylabel) in [
        (domain, "domain", ("$x$", "$y$")),
        (image, "image", ("$u$", "$v$")),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    _colorbar(fig, list(axes), fill, norm, metric)
    return fig


def plot_deformation_strip(
    transforms: AbstractBijection | Sequence[AbstractBijection],
    trajectories: Float[np.ndarray, "t q 2"] | Float[Array, "t q 2"],
    *,
    stages: Sequence[int] | None = None,
    metric: CellMetric = AREA,
    inverse: bool = False,
    box: Sequence[float] | None = None,
    curves: Sequence[Float[np.ndarray, "_ 2"]] = (),
    config: SheetConfig = SheetConfig(),
    style: SheetStyle = SheetStyle(),
) -> plt.Figure:
    """Draw a row of panels, one per selected stage of a bijection, each filled by the
    metric of the map from the domain to that stage.

    Every panel spans the `sheet_limits` of the drawn stages at equal aspect and is
    titled by its `stage_labels` entry, broken onto two lines at its first `" — "`;
    only the leftmost carries the axis labels $x$, $y$. The color range is pooled over
    the drawn stages, and one colorbar at the right shows it.

    **Arguments:**

    - `transforms`: The bijection, or its layers in the order a forward pass applies
        them.
    - `trajectories`: The trajectories to draw.
    - `stages`: The stages to draw, from left to right, each in `0, …, len(layers)`;
        every stage if `None`.
    - `metric`, `inverse`, `box`, `curves`, `config`, `style`: As in
        `plot_deformation`.

    **Returns:**

    The figure: its axes are the panels from left to right, then the colorbar.

    **Raises:**

    `ValueError` if `stages` is empty or names a stage out of range.
    """
    layers = _layers(transforms)
    sheets, values, labels, norm = _stages(
        layers,
        trajectories,
        tuple(range(len(layers) + 1)) if stages is None else tuple(stages),
        metric=metric,
        inverse=inverse,
        box=box,
        curves=curves,
        config=config,
        style=style,
    )
    count = len(sheets)
    fig, axes = plt.subplots(
        1,
        count,
        figsize=(style.strip_panel_width * count, style.strip_height),
        layout="constrained",
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    panels = list(axes[0])
    limits = sheet_limits(sheets)
    fills = [
        draw_sheet(ax, sheet, value, norm, metric, style)
        for ax, sheet, value in zip(panels, sheets, values, strict=True)
    ]
    for ax, label in zip(panels, labels, strict=True):
        set_limits(ax, limits)
        ax.set_title(label.replace(" — ", "\n", 1), fontsize=10)
    panels[0].set_xlabel("$x$")
    panels[0].set_ylabel("$y$")
    _colorbar(fig, panels, fills[0], norm, metric)
    return fig


def animate_deformation(
    transforms: AbstractBijection | Sequence[AbstractBijection],
    trajectories: Float[np.ndarray, "t q 2"] | Float[Array, "t q 2"],
    *,
    interval_ms: int = 1000,
    metric: CellMetric = AREA,
    inverse: bool = False,
    box: Sequence[float] | None = None,
    curves: Sequence[Float[np.ndarray, "_ 2"]] = (),
    config: SheetConfig = SheetConfig(),
    style: SheetStyle = SheetStyle(),
) -> FuncAnimation:
    """Animate the stages of a bijection as a flipbook: one frame per stage on a single
    panel, the domain first.

    Each frame clears the panel and redraws its stage over the `sheet_limits` of every
    stage, with the stage's label in a white box at the upper left. The colorbar at the
    right of the panel is created once, from a color range pooled over every stage.

    The caller saves the animation with the writer of their choice:
    `FFMpegWriter(fps=1000 // interval_ms, codec="libx264", extra_args=["-pix_fmt",
    "yuv420p"])` for an MP4 that a slide deck plays, or `PillowWriter` for a GIF. The
    `yuv420p` pixel format needs `figsize_frame × dpi` to be even in both dimensions:
    the default `style.figsize_frame` at `dpi=160` gives 1280×720.

    **Arguments:**

    - `transforms`: The bijection, or its layers in the order a forward pass applies
        them.
    - `trajectories`: The trajectories to draw.
    - `interval_ms`: The time each frame is shown, in milliseconds.
    - `metric`, `inverse`, `box`, `curves`, `config`, `style`: As in
        `plot_deformation`.

    **Returns:**

    The animation; its figure stays open until the caller closes it.
    """
    layers = _layers(transforms)
    sheets, values, labels, norm = _stages(
        layers,
        trajectories,
        tuple(range(len(layers) + 1)),
        metric=metric,
        inverse=inverse,
        box=box,
        curves=curves,
        config=config,
        style=style,
    )
    limits = sheet_limits(sheets)
    fig, ax = plt.subplots(figsize=style.figsize_frame, layout="constrained")

    def draw(k: int) -> QuadMesh:
        ax.clear()
        fill = draw_sheet(ax, sheets[k], values[k], norm, metric, style)
        set_limits(ax, limits)
        ax.text(
            0.015,
            0.975,
            labels[k],
            transform=ax.transAxes,
            family="monospace",
            fontsize=10,
            ha="left",
            va="top",
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9
            ),
        )
        return fill

    _colorbar(fig, [ax], draw(0), norm, metric)
    return FuncAnimation(
        fig,
        lambda k: (draw(k),),
        frames=len(sheets),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
