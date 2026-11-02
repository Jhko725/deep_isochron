"""The deformation figures: a domain | image pair, a strip of stages and a flipbook of a
planar bijection, each drawn from the bijection's layers and a batch of trajectories.

`plot_deformation`, `plot_deformation_strip` and `animate_deformation` take the layers
and the trajectories and default the rest: the box from the trajectories, the `AREA`
metric, a `SheetConfig` and a `SheetStyle`. Each builds the `Sheet`, maps it through the
stages, evaluates the metric from each stage's composed map at the cell centers and
pools the drawn stages into one `clipped_norm`, so every panel of a figure shares one
colorbar, ticked from the range by a 1–3–10 or 1–2–5 rule. With `inverse=True` each
figure plays the forward stages backward, from the deformed sheet to the regular mesh.
`draw_sheet` draws one sheet on an axes, for a custom figure or a metric defined
outside the package; a `SheetStyle` holds the drawing constants.

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

import math
from collections.abc import Sequence

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D

from ...model.invertible import AbstractBijection
from ..scales import clipped_norm
from ..style import add_colorbar, set_limits
from .metrics import AREA, CellMetric
from .sheet import _stage_sequence, Sheet, sheet_limits, SheetConfig
from .style import _curve_colors, SheetStyle


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
    trajectories = np.asarray(sheet.trajectories)
    for trajectory, color in zip(
        trajectories, _curve_colors(style.curve_colors, len(trajectories)), strict=True
    ):
        ax.plot(
            *trajectory.T,
            color=color,
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
    `config.box_pad` of its extent on each side. The sequence of stages is
    `_stage_sequence(layers, sheet, inverse)`, and the value of stage `k` is the metric
    of its composed map `k` at the cell centers of its stage 0.

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
    sheets, maps, labels = _stage_sequence(layers, sheet, inverse)
    values = tuple(np.asarray(metric.fn(maps[k], sheets[0].centers)) for k in stages)
    norm = clipped_norm(values, style.color_limit_quantile, metric.symmetric)
    return (
        tuple(sheets[k] for k in stages),
        values,
        tuple(labels[k] for k in stages),
        norm,
    )


def _factor_ticks(
    norm: Normalize, symmetric: bool
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    r"""Choose the colorbar ticks of a metric that is the logarithm of a factor or a
    ratio, from the range of `norm`.

    The ticks are the factors of the 1–3–10 series ($10^k$ and $3 \cdot 10^k$ for
    integer $k$) whose logarithms lie in `[norm.vmin, norm.vmax]`. If fewer than three
    qualify, they are those of the 1–2–5 series instead; if fewer than two of those
    qualify, they are the ticks of `MaxNLocator(5)` on the logarithmic axis inside the
    range, labeled to two significant digits. Each tick sits at the logarithm of its
    factor `v` and is labeled `×v` for a symmetric metric and `v` otherwise.

    **Returns:**

    The tick positions, in the units of `norm`, and their labels.

    **Raises:**

    `ValueError` if `norm` lacks an end.
    """
    lo, hi = norm.vmin, norm.vmax
    if lo is None or hi is None:
        raise ValueError(f"norm must have both ends set, got ({lo}, {hi}).")
    decades = range(math.floor(lo / math.log(10)) - 1, math.ceil(hi / math.log(10)) + 2)
    factors: list[float] = []
    digits = "g"
    for mantissas, minimum in (((1.0, 3.0), 3), ((1.0, 2.0, 5.0), 2)):
        factors = [
            m * 10.0**k
            for k in decades
            for m in mantissas
            if lo <= math.log(m * 10.0**k) <= hi
        ]
        if len(factors) >= minimum:
            break
    else:
        located = MaxNLocator(5).tick_values(lo, hi)
        factors = [math.exp(t) for t in located if lo <= t <= hi]
        digits = ".2g"
    prefix = "×" if symmetric else ""
    return (
        tuple(math.log(v) for v in factors),
        tuple(f"{prefix}{v:{digits}}" for v in factors),
    )


def _colorbar(
    fig: plt.Figure,
    axes: Sequence[Axes],
    fill: QuadMesh,
    norm: Normalize,
    metric: CellMetric,
    style: SheetStyle,
) -> None:
    """Add one colorbar of `fill` at the right of `axes`, laid out by the colorbar
    fields of `style`: ticked by `_factor_ticks` for a `log_scale` metric, and by
    matplotlib's default ticks and labels otherwise."""
    ticks, tick_labels = (
        _factor_ticks(norm, metric.symmetric) if metric.log_scale else (None, None)
    )
    add_colorbar(
        fig,
        axes,
        fill,
        label=metric.label,
        ticks=ticks,
        tick_labels=tick_labels,
        shrink=style.colorbar_shrink,
        aspect=style.colorbar_aspect,
        pad=style.colorbar_pad,
        tick_fontsize=style.colorbar_tick_fontsize,
        label_above=style.colorbar_label_above,
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

    Forward, the domain panel spans the box and the image panel the image's
    `sheet_limits`, with axes $x$, $y$ and $u$, $v$. With `inverse`, the pair is the
    forward pair reversed: the domain panel holds the deformed sheet over its
    `sheet_limits`, with axes $u$, $v$, and the image panel the regular mesh over the
    box, with axes $x$, $y$; the fill is the metric of the inverse map at the deformed
    sheet. Both panels are at equal aspect and titled `"domain"` and `"image"`. One
    colorbar at the right shows the metric.

    The inverse pair does not draw the pullback of a regular mesh in the image (base)
    plane. That figure is built by hand: `sheet = Sheet(base_box, trajectories)`, then
    `stage_sheets(layers, sheet, inverse=True)`, each stage drawn by `draw_sheet`.

    **Arguments:**

    - `transforms`: The bijection, or its layers in the order a forward pass applies
        them, such as the `transforms` of a `SequentialINN`.
    - `trajectories`: The trajectories to draw, each a polyline of `q` samples.
    - `metric`: The value that fills each cell.
    - `inverse`: Whether to play the forward stages backward: from the deformed sheet,
        through the inverse layers last layer first, to the regular mesh on the box.
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
    # The regular mesh fills its box exactly; the deformed sheet gets the default pad.
    set_limits(domain, sheet_limits([reference], pad=0.05 if inverse else 0.0))
    set_limits(image, sheet_limits([deformed], pad=0.0 if inverse else 0.05))
    data_axes, image_axes = ("$x$", "$y$"), ("$u$", "$v$")
    if inverse:
        data_axes, image_axes = image_axes, data_axes
    for ax, title, (xlabel, ylabel) in [
        (domain, "domain", data_axes),
        (image, "image", image_axes),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    _colorbar(fig, list(axes), fill, norm, metric, style)
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
    only the leftmost carries the axis labels, $x$, $y$ forward and $u$, $v$ with
    `inverse`, where stage 0 is the deformed sheet. The color range is pooled over the
    drawn stages, and one colorbar at the right shows it.

    **Arguments:**

    - `transforms`: The bijection, or its layers in the order a forward pass applies
        them.
    - `trajectories`: The trajectories to draw.
    - `stages`: The stages to draw, from left to right, each in `0, …, len(layers)`;
        every stage if `None`.
    - `metric`, `inverse`, `box`, `curves`, `config`, `style`: As in
        `plot_deformation`; with `inverse`, stage `k` is forward stage `n − k` and the
        pullback is drawn by hand as there.

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
    panels[0].set_xlabel("$u$" if inverse else "$x$")
    panels[0].set_ylabel("$v$" if inverse else "$y$")
    _colorbar(fig, panels, fills[0], norm, metric, style)
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
    panel, stage 0 first.

    Each frame clears the panel and redraws its stage over the `sheet_limits` of every
    stage, with the stage's label in a white box at the upper left. The colorbar at the
    right of the panel is created once, from a color range pooled over every stage.

    The caller saves the animation with the writer of their choice, such as a GIF by
    `PillowWriter`. An MP4 that a slide deck plays, at the default `interval_ms`, is

        animation.save(
            "flipbook.mp4",
            dpi=160,
            writer=FFMpegWriter(
                fps=1,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-r", "24"],
            ),
        )

    The output rate `-r 24` makes ffmpeg repeat each frame up to 24 frames per second,
    because video players handle a 1 fps stream badly. The `yuv420p` pixel format
    needs `figsize_frame × dpi` to be even in both dimensions: the default
    `style.figsize_frame` at `dpi=160` gives 1280×720. Without an `ffmpeg` on `PATH`,
    the `video` extra (`uv sync --extra video`) installs `imageio-ffmpeg`, and
    `matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()`
    points matplotlib at its binary.

    **Arguments:**

    - `transforms`: The bijection, or its layers in the order a forward pass applies
        them.
    - `trajectories`: The trajectories to draw.
    - `interval_ms`: The time each frame is shown, in milliseconds.
    - `metric`, `inverse`, `box`, `curves`, `config`, `style`: As in
        `plot_deformation`; with `inverse`, the flipbook plays the forward one backward
        and the pullback is drawn by hand as there.

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

    _colorbar(fig, [ax], draw(0), norm, metric, style)
    return FuncAnimation(
        fig,
        lambda k: (draw(k),),
        frames=len(sheets),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
