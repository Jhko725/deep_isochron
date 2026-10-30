"""Scales, style and renderers of the deformation figures.

A `FillScale` colors the cells by one of the per-cell values of `plotting.sheet`:
`area_scale` and `anisotropy_scale` build it from the values of one or more stages, so
that a pair and a sequence of stages share one scale. A `SheetStyle` holds the drawing
defaults, and `plot_deformation` draws a domain | image pair.

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import colorcet as cc
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import Affine2D

from .sheet import CellValues, Sheet


@dataclass(frozen=True)
class FillScale:
    """The color scale of one per-cell value, shared by every panel that it fills.

    **Attributes:**

    - `norm`: The range of the logarithmic value that the colormap spans, clipped.
    - `cmap`: The colormap, before `SheetStyle.fill_alpha` blends it over white.
    - `label`: The colorbar's label.
    - `ticks`: The colorbar's tick positions, in the logarithmic value.
    - `tick_labels`: The factor that each tick stands for.
    """

    norm: Normalize
    cmap: Colormap
    label: str
    ticks: tuple[float, ...]
    tick_labels: tuple[str, ...]


def _pooled_nonzero(
    values: Sequence[Float[np.ndarray, "c c"]], percentile: float
) -> Float[np.ndarray, " k"]:
    """The entries of `values` pooled and flattened, exact zeros left out, after
    checking that `percentile` lies in `(50, 100]`."""
    if not 50.0 < percentile <= 100.0:
        raise ValueError(f"percentile must lie in (50, 100], got {percentile}.")
    pool = np.concatenate([np.ravel(v) for v in values])
    return pool[pool != 0.0]


def area_scale(values: Sequence[CellValues], *, percentile: float = 98.0) -> FillScale:
    """The diverging scale of the log area factor, symmetric about 1.

    The range is `[-m, m]`, where `m` is the `percentile`-th percentile of
    `|log_area|` pooled over `values`. Exact zeros are left out of the pool, so that
    an identity stage does not narrow the range; if nothing remains, `m = 1`. The
    ticks are the factors 0.1, 0.3, 1, 3 and 10 that lie within the range.

    **Arguments:**

    - `values`: The cell values of every panel that the scale fills.
    - `percentile`: The percentile of the pooled values at the ends of the range, so
        that a few extreme cells do not wash out the rest.

    **Returns:**

    A `FillScale` on `colorcet.cm.CET_D1A`, labeled `"area factor"`.

    **Raises:**

    `ValueError` if `percentile` is not in `(50, 100]`.
    """
    pool = np.abs(_pooled_nonzero([v.log_area for v in values], percentile))
    m = float(np.percentile(pool, percentile)) if pool.size else 1.0
    factors = [f for f in (0.1, 0.3, 1.0, 3.0, 10.0) if abs(np.log(f)) <= m]
    return FillScale(
        norm=Normalize(-m, m, clip=True),
        cmap=cc.cm.CET_D1A,
        label="area factor",
        ticks=tuple(float(np.log(f)) for f in factors),
        tick_labels=tuple(f"×{f:g}" for f in factors),
    )


def anisotropy_scale(
    values: Sequence[CellValues], *, percentile: float = 98.0
) -> FillScale:
    r"""The sequential scale of the log anisotropy.

    The range is the `(100 − percentile)`-th to the `percentile`-th percentile of
    `log_anisotropy` pooled over `values`. Exact zeros are left out of the pool, so
    that an identity stage does not widen the range; if nothing remains, the range is
    `(0, 1)`. The ticks are the ratios 1, 1.5, 2, 3, 5 and 10 that lie within the
    range.

    **Arguments:**

    - `values`: The cell values of every panel that the scale fills.
    - `percentile`: The percentile of the pooled values at the upper end of the range.

    **Returns:**

    A `FillScale` on `colorcet.cm.CET_L17`, labeled
    $\sigma_{\max} / \sigma_{\min}$.

    **Raises:**

    `ValueError` if `percentile` is not in `(50, 100]`.
    """
    pool = _pooled_nonzero([v.log_anisotropy for v in values], percentile)
    if pool.size:
        lo, hi = (
            float(p) for p in np.percentile(pool, [100.0 - percentile, percentile])
        )
    else:
        lo, hi = 0.0, 1.0
    ratios = [r for r in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0) if lo <= np.log(r) <= hi]
    return FillScale(
        norm=Normalize(lo, hi, clip=True),
        cmap=cc.cm.CET_L17,
        label=r"$\sigma_{\max} / \sigma_{\min}$",
        ticks=tuple(float(np.log(r)) for r in ratios),
        tick_labels=tuple(f"{r:g}" for r in ratios),
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


def _draw_sheet(
    ax: Axes,
    sheet: Sheet,
    values: Float[np.ndarray, "c c"],
    scale: FillScale,
    style: SheetStyle,
) -> QuadMesh:
    """Draw `sheet` on `ax` with its cells filled by `values`; every renderer calls it.

    From the bottom up: the fill, the mesh polylines, the circles, the trajectories
    with a white halo, and a black triangle at the start of each trajectory, pointing
    along its first segment in this panel's coordinates. The fill colors the sub-quads
    of `sheet.fill_nodes`, each cell's value repeated over its sub-quads.

    The fill is drawn opaque, with the colormap blended over white as
    `1 − fill_alpha (1 − rgb)`, rather than with a real alpha: under a real alpha the
    pixels on the edges shared by two sub-quads are painted twice and show as darker
    seams.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `sheet`: The sheet, in the domain or in the image of a map.
    - `values`: The value of each mesh cell, indexed `[iy, ix]`.
    - `scale`: The color scale of `values`.
    - `style`: The drawing constants.

    **Returns:**

    The fill, for a colorbar.

    **Raises:**

    `ValueError` if the number of fill sub-quads along an axis is not a multiple of
    the number of cells.
    """
    cells = sheet.nodes.shape[0] - 1
    subquads = sheet.fill_nodes.shape[0] - 1
    if subquads % cells:
        raise ValueError(
            f"{subquads} fill sub-quads per axis is not a multiple of {cells} cells."
        )
    subdivisions = subquads // cells
    rgb = scale.cmap(np.linspace(0.0, 1.0, 256))[:, :3]
    blended = ListedColormap(1.0 - style.fill_alpha * (1.0 - rgb))
    fine = np.repeat(np.repeat(values, subdivisions, axis=0), subdivisions, axis=1)
    fill = ax.pcolormesh(
        sheet.fill_nodes[..., 0],
        sheet.fill_nodes[..., 1],
        fine,
        shading="flat",
        cmap=blended,
        norm=scale.norm,
        edgecolors="none",
        antialiased=False,
        zorder=0,
    )
    ax.add_collection(
        LineCollection(
            list(sheet.lines),
            colors=style.mesh_color,
            linewidths=style.mesh_width,
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(
            list(sheet.circles),
            colors=style.circle_color,
            linewidths=style.circle_width,
            zorder=2,
        )
    )
    halo = [pe.withStroke(linewidth=style.halo_width, foreground="white")]
    for i, curve in enumerate(sheet.curves):
        ax.plot(
            *curve.T,
            color=style.curve_colors[i % len(style.curve_colors)],
            linewidth=style.curve_width,
            path_effects=halo,
            zorder=3,
        )
        dx, dy = curve[1] - curve[0]
        # "^" points up; rotating it by the segment's angle − 90° points it along.
        turn = Affine2D().rotate_deg(np.degrees(np.arctan2(dy, dx)) - 90.0)
        ax.plot(
            *curve[0],
            marker=MarkerStyle("^").transformed(turn),
            markersize=style.marker_size,
            color="black",
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=5,
        )
    return fill


def plot_deformation(
    reference: Sheet,
    deformed: Sheet,
    values: Float[np.ndarray, "c c"],
    scale: FillScale,
    *,
    style: SheetStyle = SheetStyle(),
    figsize: tuple[float, float] = (9.0, 4.0),
) -> plt.Figure:
    """The domain | image pair of a map: `reference` on the left, `deformed` on the
    right, both filled by `values`.

    The domain panel spans the bounding box of `reference.fill_nodes` and the image
    panel fits `deformed`, both at equal aspect. One colorbar at the right shows
    `scale` with its factor ticks.

    **Arguments:**

    - `reference`: The domain sheet.
    - `deformed`: The image of `reference` under the map, by `Sheet.map`.
    - `values`: The value of each mesh cell, such as `cell_values(reference,
        deformed).log_area`.
    - `scale`: The color scale of `values`, from `area_scale` or `anisotropy_scale`.
    - `style`: The drawing constants.
    - `figsize`: The figure's size in inches.

    **Returns:**

    The figure: its axes are the domain panel, the image panel and the colorbar.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    domain, image = axes
    fill = _draw_sheet(domain, reference, values, scale, style)
    _draw_sheet(image, deformed, values, scale, style)
    points = reference.fill_nodes.reshape(-1, 2)
    lo, hi = points.min(axis=0), points.max(axis=0)
    domain.set_xlim(lo[0], hi[0])
    domain.set_ylim(lo[1], hi[1])
    image.autoscale_view()
    for ax in axes:
        ax.set_aspect("equal")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.grid(False)
    bar = fig.colorbar(
        fill, ax=list(axes), location="right", shrink=0.85, label=scale.label
    )
    bar.set_ticks(scale.ticks, labels=scale.tick_labels)
    bar.ax.yaxis.set_minor_locator(NullLocator())
    return fig
