"""Rubber-sheet geometry of a map of the plane: a mesh, its circles and a batch of
trajectories, mapped through a planar bijection one layer at a time.

A `Sheet` holds the point sets a deformation figure draws, as NumPy arrays: the mesh
nodes, the finer nodes of the fill, the mesh polylines, small circles and the
trajectories. `build_sheet` lays one out on a box, and `Sheet.map` sends every point
through any map of the plane in one batch, so the direction of a bijection is the
caller's choice of `model` or `model.inverse`. `cell_values` compares a reference sheet
with its image cell by cell, as a log area factor and a log anisotropy. `stage_sheets`
maps a sheet through a sequence of layers, `stage_labels` names its stages, and
`sheet_limits` gives the box that every stage fits in.

A `FillScale` colors the cells by one of those values: `area_scale` and
`anisotropy_scale` build it from the values of one or more stages, so that a pair and
a sequence of stages share one scale. A `SheetStyle` holds the drawing defaults, and
`plot_deformation` draws a domain | image pair.

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import colorcet as cc
import jax
import jax.numpy as jnp
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import Affine2D

from ..model.invertible import AbstractBijection


@dataclass(frozen=True)
class Sheet:
    """The points of a deformation figure, in the domain or in the image of a map.

    Every field is a set of points of the plane, and the image of a sheet is the sheet
    with every point mapped (`Sheet.map`), so a domain sheet and its image share their
    indexing.

    **Attributes:**

    - `nodes`: The mesh nodes, `cells + 1` along each axis, indexed `[iy, ix]`.
    - `fill_nodes`: The nodes of the sub-quads that the fill is drawn on,
        `cells · subdivisions + 1` along each axis, indexed `[iy, ix]`.
    - `lines`: The `2 (cells + 1)` mesh polylines of `s` samples each: first the lines
        of constant $x$, then the lines of constant $y$.
    - `circles`: Small circles at cell centers, closed polylines whose last sample
        repeats the first.
    - `curves`: The trajectories, each a polyline of `p` samples.
    """

    nodes: Float[np.ndarray, "n n 2"]
    fill_nodes: Float[np.ndarray, "m m 2"]
    lines: Float[np.ndarray, "l s 2"]
    circles: Float[np.ndarray, "k c 2"]
    curves: Float[np.ndarray, "t p 2"]

    def map(self, f: Callable[[Float[Array, " 2"]], Float[Array, " 2"]]) -> "Sheet":
        """The sheet with every point of every field mapped through `f`.

        `f` acts on one point of shape `(2,)`, such as a bijection `model` or
        `model.inverse`; every point is mapped in one flat batch under
        `jax.jit(jax.vmap(f))`.

        **Returns:**

        A `Sheet` of NumPy arrays with the shapes of this one.
        """
        arrays = [getattr(self, field.name) for field in fields(self)]
        flat = np.concatenate([a.reshape(-1, 2) for a in arrays])
        mapped = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(flat)))
        parts = np.split(mapped, np.cumsum([a.size // 2 for a in arrays])[:-1])
        return Sheet(
            **{
                field.name: part.reshape(a.shape)
                for field, part, a in zip(fields(self), parts, arrays)
            }
        )


def _node_grid(
    lo: Float[np.ndarray, "2"], hi: Float[np.ndarray, "2"], n: int
) -> Float[np.ndarray, "n n 2"]:
    """The `n × n` grid of nodes spanning `[lo, hi]`, indexed `[iy, ix]`."""
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    return np.stack(np.meshgrid(xs, ys), axis=-1)


def _cell_corners(nodes: Float[np.ndarray, "n n 2"]) -> Float[np.ndarray, "c c 4 2"]:
    """The four corners of each cell of `nodes`, counterclockwise from the lower
    left."""
    return np.stack(
        [nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:], nodes[1:, :-1]], axis=-2
    )


def _shoelace(quads: Float[np.ndarray, "*batch 4 2"]) -> Float[np.ndarray, " *batch"]:
    """The signed area of each quad by the shoelace formula, positive for
    counterclockwise corners."""
    x, y = quads[..., 0], quads[..., 1]
    return 0.5 * np.sum(
        x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1
    )


def build_sheet(
    lo: Float[np.ndarray, "2"],
    hi: Float[np.ndarray, "2"],
    curves: Float[np.ndarray, "t p 2"],
    *,
    cells: int = 24,
    line_samples: int = 200,
    circle_every: int = 4,
    circle_radius: float = 0.12,
    circle_samples: int = 64,
    subdivisions: int = 4,
) -> Sheet:
    """The domain sheet of the box `[lo, hi]` with the trajectories `curves`.

    **Arguments:**

    - `lo`, `hi`: The lower left and upper right corners of the box.
    - `curves`: The trajectories to draw on the sheet.
    - `cells`: The number of mesh cells along each axis.
    - `line_samples`: The number of samples of each mesh polyline.
    - `circle_every`: The spacing of the circles in cells: they sit at the centers of
        the cells with indices `1, 1 + circle_every, …` along each axis.
    - `circle_radius`: The radius of each circle, in the units of the box.
    - `circle_samples`: The number of distinct samples on each circle.
    - `subdivisions`: The number of fill sub-quads per cell along each axis, so that a
        curved image cell is filled by a finer polygon than its four corners.

    **Returns:**

    A `Sheet` of NumPy arrays.

    **Raises:**

    `ValueError` if `cells` or `subdivisions` is less than 1, or if `hi` does not
    exceed `lo` in both coordinates.
    """
    if cells < 1:
        raise ValueError(f"cells must be at least 1, got {cells}.")
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be at least 1, got {subdivisions}.")
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if np.any(hi <= lo):
        raise ValueError(
            f"hi must exceed lo in both coordinates, got lo={lo} and hi={hi}."
        )
    nodes = _node_grid(lo, hi, cells + 1)
    xs, ys = nodes[0, :, 0], nodes[:, 0, 1]
    sample_xs = np.linspace(lo[0], hi[0], line_samples)
    sample_ys = np.linspace(lo[1], hi[1], line_samples)
    constant_x = np.stack(np.broadcast_arrays(xs[:, None], sample_ys[None, :]), axis=-1)
    constant_y = np.stack(np.broadcast_arrays(sample_xs[None, :], ys[:, None]), axis=-1)
    circle_cells = np.arange(1, cells, circle_every)
    centers = _cell_corners(nodes).mean(axis=-2)[np.ix_(circle_cells, circle_cells)]
    theta = np.linspace(0.0, 2 * np.pi, circle_samples, endpoint=False)
    ring = circle_radius * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    ring = np.concatenate([ring, ring[:1]])
    return Sheet(
        nodes=nodes,
        fill_nodes=_node_grid(lo, hi, cells * subdivisions + 1),
        lines=np.concatenate([constant_x, constant_y]),
        circles=centers.reshape(-1, 2)[:, None, :] + ring[None],
        curves=np.asarray(curves, dtype=float),
    )


@dataclass(frozen=True)
class CellValues:
    r"""The deformation of each mesh cell from a reference sheet to a deformed one.

    **Attributes:**

    - `log_area`: The logarithm of the cell's area factor, deformed over reference.
    - `log_anisotropy`: The logarithm of $\sigma_{\max} / \sigma_{\min}$, the
        ratio of the singular values of the affine map that best fits the cell's
        corners: 0 for a cell that keeps its shape up to scale.
    """

    log_area: Float[np.ndarray, "c c"]
    log_anisotropy: Float[np.ndarray, "c c"]


def cell_values(reference: Sheet, deformed: Sheet) -> CellValues:
    r"""The log area factor and the log anisotropy of each mesh cell, from `reference`
    to `deformed`.

    Both values come from the four corners of each cell in the `nodes` of the two
    sheets. The area factor is the ratio of the shoelace areas, deformed over
    reference. The anisotropy is the ratio of the singular values of the least-squares
    affine fit $q - \bar q = F (p - \bar p)$ to the four corner pairs, $p$ in
    `reference` and $q$ in `deformed`. Both are exact for an affine map.

    The values are computed from the corners rather than from the Jacobian at the
    cell's center, because the cell is what the figure draws: a map that bends within
    one cell makes the two differ. On the prototype scene of this module's design (an
    eight-layer INN at random initialization, seed 7, on a $24 \times 24$ mesh) they
    differed by up to 0.65 in log area and 0.50 in log anisotropy, measured. The corner
    values converge to the Jacobian's as the mesh is refined.

    **Arguments:**

    - `reference`: The sheet that the values are relative to, usually the domain.
    - `deformed`: The image of `reference` under a map, by `Sheet.map`.

    **Returns:**

    The `CellValues` of the mesh, one entry per cell, indexed `[iy, ix]`.

    **Raises:**

    `ValueError` naming the number of cells whose area factor is not positive: a
    folded or inverted cell has no logarithm.
    """
    p, q = _cell_corners(reference.nodes), _cell_corners(deformed.nodes)
    area = _shoelace(q) / _shoelace(p)
    folded = int(np.sum(area <= 0))
    if folded:
        raise ValueError(f"{folded} cells have a non-positive area factor.")
    dp = p - p.mean(axis=-2, keepdims=True)
    dq = q - q.mean(axis=-2, keepdims=True)
    # Each row of dp maps to the same row of dq: dp F^T = dq, so the least-squares fit
    # is F^T = (dp^T dp)^{-1} dp^T dq.
    dp_t = np.swapaxes(dp, -1, -2)
    fit = np.linalg.solve(dp_t @ dp, dp_t @ dq)
    singular = np.linalg.svd(fit, compute_uv=False)
    anisotropy = singular[..., 0] / singular[..., 1]
    return CellValues(log_area=np.log(area), log_anisotropy=np.log(anisotropy))


def stage_sheets(
    layers: Sequence[AbstractBijection], sheet: Sheet, inverse: bool = False
) -> tuple[Sheet, ...]:
    """`sheet` after each layer of `layers` in turn.

    Each stage maps the previous stage's points through one layer, so the stages cost
    one pass through the layers in all.

    **Arguments:**

    - `layers`: The layers, in the order a forward pass applies them, as in the
        `transforms` of a `SequentialINN`.
    - `sheet`: The sheet of stage 0.
    - `inverse`: Whether to apply the inverses of the layers, last layer first.

    **Returns:**

    `len(layers) + 1` sheets. Entry 0 is `sheet`. Forward, entry `k` is `sheet` after
    `layers[0], …, layers[k - 1]`; inverted, after `layers[-1].inverse, …,
    layers[-k].inverse`. The last entry is `sheet.map(model)` or
    `sheet.map(model.inverse)` for the `SequentialINN` of `layers`.
    """
    steps = [layer.inverse for layer in reversed(layers)] if inverse else list(layers)
    sheets = [sheet]
    for step in steps:
        sheets.append(sheets[-1].map(step))
    return tuple(sheets)


def stage_labels(
    layers: Sequence[AbstractBijection], inverse: bool = False
) -> tuple[str, ...]:
    """The label of each stage of `stage_sheets(layers, sheet, inverse)`.

    Stage 0 reads `"stage 0 / n — identity"` and stage `k` reads
    `"stage k / n — <ClassName> (layer k)"`, where `n` is the number of layers and the
    layer is the `k`-th one applied: `layers[k - 1]` forward, `layers[-k]` inverted.

    **Returns:**

    `len(layers) + 1` strings.
    """
    applied = list(reversed(layers)) if inverse else list(layers)
    n = len(applied)
    return (
        f"stage 0 / {n} — identity",
        *(
            f"stage {k} / {n} — {type(layer).__name__} (layer {k})"
            for k, layer in enumerate(applied, start=1)
        ),
    )


def sheet_limits(
    sheets: Sequence[Sheet], pad: float = 0.05
) -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "2"]]:
    """The box that every sheet of `sheets` fits in, for shared axis limits.

    **Arguments:**

    - `sheets`: The sheets, such as the stages of `stage_sheets`.
    - `pad`: The margin on each side, as a fraction of the box's extent along that
        axis.

    **Returns:**

    `(lo, hi)`: the lower left and upper right corners of the bounding box of every
    sheet's `fill_nodes`, padded.
    """
    points = np.concatenate([s.fill_nodes.reshape(-1, 2) for s in sheets])
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


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
