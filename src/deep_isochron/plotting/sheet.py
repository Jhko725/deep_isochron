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

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

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
