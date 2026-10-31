"""The rubber sheet of a deformation figure: a mesh, its circles, the trajectories and
any extra curves on a box of the plane, mapped through a planar bijection one layer at
a time.

A `SheetConfig` holds the resolution of a sheet, and `Sheet` builds every point set
from a box, the trajectories and a config. `Sheet.map` sends every point through any
map of the plane, so the direction of a bijection is the caller's choice of `model` or
`model.inverse`; it is a tree map over the sheet's array leaves, and the caller jits
the map if it wants. `stage_sheets` maps a sheet through a sequence of layers,
`stage_maps` gives the composed map of each stage, and `stage_labels` names the
stages. `cell_jacobians` evaluates a map's Jacobian at the cell centers, which is what
the metrics of `metrics.py` color the cells by, and `sheet_limits` gives the box that
every stage fits in.

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ...model.invertible import AbstractBijection, SequentialINN


@dataclass(frozen=True)
class SheetConfig:
    """The resolution of a sheet; `copy.replace` changes one field.

    **Attributes:**

    - `cells`: The number of mesh cells along each axis.
    - `subdivisions`: The number of fill sub-quads per cell along each axis, so that a
        curved image cell is filled by a finer polygon than its four corners.
    - `line_samples`: The number of samples of each mesh polyline.
    - `circle_every`: The spacing of the circles in cells: they sit at the centers of
        the cells with indices `1, 1 + circle_every, …` along each axis.
    - `circle_radius`: The radius of each circle, as a fraction of the cell width.
    - `circle_samples`: The number of distinct samples on each circle.
    - `box_pad`: The margin added on each side of the trajectories' bounding box when
        a figure derives the box from them, as a fraction of the box's extent.

    **Raises:**

    `ValueError` naming the field if `cells`, `subdivisions` or `circle_every` is less
    than 1, `line_samples` is less than 2, `circle_samples` is less than 3,
    `circle_radius` is not positive or `box_pad` is negative.
    """

    cells: int = 24
    subdivisions: int = 4
    line_samples: int = 200
    circle_every: int = 4
    circle_radius: float = 0.45
    circle_samples: int = 64
    box_pad: float = 0.1

    def __post_init__(self) -> None:
        """Check every field's range."""
        for name in ("cells", "subdivisions", "circle_every"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}."
                )
        if self.line_samples < 2:
            raise ValueError(
                f"line_samples must be at least 2, got {self.line_samples}."
            )
        if self.circle_samples < 3:
            raise ValueError(
                f"circle_samples must be at least 3, got {self.circle_samples}."
            )
        if self.circle_radius <= 0:
            raise ValueError(
                f"circle_radius must be positive, got {self.circle_radius}."
            )
        if self.box_pad < 0:
            raise ValueError(f"box_pad must not be negative, got {self.box_pad}.")


def _node_grid(
    lo: Float[np.ndarray, " 2"], hi: Float[np.ndarray, " 2"], n: int
) -> Float[np.ndarray, "n n 2"]:
    """The `n × n` grid of nodes spanning `[lo, hi]`, indexed `[iy, ix]`."""
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    return np.stack(np.meshgrid(xs, ys), axis=-1)


class Sheet(eqx.Module):
    """The points of a deformation figure, in the domain or in the image of a map.

    Every array leaf is a set of points of the plane, and the image of a sheet is the
    sheet with every point mapped (`Sheet.map`), so a domain sheet and its image share
    their indexing.

    **Arguments:**

    - `box`: The box `(x_lo, y_lo, x_hi, y_hi)` that the mesh spans.
    - `trajectories`: The trajectories to draw on the sheet, each a polyline of `q`
        samples.
    - `curves`: Extra polylines to draw on the sheet, such as isochrons, each of its
        own length.
    - `config`: The resolution of the sheet.

    **Attributes:**

    - `nodes`: The mesh nodes, `cells + 1` along each axis, indexed `[iy, ix]`.
    - `centers`: The centers of the mesh cells, indexed `[iy, ix]`.
    - `fill_nodes`: The nodes of the sub-quads that the fill is drawn on,
        `cells · subdivisions + 1` along each axis, indexed `[iy, ix]`.
    - `lines`: The `2 (cells + 1)` mesh polylines of `s` samples each: first the lines
        of constant $x$, then the lines of constant $y$.
    - `circles`: Small circles at cell centers, closed polylines whose last sample
        repeats the first.
    - `trajectories`: The trajectories.
    - `curves`: The extra curves.
    - `config`: The resolution that the sheet was built at.

    **Raises:**

    `ValueError` if `box` does not have four entries, or if `x_hi` does not exceed
    `x_lo` or `y_hi` does not exceed `y_lo`.
    """

    nodes: Float[Array, "n n 2"]
    centers: Float[Array, "c c 2"]
    fill_nodes: Float[Array, "m m 2"]
    lines: Float[Array, "l s 2"]
    circles: Float[Array, "k p 2"]
    trajectories: Float[Array, "t q 2"]
    curves: tuple[Float[Array, "_ 2"], ...]
    config: SheetConfig = eqx.field(static=True)

    def __init__(
        self,
        box: Sequence[float],
        trajectories: Float[np.ndarray, "t q 2"] | Float[Array, "t q 2"],
        curves: Sequence[Float[np.ndarray, "_ 2"]] = (),
        config: SheetConfig = SheetConfig(),
    ):
        if len(box) != 4:
            raise ValueError(
                f"box must be (x_lo, y_lo, x_hi, y_hi), got {len(box)} entries."
            )
        x_lo, y_lo, x_hi, y_hi = (float(v) for v in box)
        if x_hi <= x_lo or y_hi <= y_lo:
            raise ValueError(
                f"box must have x_hi > x_lo and y_hi > y_lo, got {tuple(box)}."
            )
        lo, hi = np.array([x_lo, y_lo]), np.array([x_hi, y_hi])
        cells = config.cells
        nodes = _node_grid(lo, hi, cells + 1)
        xs, ys = nodes[0, :, 0], nodes[:, 0, 1]
        sample_xs = np.linspace(x_lo, x_hi, config.line_samples)
        sample_ys = np.linspace(y_lo, y_hi, config.line_samples)
        constant_x = np.stack(
            np.broadcast_arrays(xs[:, None], sample_ys[None, :]), axis=-1
        )
        constant_y = np.stack(
            np.broadcast_arrays(sample_xs[None, :], ys[:, None]), axis=-1
        )
        centers = 0.25 * (
            nodes[:-1, :-1] + nodes[:-1, 1:] + nodes[1:, 1:] + nodes[1:, :-1]
        )
        circle_cells = np.arange(1, cells, config.circle_every)
        circle_centers = centers[np.ix_(circle_cells, circle_cells)].reshape(-1, 2)
        theta = np.linspace(0.0, 2 * np.pi, config.circle_samples, endpoint=False)
        radius = config.circle_radius * (x_hi - x_lo) / cells
        ring = radius * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        ring = np.concatenate([ring, ring[:1]])
        self.nodes = jnp.asarray(nodes)
        self.centers = jnp.asarray(centers)
        self.fill_nodes = jnp.asarray(
            _node_grid(lo, hi, cells * config.subdivisions + 1)
        )
        self.lines = jnp.asarray(np.concatenate([constant_x, constant_y]))
        self.circles = jnp.asarray(circle_centers[:, None, :] + ring[None])
        self.trajectories = jnp.asarray(trajectories, dtype=float)
        self.curves = tuple(jnp.asarray(curve, dtype=float) for curve in curves)
        self.config = config

    def map(self, f: Callable[[Float[Array, " 2"]], Float[Array, " 2"]]) -> "Sheet":
        """The sheet with every point of every array leaf mapped through `f`.

        `f` acts on one point of shape `(2,)`, such as a bijection `model` or
        `model.inverse`. Each leaf, every extra curve on its own, is flattened to its
        points and mapped under `jax.vmap(f)`; nothing is jitted here, so the caller
        passes `eqx.filter_jit(f)` to compile it once.

        **Returns:**

        A `Sheet` with the shapes and the config of this one.
        """
        batched = jax.vmap(f)
        return jax.tree.map(
            lambda points: batched(points.reshape(-1, 2)).reshape(points.shape), self
        )


def stage_sheets(
    layers: Sequence[AbstractBijection], sheet: Sheet, inverse: bool = False
) -> tuple[Sheet, ...]:
    """Compute `sheet` after each layer of `layers` in turn.

    Each stage maps the previous stage's points through one layer, jitted once by
    `eqx.filter_jit`, so the stages cost one pass through the layers in all.

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
    # The bound `__call__`, since `eqx.filter_jit` takes a function's `__name__`, which
    # a module instance does not have.
    steps: list[Callable[[Float[Array, " 2"]], Float[Array, " 2"]]] = (
        [layer.inverse for layer in reversed(layers)]
        if inverse
        else [layer.__call__ for layer in layers]
    )
    sheets = [sheet]
    for step in steps:
        sheets.append(sheets[-1].map(eqx.filter_jit(step)))
    return tuple(sheets)


def stage_maps(
    layers: Sequence[AbstractBijection], inverse: bool = False
) -> tuple[Callable[[Float[Array, " 2"]], Float[Array, " 2"]], ...]:
    """Compose the map from the domain to each stage of `stage_sheets(layers, sheet,
    inverse)`.

    **Arguments:**

    - `layers`: The layers, in the order a forward pass applies them.
    - `inverse`: Whether the stages apply the inverses of the layers, last layer
        first.

    **Returns:**

    `len(layers) + 1` maps of one point. Entry 0 is the identity. Forward, entry `k`
    is the `SequentialINN` of `layers[:k]`; inverted, it is the `inverse` of the
    `SequentialINN` of `layers[-k:]`, which applies the inverses of the last `k` layers,
    last layer first.
    """
    layers = tuple(layers)
    if inverse:
        composed = [
            SequentialINN(layers[-k:]).inverse for k in range(1, len(layers) + 1)
        ]
    else:
        composed = [SequentialINN(layers[:k]) for k in range(1, len(layers) + 1)]
    return (lambda x: x, *composed)


def stage_labels(
    layers: Sequence[AbstractBijection], inverse: bool = False
) -> tuple[str, ...]:
    """Name each stage of `stage_sheets(layers, sheet, inverse)`.

    Stage 0 reads `"stage 0 / n — identity"` and stage `k` reads
    `"stage k / n — <ClassName> (layer k)"`, where `n` is the number of layers and the
    layer is the `k`-th one applied: `layers[k - 1]` forward, `layers[-k]` inverted.
    An inverted layer's class name carries a trailing `⁻¹`, as in
    `"stage 2 / 8 — CouplingFlow⁻¹ (layer 2)"`.

    **Returns:**

    `len(layers) + 1` strings.
    """
    applied = list(reversed(layers)) if inverse else list(layers)
    marker = "⁻¹" if inverse else ""
    n = len(applied)
    return (
        f"stage 0 / {n} — identity",
        *(
            f"stage {k} / {n} — {type(layer).__name__}{marker} (layer {k})"
            for k, layer in enumerate(applied, start=1)
        ),
    )


def cell_jacobians(
    f: Callable[[Float[Array, " 2"]], Float[Array, " 2"]],
    centers: Float[Array, "c c 2"],
) -> Float[Array, "c c 2 2"]:
    """Evaluate the Jacobian of `f` at every cell center, by forward-mode
    differentiation.

    The cells are colored from the map at their centers rather than from the four
    corner images. A straight-sided quad through the corner images cannot represent a
    cell whose image bends inside it, and its signed area goes negative on a trained
    map (the FitzHugh–Nagumo model, 2026-10-31: one cell at $24^2$, two at $48^2$),
    while the Jacobian of a diffeomorphism is defined and non-singular at every point.

    **Arguments:**

    - `f`: The map of one point, such as an entry of `stage_maps`.
    - `centers`: The cell centers, such as `Sheet.centers`.

    **Returns:**

    The Jacobian of `f` at each center, indexed `[iy, ix]`.
    """
    jacobians = jax.vmap(jax.jacfwd(f))(centers.reshape(-1, 2))
    return jacobians.reshape(*centers.shape[:2], 2, 2)


def sheet_limits(
    sheets: Sequence[Sheet], pad: float = 0.05
) -> tuple[float, float, float, float]:
    """Compute the box that every sheet of `sheets` fits in, for shared axis limits.

    **Arguments:**

    - `sheets`: The sheets, such as the stages of `stage_sheets`.
    - `pad`: The margin on each side, as a fraction of the box's extent along that
        axis.

    **Returns:**

    `(x_lo, y_lo, x_hi, y_hi)`: the bounding box of every sheet's `fill_nodes`,
    padded.
    """
    points = np.concatenate([np.asarray(s.fill_nodes).reshape(-1, 2) for s in sheets])
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = pad * (hi - lo)
    (x_lo, y_lo), (x_hi, y_hi) = lo - margin, hi + margin
    return float(x_lo), float(y_lo), float(x_hi), float(y_hi)
