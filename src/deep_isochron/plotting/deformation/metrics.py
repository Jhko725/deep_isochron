r"""The per-cell metrics that color a deformation figure: what each cell's fill shows.

A `CellMetric` holds a function from a map of one point and the cell centers to one
value per cell, with the colormap, the range convention, the colorbar label and the
ticks that go with it. `AREA` is the log area factor, $\log |\det J|$, and `ANISOTROPY`
the log ratio of the singular values of $J$, both from the Jacobian $J$ of the map at
the cell centers by `sheet.cell_jacobians`. A notebook defines a third metric as one
more value, for instance the phase of a model at the image of each cell center:

    PHASE = CellMetric(
        fn=lambda f, c: jax.vmap(jax.vmap(lambda x: model.phase(f(x))))(c),
        cmap=colorcet.cm.CET_C6, symmetric=False, label="phase", tick_values=(),
    )

The design and its decisions are in `docs/changes/2026-10-30-deformation-plots.md`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import colorcet as cc
import jax.numpy as jnp
from jaxtyping import Array, Float
from matplotlib.colors import Colormap

from .sheet import cell_jacobians


@dataclass(frozen=True)
class CellMetric:
    r"""A value per mesh cell and the color scale that shows it.

    **Attributes:**

    - `fn`: The value of each cell, from the map of one point and the cell centers,
        such as `fn(stage_maps(layers)[k], sheet.centers)`.
    - `cmap`: The colormap.
    - `symmetric`: Whether the color range is symmetric about 0, for the logarithm of
        a factor that is 1 where the map changes nothing; otherwise the range runs from
        a low to a high quantile of the values.
    - `label`: The colorbar's label.
    - `tick_values`: The factors or ratios that the colorbar may tick, at their
        logarithms.
    """

    fn: Callable[
        [Callable[[Float[Array, " 2"]], Float[Array, " 2"]], Float[Array, "c c 2"]],
        Float[Array, "c c"],
    ]
    cmap: Colormap
    symmetric: bool
    label: str
    tick_values: tuple[float, ...]


def _log_area(
    f: Callable[[Float[Array, " 2"]], Float[Array, " 2"]],
    centers: Float[Array, "c c 2"],
) -> Float[Array, "c c"]:
    r"""The log area factor of `f` at each cell center, $\log |\det J|$.

    The absolute value gives an orientation-reversing map, such as a reflection, its
    area factor: a reflection is a diffeomorphism that keeps every area, and its
    $\det J = -1$ has no logarithm."""
    return jnp.log(jnp.abs(jnp.linalg.det(cell_jacobians(f, centers))))


def _log_anisotropy(
    f: Callable[[Float[Array, " 2"]], Float[Array, " 2"]],
    centers: Float[Array, "c c 2"],
) -> Float[Array, "c c"]:
    r"""The log anisotropy of `f` at each cell center,
    $\log(\sigma_{\max} / \sigma_{\min})$ of $J$: 0 where `f` keeps shapes up to
    scale."""
    singular = jnp.linalg.svd(cell_jacobians(f, centers), compute_uv=False)
    return jnp.log(singular[..., 0] / singular[..., -1])


AREA = CellMetric(
    fn=_log_area,
    cmap=cc.cm.CET_D1A,
    symmetric=True,
    label="area factor",
    tick_values=(0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0),
)

ANISOTROPY = CellMetric(
    fn=_log_anisotropy,
    cmap=cc.cm.CET_L17,
    symmetric=False,
    label=r"$\sigma_{\max} / \sigma_{\min}$",
    tick_values=(1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
