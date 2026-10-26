"""The monotone piecewise linear spline, $C^0$: the simplest ``AbstractSpline``."""

from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constraints import Widths
from .base import AbstractSpline


class LinearSplineParams(NamedTuple):
    """The constrained parameters of `LinearSpline`: the bin widths on each axis."""

    x_widths: Float[Array, " K"]
    y_widths: Float[Array, " K"]


class LinearSpline(AbstractSpline[LinearSplineParams]):
    """The $C^0$ spline: linear interpolation of ``(xs, ys)`` inside the range, the
    identity outside, with a join that is only continuous.

    The raw parameters are `num_bins` x-widths then `num_bins` y-widths, each block
    mapped by the floored softmax `Widths` onto widths summing to the range;
    ``raw = 0`` gives uniform knots on both axes, the identity.

    **Arguments:**

    - `num_bins`: The number of bins, at least 1.
    - `xy_range`: The range.
    - `raw`: The raw parameters; the identity when omitted.
    - `min_rel_width`: The floor of the relative widths.

    **Raises:** `ValueError` for fewer than 1 bin or a bad range.
    """

    raw: Float[Array, " {self.num_params}"] | None
    num_params: int = eqx.field(static=True, init=False)
    num_bins: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)
    min_rel_width: float = eqx.field(static=True)
    smoothness: int | None = eqx.field(static=True, default=0, init=False)

    def __init__(
        self,
        num_bins: int,
        xy_range: Sequence[float] = (-1.0, 1.0),
        *,
        raw: Float[Array, " {self.num_params}"] | None = None,
        min_rel_width: float = 1e-3,
    ):
        if num_bins < 1:
            raise ValueError("A linear spline needs at least 1 bin.")
        self.num_bins = num_bins
        self.xy_range = self._check_range(xy_range)
        self.min_rel_width = min_rel_width
        self.num_params = 2 * self.num_bins
        self.raw = self._init_raw(jnp.zeros(self.num_params) if raw is None else raw)

    def constrain(self, raw) -> LinearSplineParams:
        """Both blocks through `Widths` over the range."""
        x_raw, y_raw = jnp.split(raw, 2)
        widths = Widths(self.range_width, self.min_rel_width)
        return LinearSplineParams(widths(x_raw), widths(y_raw))

    @property
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        """The knots from the x-widths."""
        return self._knots_from_widths(self.params.x_widths, *self.xy_range)

    @property
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        """The knot values from the y-widths."""
        return self._knots_from_widths(self.params.y_widths, *self.xy_range)

    def _forward_in_range(self, x):
        """Linear interpolation in the bin of `x`."""
        p = self.params
        k, dx = self.get_bin_and_offset(x, self.xs)
        return self.ys[k] + dx * p.y_widths[k] / p.x_widths[k]

    def _inverse_in_range(self, y):
        """Linear interpolation in the bin of `y`, the other way."""
        p = self.params
        k, dy = self.get_bin_and_offset(y, self.ys)
        return self.xs[k] + dy * p.x_widths[k] / p.y_widths[k]
