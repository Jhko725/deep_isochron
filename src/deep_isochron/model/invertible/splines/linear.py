"""Monotone piecewise linear spline (C^0). This is the simplest implementation of an
``AbstractSpline``."""

from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constraints import Widths
from .base import AbstractSpline


class LinearSplineParams(NamedTuple):
    x_widths: Float[Array, " K"]
    y_widths: Float[Array, " K"]


class LinearSpline(AbstractSpline[LinearSplineParams]):
    """C^0 spline: linear interpolation of ``(xs, ys)`` inside the range, identity
    outside. The join with the tails is only continuous.

    Raw parameters: ``num_bins`` x-widths then ``num_bins`` y-widths, each block mapped
    by a floored softmax (``Widths``) onto widths summing to the range; ``raw = 0``
    gives uniform knots on both axes, i.e. the identity.
    """

    raw: Float[Array, " {self.num_params}"] | None
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
        self.raw = self._init_raw(jnp.zeros(self.num_params) if raw is None else raw)

    @property
    def num_params(self) -> int:
        return 2 * self.num_bins

    def constrain(self, raw) -> LinearSplineParams:
        x_raw, y_raw = jnp.split(raw, 2)
        widths = Widths(self.range_width, self.min_rel_width)
        return LinearSplineParams(widths(x_raw), widths(y_raw))

    @property
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.params.x_widths, *self.xy_range)

    @property
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.params.y_widths, *self.xy_range)

    def _forward_in_range(self, x):
        p = self.params
        k, dx = self.get_bin_and_offset(x, self.xs)
        return self.ys[k] + dx * p.y_widths[k] / p.x_widths[k]

    def _inverse_in_range(self, y):
        p = self.params
        k, dy = self.get_bin_and_offset(y, self.ys)
        return self.xs[k] + dy * p.x_widths[k] / p.y_widths[k]
