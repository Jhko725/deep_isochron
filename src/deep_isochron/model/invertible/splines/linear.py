"""Monotone piecewise linear spline (C^0), the simplest instance of ``AbstractSpline``."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import AbstractSpline, check_positive, constrain_widths


class LinearSpline(AbstractSpline):
    """C^0 spline: linear interpolation of ``(xs, ys)`` inside the range, identity
    outside. The join with the tails is only continuous. Trainable state is the
    constrained ``x_widths`` and ``y_widths`` (``num_bins`` each)."""

    x_widths: Float[Array, " K"]
    y_widths: Float[Array, " K"]

    num_bins: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        x_widths: Float[Array, " K"],
        y_widths: Float[Array, " K"],
        xy_range: tuple[float, float] = (-1.0, 1.0),
    ):
        if len(x_widths) != len(y_widths):
            raise ValueError("x_widths and y_widths must have the same length.")
        self.x_widths = x_widths
        self.y_widths = y_widths
        self.num_bins = len(x_widths)
        self.xy_range = xy_range

    def __check_init__(self):
        check_positive("x_widths", self.x_widths)
        check_positive("y_widths", self.y_widths)

    @classmethod
    def identity(
        cls, num_bins: int, xy_range: tuple[float, float] = (-1.0, 1.0)
    ) -> "LinearSpline":
        w = jnp.full(num_bins, (xy_range[1] - xy_range[0]) / num_bins)
        return cls(w, w, xy_range)

    @property
    def num_params(self) -> int:
        return 2 * self.num_bins

    def from_unconstrained(
        self,
        params_raw: Float[Array, " {self.num_params}"],
        min_rel_x_bin_width: float = 1e-3,
        min_rel_y_bin_width: float = 1e-3,
        **kwargs,
    ) -> "LinearSpline":
        del kwargs
        x_raw, y_raw = jnp.split(params_raw, 2)
        x_widths = constrain_widths(x_raw, self.range_width, min_rel_x_bin_width)
        y_widths = constrain_widths(y_raw, self.range_width, min_rel_y_bin_width)
        return type(self)(x_widths, y_widths, self.xy_range)

    @property
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.x_widths, *self.xy_range)

    @property
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.y_widths, *self.xy_range)

    def _forward_in_range(self, x):
        k, dx = self.bin_value(x, self.xs)
        return self.ys[k] + dx * self.y_widths[k] / self.x_widths[k]

    def _inverse_in_range(self, y):
        k, dy = self.bin_value(y, self.ys)
        return self.xs[k] + dy * self.x_widths[k] / self.y_widths[k]
