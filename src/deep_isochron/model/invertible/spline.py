from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .base import AbstractBijection


class MonotonicRQSpline(AbstractBijection):
    dim: ClassVar[int] = 1  # ty: ignore

    _x_widths: Float[Array, " K"]
    _y_widths: Float[Array, " K"]
    _derivs: Float[Array, " K"]

    num_knots: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)
    min_rel_x_bin_width: float = eqx.field(static=True)
    min_rel_y_bin_width: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)
    linear_tails: bool = eqx.field(static=True)

    def __init__(
        self,
        x_widths: Float[Array, " K-1"],
        y_widths: Float[Array, " K-1"],
        derivatives: Float[Array, " K-2"],
        xy_range: tuple[float, float] = (-1, 1),
        min_rel_x_bin_width: float = 1e-3,
        min_rel_y_bin_width: float = 1e-3,
        min_derivative: float = 1e-3,
        linear_tails: bool = True,
    ):
        if not len(x_widths) == len(y_widths) == (len(derivatives) + 1):
            raise ValueError(
                """Lengths of x_widths, y_widths, and derivatives must equal
                num_knots-1, num_knots-1, num_knots-2, respectively."""
            )
        self._x_widths = x_widths
        self._y_widths = y_widths
        self._derivs = derivatives
        self.num_knots = len(x_widths) + 1

        self.xy_range = xy_range
        self.min_rel_x_bin_width = min_rel_x_bin_width
        self.min_rel_y_bin_width = min_rel_y_bin_width
        self.min_derivative = min_derivative
        self.linear_tails = linear_tails

    @property
    def x_widths(self) -> Float[Array, " K-1"]:
        rel_widths = (
            jax.nn.softmax(self._x_widths)
            * (1 - len(self._x_widths) * self.min_rel_x_bin_width)
            + self.min_rel_x_bin_width
        )
        return rel_widths * (self.xy_range[1] - self.xy_range[0])

    @property
    def y_widths(self) -> Float[Array, " K-1"]:
        rel_widths = (
            jax.nn.softmax(self._y_widths)
            * (1 - len(self._y_widths) * self.min_rel_y_bin_width)
            + self.min_rel_y_bin_width
        )
        return rel_widths * (self.xy_range[1] - self.xy_range[0])

    @property
    def xs(self) -> Float[Array, " K"]:
        return jnp.pad(jnp.cumsum(self.x_widths), pad_width=(1, 0)) + self.xy_range[0]

    @property
    def ys(self) -> Float[Array, " K"]:
        return jnp.pad(jnp.cumsum(self.y_widths), pad_width=(1, 0)) + self.xy_range[0]

    @property
    def derivs(self) -> Float[Array, " K"]:
        return jnp.pad(
            jax.nn.softplus(self._derivs) + self.min_derivative,
            1,
            mode="constant",
            constant_values=1,
        )

    def compute_bin_and_normalized_x(
        self, x: Float[Array, ""]
    ) -> tuple[Int[Array, ""], Float[Array, ""]]:
        k = jnp.searchsorted(self.xs, x, method="compare_all") - 1
        xi = (x - self.xs[k]) / (self.x_widths[k])
        return k, xi

    def __call__(self, x):
        condlist = [x <= self.xy_range[0], x >= self.xy_range[1]]
        s = self.y_widths / self.x_widths

        def _x_leq_x_min(x):
            return self.ys[0] + self.derivs[0] * (x - self.xs[0])

        def _x_gt_x_max(x):
            return self.ys[-1] + self.derivs[-1] * (x - self.xs[-1])

        def _in_bounds(x):
            k, xi = self.compute_bin_and_normalized_x(x)
            numer = (s[k] * xi**2 + self.derivs[k] * xi * (1 - xi)) * self.y_widths[k]
            denom = s[k] + (self.derivs[k + 1] + self.derivs[k] - 2 * s[k]) * xi * (
                1 - xi
            )
            return self.ys[k] + numer / denom

        return jnp.piecewise(x, condlist, [_x_leq_x_min, _x_gt_x_max, _in_bounds])

    def inverse(self, y):
        condlist = [y <= self.xy_range[0], y >= self.xy_range[1]]
        s = self.y_widths / self.x_widths

        def _y_leq_y_min(y):
            return self.xs[0] + (y - self.ys[0]) / self.derivs[0]

        def _y_gt_y_max(y):
            return self.xs[-1] + (y - self.ys[-1]) / self.derivs[-1]

        def _in_bounds(y):
            k = jnp.searchsorted(self.ys, y, method="compare_all") - 1
            Dy = y - self.ys[k]
            _common = Dy * (self.derivs[k + 1] + self.derivs[k] - 2 * s[k])

            a = self.y_widths[k] * (s[k] - self.derivs[k]) + _common
            b = self.y_widths[k] * self.derivs[k] - _common
            c = -s[k] * Dy

            xi = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
            return self.xs[k] + xi * self.x_widths[k]

        return jnp.piecewise(y, condlist, [_y_leq_y_min, _y_gt_y_max, _in_bounds])
