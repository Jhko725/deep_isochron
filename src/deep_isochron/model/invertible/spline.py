from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .base import AbstractInvertibleTransform


class MonotonicRationalQuadraticSpline(AbstractInvertibleTransform):
    dim: ClassVar[int] = 1  # ty: ignore
    xs: Float[Array, " K+1"]
    ys: Float[Array, " K+1"]
    ds: Float[Array, " K+1"]
    _x_widths: Float[Array, " K"]
    _y_widths: Float[Array, " K"]
    _s: Float[Array, " K"]

    linear_tails: bool = eqx.field(static=True)

    def __init__(self, xs, ys, ds, linear_tails: bool = True):
        self.xs = xs
        self.ys = ys
        self.ds = ds
        self._x_widths = jnp.diff(xs)
        self._y_widths = jnp.diff(ys)
        self._s = self._y_widths / self._x_widths
        self.linear_tails = linear_tails

    def compute_bin_and_normalized_x(
        self, x: Float[Array, ""]
    ) -> tuple[Int[Array, ""], Float[Array, ""]]:
        k = jnp.searchsorted(self.xs, x, method="compare_all") - 1
        xi = (x - self.xs[k]) / (self.xs[k + 1] - self.xs[k])
        return k, xi

    def __call__(self, x):
        condlist = [x < self.xs[0], x >= self.xs[-1]]

        def _x_leq_x_min(x):
            return self.ys[0] + self.ds[0] * (x - self.xs[0])

        def _x_gt_x_max(x):
            return self.ys[-1] + self.ds[-1] * (x - self.xs[-1])

        def _in_bounds(x):
            k, xi = self.compute_bin_and_normalized_x(x)
            numer = (self._s[k] * xi**2 + self.ds[k] * xi * (1 - xi)) * self._y_widths[
                k
            ]
            denom = self._s[k] + (self.ds[k + 1] + self.ds[k] - 2 * self._s[k]) * xi * (
                1 - xi
            )
            return self.ys[k] + numer / denom

        return jnp.piecewise(x, condlist, [_x_leq_x_min, _x_gt_x_max, _in_bounds])

    def inverse(self, y):
        condlist = [y < self.ys[0], y >= self.ys[-1]]

        def _y_leq_y_min(y):
            return self.xs[0] + (y - self.ys[0]) / self.ds[0]

        def _y_gt_y_max(y):
            return self.xs[-1] + (y - self.ys[-1]) / self.ds[-1]

        def _in_bounds(y):
            k = jnp.searchsorted(self.ys, y, method="compare_all") - 1
            Dy = y - self.ys[k]
            _common = Dy * (self.ds[k + 1] + self.ds[k] - 2 * self._s[k])

            a = self._y_widths[k] * (self._s[k] - self.ds[k]) + _common
            b = self._y_widths[k] * self.ds[k] - _common
            c = -self._s[k] * Dy

            xi = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
            return self.xs[k] + xi * self._x_widths[k]

        return jnp.piecewise(y, condlist, [_y_leq_y_min, _y_gt_y_max, _in_bounds])


class MonotonicRQCoupling(AbstractInvertibleTransform):
    mlp: eqx.nn.MLP

    dim: int = eqx.field(static=True)
    split_idx: int = eqx.field(static=True)
    flip: bool = eqx.field(static=True)
    num_knots: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        split_idx: int | None = None,
        width_hidden: int = 10,
        depth: int = 1,
        flip: bool = False,
        num_knots: int = 10,
        xy_range: tuple[float, float] = (-1, 1),
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.dim = dim
        self.split_idx = dim // 2 if split_idx is None else split_idx
        self.flip = flip

        in_size, out_size = (
            self.split_sizes if not self.flip else self.split_sizes[::-1]
        )
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=(3 * num_knots - 2) * out_size,
            width_size=width_hidden,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=key,
        )
        self.num_knots = num_knots
        self.xy_range = xy_range

    @property
    def split_sizes(self) -> tuple[int, int]:
        out_sizes = self.split_idx, self.dim - self.split_idx
        return out_sizes

    def make_spline(
        self, theta: Float[Array, " 3*{self.num_knots}-2"]
    ) -> MonotonicRationalQuadraticSpline:
        xs, ys, ds = jnp.split(theta, [self.num_knots, 2 * self.num_knots])
        xs = (
            jnp.cumsum(jax.nn.softmax(xs)) * (self.xy_range[1] - self.xy_range[0])
            + self.xy_range[0]
        )
        ys = (
            jnp.cumsum(jax.nn.softmax(ys)) * (self.xy_range[1] - self.xy_range[0])
            + self.xy_range[0]
        )
        ds = jnp.pad(
            jax.nn.softplus(ds),
            1,
            mode="constant",
            constant_values=1,
        )
        return MonotonicRationalQuadraticSpline(xs, ys, ds)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        x_up, x_down = jnp.split(x, [self.split_idx])
        y_up = x_up
        theta = self.mlp(x_up).reshape((-1, 3 * self.num_knots - 2))
        spl = eqx.filter_vmap(self.make_spline)(theta)
        y_down = eqx.filter_vmap(lambda spl, x: spl(x))(spl, x_down)
        return jnp.concatenate((y_up, y_down))

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        y_up, y_down = jnp.split(y, [self.split_idx])
        x_up = y_up
        theta = self.mlp(y_up).reshape((-1, 3 * self.num_knots - 2))
        spl = eqx.filter_vmap(self.make_spline)(theta)
        x_down = eqx.filter_vmap(lambda spl, y: spl.inverse(y))(spl, y_down)
        return jnp.concatenate((x_up, x_down))
