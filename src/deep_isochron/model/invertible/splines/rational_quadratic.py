from copy import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from deep_isochron.misc import inv_softplus

from ..base import AbstractScalarBijection


class MonotonicRQSpline(AbstractScalarBijection):
    x_widths: Float[Array, " K-1"]
    y_widths: Float[Array, " K-1"]
    derivs: Float[Array, " K-2"]

    num_knots: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)
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

    def __check_init__(self):
        if jnp.any(self.x_widths <= 0):
            raise ValueError("x_widths must be a positive array.")
        if jnp.any(self.y_widths <= 0):
            raise ValueError("y_widths must be a positive array.")
        if jnp.any(self.derivs <= 0):
            raise ValueError("derivs must be a positive array.")

    def num_params(self) -> int:
        return 3 * self.num_knots - 4

    @property
    def xs(self) -> Float[Array, " {self.num_knots}"]:
        xs = jnp.cumulative_sum(self.x_widths, include_initial=True) + self.xy_range[0]
        return jnp.clip(xs, *self.xy_range)

    @property
    def ys(self) -> Float[Array, " {self.num_knots}"]:
        ys = jnp.cumulative_sum(self.y_widths, include_initial=True) + self.xy_range[0]
        return jnp.clip(ys, *self.xy_range)

    @property
    def ds(self) -> Float[Array, " {self.num_knots}"]:
        return jnp.pad(self.derivs, pad_width=1, constant_values=1)

    def is_inrange(self, x: Float[Array, ""]) -> Bool[Array, ""]:
        lo, hi = self.xy_range
        return (x > lo) & (x < hi)

    def bin_value(
        self, v: Float[Array, ""], bin_edges: Float[Array, " edges"]
    ) -> tuple[Int[Array, ""], Float[Array, ""]]:
        """Given a value and an monotonically increasing array of bin edges, return the
        bin index and the fractional part (value - lower bin_edge).

        Corresponds to the `_interpret_t` function of
        `diffrax.AbstractGlobalInterpolation`."""
        idx = jnp.clip(
            jnp.searchsorted(bin_edges, v, method="compare_all", side="right") - 1,
            0,
            len(bin_edges) - 1,
        )
        return idx, v - bin_edges[idx]

    def __call__(self, x):
        def _in_bounds(x):
            s = self.y_widths / self.x_widths
            k, x_frac = self.bin_value(x, self.xs)
            xi = x_frac / self.x_widths[k]
            numer = (s[k] * xi**2 + self.derivs[k] * xi * (1 - xi)) * self.y_widths[k]
            denom = s[k] + (self.derivs[k + 1] + self.derivs[k] - 2 * s[k]) * xi * (
                1 - xi
            )
            return self.ys[k] + numer / denom

        return jnp.where(self.is_inrange(x), _in_bounds(x), x)

    def inverse(self, y):
        s = self.y_widths / self.x_widths

        def _in_bounds(y):
            k = jnp.clip(
                jnp.searchsorted(self.ys, y, method="compare_all", side="right") - 1,
                0,
                self.num_knots - 1,
            )
            k, Dy = self.bin_value(y, self.ys)
            _common = Dy * (self.derivs[k + 1] + self.derivs[k] - 2 * s[k])

            a = self.y_widths[k] * (s[k] - self.derivs[k]) + _common
            b = self.y_widths[k] * self.derivs[k] - _common
            c = -s[k] * Dy

            xi = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
            return self.xs[k] + xi * self.x_widths[k]

        return jnp.where(self.is_inrange(y), _in_bounds(y), y)

    def from_unconstrained(
        self,
        params_raw,
        min_rel_x_bin_width: float = 1e-3,
        min_rel_y_bin_width: float = 1e-3,
        min_derivative: float = 1e-3,
        **kwargs,
    ):
        del kwargs
        n_bins = self.num_knots - 1
        x_widths, y_widths, derivs = jnp.split(params_raw, [n_bins, 2 * n_bins])

        x_widths = (
            jax.nn.softmax(x_widths) * (1 - n_bins * min_rel_x_bin_width)
            + min_rel_x_bin_width
        ) * (self.xy_range[1] - self.xy_range[0])

        y_widths = (
            jax.nn.softmax(y_widths) * (1 - n_bins * min_rel_y_bin_width)
            + min_rel_y_bin_width
        ) * (self.xy_range[1] - self.xy_range[0])

        derivs = (
            jax.nn.softplus(derivs + (1 - inv_softplus(1 - min_derivative)))
            + min_derivative
        )

        return replace(self, x_widths=x_widths, y_widths=y_widths, derivs=derivs)
