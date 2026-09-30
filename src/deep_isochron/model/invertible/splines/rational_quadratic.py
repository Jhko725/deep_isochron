"""Monotone rational quadratic spline of [1], implemented after [2] with the
identity-tails convention (boundary derivatives fixed to 1) and identity at zero
unconstrained parameters.

[1] C. Durkan et al. Neural Spline Flows. NeurIPS (2019).
[2] https://github.com/bayesiains/nflows/blob/master/nflows/transforms/splines/rational_quadratic.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from deep_isochron.misc import inv_softplus

from .base import AbstractSpline, check_positive, constrain_widths


class MonotonicRQSpline(AbstractSpline):
    """C^1 spline: in bin ``k`` the map is the rational quadratic (Gregory-Delbourgo)
    interpolant through ``(xs[k], ys[k])``, ``(xs[k+1], ys[k+1])`` with derivatives
    ``knot_derivs[k]``, ``knot_derivs[k+1]``. Positive bin widths, heights and
    derivatives make it strictly increasing.

    Trainable state is the constrained ``x_widths``, ``y_widths`` (``num_bins`` each) and
    the interior knot derivatives ``derivs`` (``num_bins - 1``); the boundary
    derivatives are 1 so the identity tails join C^1.
    """

    x_widths: Float[Array, " K"]
    y_widths: Float[Array, " K"]
    derivs: Float[Array, " K-1"]

    num_bins: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        x_widths: Float[Array, " K"],
        y_widths: Float[Array, " K"],
        derivs: Float[Array, " K-1"],
        xy_range: tuple[float, float] = (-1.0, 1.0),
    ):
        if not len(x_widths) == len(y_widths) == len(derivs) + 1:
            raise ValueError(
                "Lengths of x_widths, y_widths, derivs must be num_bins, num_bins, "
                "num_bins-1, respectively."
            )
        self.x_widths = x_widths
        self.y_widths = y_widths
        self.derivs = derivs
        self.num_bins = len(x_widths)
        self.xy_range = xy_range

    def __check_init__(self):
        check_positive("x_widths", self.x_widths)
        check_positive("y_widths", self.y_widths)
        check_positive("derivs", self.derivs)

    @classmethod
    def identity(
        cls, num_bins: int, xy_range: tuple[float, float] = (-1.0, 1.0)
    ) -> "MonotonicRQSpline":
        """Uniform knots, unit derivatives: the identity map (a template for
        ``from_unconstrained``)."""
        w = jnp.full(num_bins, (xy_range[1] - xy_range[0]) / num_bins)
        return cls(w, w, jnp.ones(num_bins - 1), xy_range)

    # -------------------------------------------------------------- interface -----
    @property
    def num_params(self) -> int:
        return 3 * self.num_bins - 1

    def from_unconstrained(
        self,
        params_raw: Float[Array, " {self.num_params}"],
        min_rel_x_bin_width: float = 1e-3,
        min_rel_y_bin_width: float = 1e-3,
        min_derivative: float = 1e-3,
        **kwargs,
    ) -> "MonotonicRQSpline":
        del kwargs
        K = self.num_bins
        x_raw, y_raw, d_raw = jnp.split(params_raw, [K, 2 * K])
        x_widths = constrain_widths(x_raw, self.range_width, min_rel_x_bin_width)
        y_widths = constrain_widths(y_raw, self.range_width, min_rel_y_bin_width)
        # Shifted softplus, so that d_raw = 0 gives a derivative of exactly 1.
        derivs = (
            jax.nn.softplus(d_raw + inv_softplus(1.0 - min_derivative)) + min_derivative
        )
        return type(self)(x_widths, y_widths, derivs, self.xy_range)

    # -------------------------------------------------------------- knot data -----
    @property
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.x_widths, *self.xy_range)

    @property
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.y_widths, *self.xy_range)

    @property
    def knot_derivs(self) -> Float[Array, " {self.num_bins+1}"]:
        """``f'(xs)``, boundaries included (equal to 1)."""
        return jnp.pad(self.derivs, pad_width=1, constant_values=1.0)

    # ------------------------------------------------------------ interpolant -----
    def _forward_in_range(self, x):
        d, s = self.knot_derivs, self.y_widths / self.x_widths
        k, dx = self.bin_value(x, self.xs)
        xi = dx / self.x_widths[k]
        numer = (s[k] * xi**2 + d[k] * xi * (1 - xi)) * self.y_widths[k]
        denom = s[k] + (d[k + 1] + d[k] - 2 * s[k]) * xi * (1 - xi)
        return self.ys[k] + numer / denom

    def _inverse_in_range(self, y):
        d, s = self.knot_derivs, self.y_widths / self.x_widths
        k, dy = self.bin_value(y, self.ys)
        common = dy * (d[k + 1] + d[k] - 2 * s[k])
        a = self.y_widths[k] * (s[k] - d[k]) + common
        b = self.y_widths[k] * d[k] - common
        c = -s[k] * dy
        # Root of a xi^2 + b xi + c = 0 in [0, 1], in the form that is stable for
        # small a (nflows). The discriminant is non-negative for a monotone spline.
        xi = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
        return self.xs[k] + xi * self.x_widths[k]
