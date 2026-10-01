"""Monotone rational quadratic spline of [1], implemented after [2] with the
identity-tails convention (boundary derivatives fixed to 1) and identity at zero
unconstrained parameters.

[1] C. Durkan et al. Neural Spline Flows. NeurIPS (2019).
[2] https://github.com/bayesiains/nflows/blob/master/nflows/transforms/splines/rational_quadratic.py
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constraints import Positive, Widths
from .base import AbstractSpline


class RQSplineParams(NamedTuple):
    x_widths: Float[Array, " K"]
    y_widths: Float[Array, " K"]
    derivs: Float[Array, " K-1"]
    """Interior knot derivatives; the boundary derivatives are fixed to 1."""


class MonotonicRQSpline(AbstractSpline[RQSplineParams]):
    """C^1 spline: in bin ``k`` the map is the rational quadratic (Gregory-Delbourgo)
    interpolant through ``(xs[k], ys[k])``, ``(xs[k+1], ys[k+1])`` with derivatives
    ``knot_derivs[k]``, ``knot_derivs[k+1]``. Positive bin widths, heights and
    derivatives make it strictly increasing.

    Raw parameters: ``num_bins`` x-widths, ``num_bins`` y-widths (floored softmax onto
    the range) and ``num_bins - 1`` interior derivatives (``Positive`` with floor
    ``min_derivative`` and ``at_zero = 1``). The boundary derivatives are 1 so the
    identity tails join C^1. ``raw = 0`` is the identity.
    """

    raw: Float[Array, " {self.num_params}"] | None
    num_bins: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)
    min_rel_width: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)
    smoothness: int | None = eqx.field(static=True, default=1, init=False)

    def __init__(
        self,
        num_bins: int,
        xy_range: tuple[float, float] = (-1.0, 1.0),
        *,
        raw: Float[Array, " {self.num_params}"] | None = None,
        min_rel_width: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        if num_bins < 1:
            raise ValueError("An RQ spline needs at least 1 bin.")
        self.num_bins = num_bins
        self.xy_range = self._check_range(xy_range)
        self.min_rel_width = min_rel_width
        self.min_derivative = min_derivative
        self.raw = self._init_raw(jnp.zeros(self.num_params) if raw is None else raw)

    @property
    def num_params(self) -> int:
        return 3 * self.num_bins - 1

    def constrain(self, raw) -> RQSplineParams:
        K = self.num_bins
        x_raw, y_raw, d_raw = jnp.split(raw, [K, 2 * K])
        widths = Widths(self.range_width, self.min_rel_width)
        derivs = Positive(self.min_derivative, at_zero=1.0)(d_raw)
        return RQSplineParams(widths(x_raw), widths(y_raw), derivs)

    # -------------------------------------------------------------- knot data -----
    @property
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.params.x_widths, *self.xy_range)

    @property
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        return self._knots_from_widths(self.params.y_widths, *self.xy_range)

    @property
    def knot_derivs(self) -> Float[Array, " {self.num_bins+1}"]:
        """``f'(xs)``, boundaries included (equal to 1)."""
        return jnp.pad(self.params.derivs, pad_width=1, constant_values=1.0)

    # ------------------------------------------------------------ interpolant -----
    def _forward_in_range(self, x):
        p = self.params
        d, s = self.knot_derivs, p.y_widths / p.x_widths
        k, dx = self.get_bin_and_offset(x, self.xs)
        xi = dx / p.x_widths[k]
        numer = (s[k] * xi**2 + d[k] * xi * (1 - xi)) * p.y_widths[k]
        denom = s[k] + (d[k + 1] + d[k] - 2 * s[k]) * xi * (1 - xi)
        return self.ys[k] + numer / denom

    def _inverse_in_range(self, y):
        p = self.params
        d, s = self.knot_derivs, p.y_widths / p.x_widths
        k, dy = self.get_bin_and_offset(y, self.ys)
        common = dy * (d[k + 1] + d[k] - 2 * s[k])
        a = p.y_widths[k] * (s[k] - d[k]) + common
        b = p.y_widths[k] * d[k] - common
        c = -s[k] * dy
        # Root of a xi^2 + b xi + c = 0 in [0, 1], in the form that is stable for
        # small a (nflows). The discriminant is non-negative for a monotone spline.
        xi = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
        return self.xs[k] + xi * p.x_widths[k]
