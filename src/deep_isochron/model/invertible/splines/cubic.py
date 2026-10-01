"""Monotone non-uniform cubic B-spline of [1] (order k = 4), with identity tails that
join C^2.

[1] S. Hong and S. Y. Chun. Neural Diffeomorphic Non-uniform B-spline Flows.
    AAAI (2023).
"""

from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constraints import Widths
from .base import AbstractSpline


def _cubic(c: Float[Array, " 4"], u: Float[Array, ""]) -> Float[Array, ""]:
    return c[0] + u * (c[1] + u * (c[2] + u * c[3]))


def _dcubic(c: Float[Array, " 4"], u: Float[Array, ""]) -> Float[Array, ""]:
    return c[1] + u * (2 * c[2] + u * 3 * c[3])


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def monotone_cubic_root(
    iters: int, c: Float[Array, " 4"], y: Float[Array, ""]
) -> Float[Array, ""]:
    """Solve ``c0 + c1 u + c2 u^2 + c3 u^3 = y`` for ``u`` in [0, 1], assuming the
    cubic is strictly increasing there.

    Newton iteration safeguarded by bisection: the bracket ``[lo, hi]`` always contains
    the root, so the iterate cannot leave the bin. This replaces the closed-form root
    formula of the reference implementation, whose numerical instability the authors
    of [1] suspect behind their ~1-in-10 training outliers.
    """

    def body(_, state):
        lo, hi, u = state
        r = _cubic(c, u) - y
        lo = jnp.where(r < 0, u, lo)
        hi = jnp.where(r < 0, hi, u)
        u_newton = u - r / _dcubic(c, u)
        # inclusive: a converged iterate sits on the bracket
        ok = (u_newton >= lo) & (u_newton <= hi)
        return lo, hi, jnp.where(ok, u_newton, 0.5 * (lo + hi))

    u0 = jnp.clip((y - c[0]) / (c[1] + c[2] + c[3]), 0.0, 1.0)  # secant guess
    zero, one = jnp.zeros_like(u0), jnp.ones_like(u0)
    _, _, u = jax.lax.fori_loop(0, iters, body, (zero, one, u0))
    return u


@monotone_cubic_root.defjvp
def _monotone_cubic_root_jvp(iters, primals, tangents):
    # Implicit function theorem on p(u; c) = y:  du = (dy - d_c p . dc) / p'(u).
    # Calls the custom_jvp function itself, so the rule is differentiable to any order.
    c, y = primals
    dc, dy = tangents
    u = monotone_cubic_root(iters, c, y)
    du = (dy - _cubic(dc, u)) / _dcubic(c, u)
    return u, du


class BSplineParams(NamedTuple):
    knot_widths: Float[Array, " M"]
    """``num_bins + 4`` positive knot spacings ``t_{i+1} - t_i``; the interior
    ``num_bins`` sum to the range width."""
    coeff_incrs: Float[Array, " M-6"]
    """``num_bins - 2`` positive increments between the non-pinned coefficients."""


class CubicBSpline(AbstractSpline[BSplineParams]):
    r"""C^2 spline: on ``xy_range`` the map is a non-uniform cubic B-spline
    $f(x) = \sum_j \alpha_j B_{j,4}(x)$ with strictly increasing coefficients.

    Knots and coefficients follow Algorithm 1 of [1] (``r = 0``, ``s = num_bins``),
    except at the boundary: instead of the rescaling in its lines 13-17 (which only
    fixes ``f(a) = a``, ``f(b) = b``), the three outermost coefficients on each side are
    pinned to their Greville abscissae ``(t_j + t_{j+1} + t_{j+2}) / 3``. On a
    stretch where the coefficients equal the Greville abscissae a cubic B-spline
    reproduces the identity (Marsden's identity), so ``f = id``, ``f' = 1``, ``f'' = 0``
    at both ends of the range and the identity tails join C^2.

    Raw parameters: ``num_bins + 4`` knot widths then ``num_bins - 2`` coefficient
    increments, each block through a floored softmax (``Widths``). The knot block is
    normalised so that the ``num_bins`` *interior* widths span the range (the four
    exterior widths scale along; this differs from Algorithm 1, which normalises all);
    the coefficient block sums to the Greville span between the pinned ends. ``raw = 0``
    gives uniform knots and Greville coefficients: the identity.

    Index conventions for ``K = num_bins``:
        knots[i]  = t_{i-2},      i = 0..K+4   (t_0 = a, t_K = b)
        coeffs[i] = alpha_{i-3},  i = 0..K+2

    The inverse is a bisection-safeguarded Newton solve with ``newton_iters`` fixed
    iterations (bracket halves at worst, so the inverse is exact to ``2^-newton_iters``
    of a bin in the pathological case and to round-off in practice).
    """

    raw: Float[Array, " {self.num_params}"] | None
    num_bins: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)
    min_rel_knot_width: float = eqx.field(static=True)
    min_rel_coeff_incr: float = eqx.field(static=True)
    newton_iters: int = eqx.field(static=True)
    smoothness: int | None = eqx.field(static=True, default=2, init=False)

    def __init__(
        self,
        num_bins: int,
        xy_range: tuple[float, float] = (-1.0, 1.0),
        *,
        raw: Float[Array, " {self.num_params}"] | None = None,
        min_rel_knot_width: float = 1e-3,
        min_rel_coeff_incr: float = 1e-3,
        newton_iters: int = 20,
    ):
        if num_bins < 4:
            raise ValueError("A cubic B-spline needs at least 4 bins.")
        self.num_bins = num_bins
        self.xy_range = self._check_range(xy_range)
        self.min_rel_knot_width = min_rel_knot_width
        self.min_rel_coeff_incr = min_rel_coeff_incr
        self.newton_iters = newton_iters
        self.raw = self._init_raw(jnp.zeros(self.num_params) if raw is None else raw)

    # -------------------------------------------------------------- interface -----
    @property
    def num_params(self) -> int:
        return 2 * self.num_bins + 2

    def constrain(self, raw) -> BSplineParams:
        K = self.num_bins
        t_raw, a_raw = jnp.split(raw, [K + 4])
        # Algorithm 1, lines 1-3 of [1]: floored softmax, then normalised so that the
        # K interior widths span the range (the 4 exterior ones scale along).
        w = Widths(1.0, self.min_rel_knot_width)(t_raw)
        knot_widths = w / jnp.sum(w[2 : K + 2]) * self.range_width
        greville = self._greville(self._knots(knot_widths))
        span = greville[K] - greville[2]
        coeff_incrs = Widths(span, self.min_rel_coeff_incr)(a_raw)
        return BSplineParams(knot_widths, coeff_incrs)

    # -------------------------------------------------------------- knot data -----
    def _knots(
        self, knot_widths: Float[Array, " {self.num_bins+4}"]
    ) -> Float[Array, " {self.num_bins+5}"]:
        lo, hi = self.xy_range
        t = jnp.cumulative_sum(knot_widths, include_initial=True)
        t = t - t[2] + lo  # t_0 = lo
        return t.at[2 + self.num_bins].set(hi)

    @staticmethod
    def _greville(knots: Float[Array, " M"]) -> Float[Array, " M-2"]:
        return (knots[:-2] + knots[1:-1] + knots[2:]) / 3

    @property
    def knots(self) -> Float[Array, " {self.num_bins+5}"]:
        """Full knot sequence ``t_{-2} < ... < t_{K+2}`` (two knots beyond each end of
        the range, as the cubic pieces at the boundary need them)."""
        return self._knots(self.params.knot_widths)

    @property
    def coeffs(self) -> Float[Array, " {self.num_bins+3}"]:
        """Strictly increasing B-spline coefficients ``alpha_{-3}, ..., alpha_{K-1}``."""
        K = self.num_bins
        greville = self._greville(self.knots)
        interior = greville[2] + jnp.cumsum(self.params.coeff_incrs)[:-1]
        return jnp.concatenate([greville[:3], interior, greville[K:]])

    @property
    def pieces(self) -> Float[Array, "{self.num_bins} 4"]:
        """Power-basis coefficients of each bin's cubic in the local coordinate
        ``u = (x - t_j) / (t_{j+1} - t_j)``: the Taylor expansion at ``t_j``, obtained
        from the B-spline derivative recursion (de Boor; see the proof of Thm. 2 in
        [1]) applied to the three basis functions active on the bin."""
        K = self.num_bins
        t, a = self.knots, self.coeffs
        D = (a[1:] - a[:-1]) / (t[3:] - t[:-3])  # first divided differences
        E = (D[1:] - D[:-1]) / (t[3:-1] - t[1:-3])  # second
        tm2, tm1, t0, t1 = t[0:K], t[1 : K + 1], t[2 : K + 2], t[3 : K + 3]
        h = t1 - t0
        w = (t0 - tm1) / (t1 - tm1)
        c0 = (a[0:K] + (t0 - tm2) * D[0:K]) * (1 - w) + (
            a[1 : K + 1] + (t0 - tm1) * D[1 : K + 1]
        ) * w
        c1 = 3 * h * (D[0:K] * (1 - w) + D[1 : K + 1] * w)
        c2 = 3 * h**2 * E[0:K]
        c3 = h**2 * (E[1 : K + 1] - E[0:K])
        return jnp.stack([c0, c1, c2, c3], axis=-1)

    @property
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        return self.knots[2 : self.num_bins + 3]

    @property
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        return jnp.append(self.pieces[:, 0], self.xy_range[1])

    @property
    def knot_derivs(self) -> Float[Array, " {self.num_bins+1}"]:
        """``f'(xs)``, boundaries included (equal to 1)."""
        return jnp.append(self.pieces[:, 1] / jnp.diff(self.xs), 1.0)

    # ------------------------------------------------------------ interpolant -----
    def _forward_in_range(self, x):
        k, dx = self.get_bin_and_offset(x, self.xs)
        return _cubic(self.pieces[k], dx / jnp.diff(self.xs)[k])

    def _inverse_in_range(self, y):
        k = self.get_bin_and_offset(y, self.ys)[0]
        u = monotone_cubic_root(self.newton_iters, self.pieces[k], y)
        return self.xs[k] + u * jnp.diff(self.xs)[k]
