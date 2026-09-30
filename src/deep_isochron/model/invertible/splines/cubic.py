from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..base import AbstractScalarBijection


def _cubic(c: Float[Array, " 4"], u):
    return c[0] + u * (c[1] + u * (c[2] + u * c[3]))


def _dcubic(c: Float[Array, " 4"], u):
    return c[1] + u * (2 * c[2] + u * 3 * c[3])


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _monotone_cubic_root(iters: int, c: Float[Array, " 4"], y):
    """Solve c0 + c1 u + c2 u^2 + c3 u^3 = y for u in [0, 1], assuming the cubic is
    strictly increasing there. Newton iteration safeguarded by bisection; the bracket
    [lo, hi] always contains the root, so the result cannot leave the bin."""

    def body(_, state):
        lo, hi, u = state
        r = _cubic(c, u) - y
        lo = jnp.where(r < 0, u, lo)
        hi = jnp.where(r < 0, hi, u)
        u_newton = u - r / _dcubic(c, u)
        ok = (u_newton >= lo) & (
            u_newton <= hi
        )  # inclusive: a converged iterate sits on the bracket
        return lo, hi, jnp.where(ok, u_newton, 0.5 * (lo + hi))

    u0 = jnp.clip((y - c[0]) / (c[1] + c[2] + c[3]), 0.0, 1.0)  # secant guess
    zero, one = jnp.zeros_like(u0), jnp.ones_like(u0)
    _, _, u = jax.lax.fori_loop(0, iters, body, (zero, one, u0))
    return u


@_monotone_cubic_root.defjvp
def _monotone_cubic_root_jvp(iters, primals, tangents):
    # Implicit function theorem on p(u; c) = y:  du = (dy - d_c p . dc) / p'(u).
    # Calls the custom_jvp function itself, so the rule is differentiable to any order.
    c, y = primals
    dc, dy = tangents
    u = _monotone_cubic_root(iters, c, y)
    du = (dy - _cubic(dc, u)) / _dcubic(c, u)
    return u, du


class CubicBSpline(AbstractScalarBijection):
    """Scalar C^2-diffeomorphism of R: a monotone non-uniform cubic B-spline on
    `xy_range`, identity outside it, with C^2 joins.

    Follows Hong & Chun, "Neural Diffeomorphic Non-uniform B-spline Flows" (AAAI 2023)
    for order k = 4, except at the boundary: instead of the rescaling in lines 13-17 of
    their Algorithm 1 (which fixes only f(0)=0, f(1)=1), the three outermost
    coefficients on each side are pinned to their Greville abscissae, which gives
    f = id, f' = 1, f'' = 0 at both ends of the range.

    Index conventions for K = num_bins (paper's r = 0, s = K):
        knots[i]  = t_{i-2},      i = 0..K+4   (t_0 = 0, t_K = 1, normalised coords)
        coeffs[i] = alpha_{i-3},  i = 0..K+2
    """

    _dt: Float[Array, " K+4"]
    _dalpha: Float[Array, " K-2"]

    num_bins: int = eqx.field(static=True)
    xy_range: tuple[float, float] = eqx.field(static=True)
    min_rel_knot_width: float = eqx.field(static=True)
    min_rel_coeff_incr: float = eqx.field(static=True)
    newton_iters: int = eqx.field(static=True)

    def __init__(
        self,
        knot_widths: Float[Array, " K+4"],
        coeff_incrs: Float[Array, " K-2"],
        xy_range: tuple[float, float] = (-1, 1),
        min_rel_knot_width: float = 1e-3,
        min_rel_coeff_incr: float = 1e-3,
        newton_iters: int = 20,
    ):
        num_bins = len(knot_widths) - 4
        if num_bins < 4:
            raise ValueError("Need at least 4 bins (len(knot_widths) >= 8).")
        if len(coeff_incrs) != num_bins - 2:
            raise ValueError(
                "Lengths of knot_widths and coeff_incrs must be num_bins+4 and "
                "num_bins-2, respectively."
            )
        if (
            max(
                min_rel_knot_width * (num_bins + 4), min_rel_coeff_incr * (num_bins - 2)
            )
            >= 1
        ):
            raise ValueError("Minimal relative width/increment too large for num_bins.")
        self._dt = knot_widths
        self._dalpha = coeff_incrs
        self.num_bins = num_bins
        self.xy_range = xy_range
        self.min_rel_knot_width = min_rel_knot_width
        self.min_rel_coeff_incr = min_rel_coeff_incr
        self.newton_iters = newton_iters

    @classmethod
    def identity(cls, num_bins: int, dtype=None, **kwargs) -> "CubicBSpline":
        """Zero raw parameters; this is exactly the identity map."""
        return cls(
            jnp.zeros(num_bins + 4, dtype), jnp.zeros(num_bins - 2, dtype), **kwargs
        )

    # ---- constrained parameters (normalised coordinates, range = [0, 1]) ----

    @property
    def knots(self) -> Float[Array, " K+5"]:
        """t_{-2} < ... < t_{K+2} with t_0 = 0, t_K = 1 (Algorithm 1, lines 1-7)."""
        K, eps = self.num_bins, self.min_rel_knot_width
        w = jax.nn.softmax(self._dt) * (1 - (K + 4) * eps) + eps
        w = w / jnp.sum(w[2 : K + 2])
        return jnp.pad(jnp.cumsum(w), (1, 0)) - (w[0] + w[1])

    @property
    def coeffs(self) -> Float[Array, " K+3"]:
        """Strictly increasing alpha_{-3..K-1}; outer three on each side are Greville
        ."""
        K, eps = self.num_bins, self.min_rel_coeff_incr
        t = self.knots
        greville = (t[:-2] + t[1:-1] + t[2:]) / 3  # (K+3,)
        incr = jax.nn.softmax(self._dalpha) * (1 - (K - 2) * eps) + eps
        span = greville[K] - greville[2]
        interior = greville[2] + span * jnp.cumsum(incr)[:-1]  # alpha_0..alpha_{K-4}
        return jnp.concatenate([greville[:3], interior, greville[K:]])

    @property
    def pieces(self) -> Float[Array, "K 4"]:
        """Power-basis coefficients of each bin's cubic in the local coordinate
        u = (x - t_j) / (t_{j+1} - t_j): Taylor expansion at t_j, from the B-spline
        derivative recursion (de Boor; eq. in proof of Thm. 2 of the paper)."""
        K = self.num_bins
        t, a = self.knots, self.coeffs
        D = (a[1:] - a[:-1]) / (t[3:] - t[:-3])  # D[m] = D_{m-2},  (K+2,)
        E = (D[1:] - D[:-1]) / (t[3:-1] - t[1:-3])  # E[m] = E_{m-1}, (K+1,)
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

    # ---- helpers ----

    def _normalise(self, x):
        lo, hi = self.xy_range
        return (x - lo) / (hi - lo)

    def _denormalise(self, z):
        lo, hi = self.xy_range
        return lo + z * (hi - lo)

    def _bin(self, grid: Float[Array, " K+1"], z) -> Int[Array, ""]:
        k = jnp.searchsorted(grid, z, side="right", method="compare_all") - 1
        return jnp.clip(k, 0, self.num_bins - 1)

    # ---- bijection ----

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        lo, hi = self.xy_range
        inside = (x > lo) & (x < hi)
        z = jnp.clip(self._normalise(x), 0.0, 1.0)
        t = self.knots[2 : self.num_bins + 3]
        k = self._bin(t, z)
        u = (z - t[k]) / (t[k + 1] - t[k])
        y = self._denormalise(_cubic(self.pieces[k], u))
        return jnp.where(inside, y, x)

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        lo, hi = self.xy_range
        inside = (y > lo) & (y < hi)
        z = jnp.clip(self._normalise(y), 0.0, 1.0)
        t = self.knots[2 : self.num_bins + 3]
        pieces = self.pieces
        f_at_knots = jnp.concatenate([pieces[:, 0], jnp.ones((1,), pieces.dtype)])
        k = self._bin(f_at_knots, z)
        u = _monotone_cubic_root(self.newton_iters, pieces[k], z)
        x = self._denormalise(t[k] + u * (t[k + 1] - t[k]))
        return jnp.where(inside, x, y)

    def from_unconstrained(self, params_raw, **kwargs):
        pass
