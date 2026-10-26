r"""The shared structure of the monotone spline bijections.

Every spline here is a strictly increasing map $f:\mathbb{R}\to\mathbb{R}$ that is a
piecewise function on ``xy_range`` $= [a, b]$ and the identity outside it. The pieces
are delimited by ``num_bins + 1`` knots ``xs``, with ``xs[0] == a`` and
``xs[-1] == b``, and interpolate the values ``ys = f(xs)``, with ``ys[0] == a`` and
``ys[-1] == b``, so the tails join continuously. What differs between splines is the
interpolant inside a bin and hence the regularity of the join with the tails: linear
gives $C^0$; the rational quadratic of Durkan et al. 2019 gives $C^1$, with the
boundary derivatives pinned to 1; the cubic B-spline of Hong & Chun 2023 gives $C^2$,
with the boundary coefficients pinned to the Greville abscissae.

A concrete spline follows the ``AbstractScalarBijection`` implementation checklist
and, in place of a bare ``__call__`` and ``inverse``, implements ``xs`` and ``ys``
from ``self.params``, ``_forward_in_range`` and ``_inverse_in_range``, called only
with their argument in $[a, b]$, and ``num_params`` and ``constrain``. The
constructor takes ``num_bins`` and ``xy_range``, the floors, and ``raw`` as an optional
keyword; omitted, the spline is the identity. ``knot_derivs`` is provided by the
splines that are at least $C^1$, and the tests use it to check the join.
"""

import abc
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..base import AbstractScalarBijection


class AbstractSpline[P: tuple](AbstractScalarBijection[P]):
    """A monotone spline on `xy_range`, the identity outside it.

    **Attributes:**

    - `xy_range`: The range ``(lo, hi)``, static.
    - `num_bins`: The number of bins, static.
    """

    xy_range: eqx.AbstractVar[tuple[float, float]]
    num_bins: eqx.AbstractVar[int]

    @staticmethod
    def _check_range(xy_range: Sequence[float]) -> tuple[float, float]:
        """``(lo, hi)`` as floats from any two-element sequence, since configs give
        lists.

        **Raises:** `ValueError` for another length or ``lo >= hi``.
        """
        if len(xy_range) != 2:
            raise ValueError("xy_range must have two entries, (lo, hi).")
        lo, hi = float(xy_range[0]), float(xy_range[1])
        if not lo < hi:
            raise ValueError("xy_range must satisfy lo < hi.")
        return (lo, hi)

    # ------------------------------------------------------------ knot data -------
    @property
    @abc.abstractmethod
    def xs(self) -> Float[Array, " {self.num_bins+1}"]:
        """The knot positions, shape ``(num_bins + 1,)``; ``xs[0]`` and ``xs[-1]`` are
        the range's ends."""

    @property
    @abc.abstractmethod
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        """The knot values ``f(xs)``, shape ``(num_bins + 1,)``; ``ys[0]`` and
        ``ys[-1]`` are the range's ends."""

    @property
    def num_knots(self) -> int:
        """``num_bins + 1``."""
        return self.num_bins + 1

    @property
    def range_width(self) -> float:
        """``hi - lo``."""
        lo, hi = self.xy_range
        return hi - lo

    # ------------------------------------------------------ interpolant ----------
    @abc.abstractmethod
    def _forward_in_range(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """``f(x)`` for `x` in the range. It must be finite and differentiable on the
        closed range, since it is also evaluated at the clipped tail inputs."""

    @abc.abstractmethod
    def _inverse_in_range(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """``f⁻¹(y)`` for `y` in the range, with the same finiteness requirement."""

    # --------------------------------------------------------- bijection ---------
    def is_inrange(self, v: Float[Array, ""]) -> Bool[Array, ""]:
        """Whether `v` lies strictly inside the range."""
        lo, hi = self.xy_range
        return (v > lo) & (v < hi)

    def clip_to_range(self, v: Float[Array, ""]) -> Float[Array, ""]:
        """`v` clipped to the closed range."""
        return jnp.clip(v, *self.xy_range)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """The interpolant inside the range, `x` itself outside.

        The interpolant is evaluated at the clipped input, so that the branch
        ``where`` does not select is still finite for tail inputs; otherwise a ``nan``
        in the unselected branch would leak into the gradient (the JAX FAQ on
        ``where``).
        """
        y_in = self._forward_in_range(self.clip_to_range(x))
        return jnp.where(self.is_inrange(x), y_in, x)

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """The inverse interpolant inside the range, `y` itself outside."""
        x_in = self._inverse_in_range(self.clip_to_range(y))
        return jnp.where(self.is_inrange(y), x_in, y)

    # ------------------------------------------------------------- helpers -------
    def get_bin_and_offset(
        self, v: Float[Array, ""], edges: Float[Array, " {self.num_bins+1}"]
    ) -> tuple[Int[Array, ""], Float[Array, ""]]:
        """The bin index ``k`` of `v` in `edges`, with ``edges[k] <= v < edges[k+1]``,
        and its offset ``v - edges[k]`` from the bin's left edge, after
        ``_interpret_t`` in ``diffrax.AbstractGlobalInterpolation``.

        ``k`` is clipped to the ``n - 1`` bins, so ``v == edges[-1]`` lands in the
        last bin and out-of-range values land in the first or last bin.
        """
        k = jnp.searchsorted(edges, v, side="right", method="compare_all") - 1
        k = jnp.clip(k, 0, edges.shape[0] - 2)
        return k, v - edges[k]

    @staticmethod
    def _knots_from_widths(
        widths: Float[Array, " n"], lo: float, hi: float
    ) -> Float[Array, " n+1"]:  # symbolic axis in a return annotation is fine
        """The knots ``lo + cumsum(widths)``, with the last snapped to `hi` exactly as
        nflows does, so that the range's endpoint is a knot regardless of rounding."""
        knots = jnp.cumulative_sum(widths, include_initial=True) + lo
        return knots.at[-1].set(hi)
