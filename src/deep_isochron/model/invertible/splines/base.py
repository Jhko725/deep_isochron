r"""Shared structure of the monotone spline bijections.

Every spline here is a strictly increasing map $f:\mathbb{R}\to\mathbb{R}$ that is a
piecewise function on ``xy_range`` $=[a, b]$ and the *identity* outside it. The pieces
are delimited by ``num_bins + 1`` knots ``xs`` (``xs[0] == a``, ``xs[-1] == b``) and
interpolate the values ``ys = f(xs)`` (``ys[0] == a``, ``ys[-1] == b``, so the tails
join continuously).

What differs between splines is the interpolant inside a bin and hence the regularity
of the join with the tails:

* linear                    -> C^0
* rational quadratic [1]    -> C^1 (boundary derivatives pinned to 1)
* cubic B-spline [2]        -> C^2 (boundary B-spline coefficients pinned to Greville)

A concrete spline follows the ``AbstractScalarBijection`` implementation checklist (see
that class's docstring) — one unconstrained leaf ``raw``, constrained parameters
(positive widths, derivatives, ...) computed on read by ``constrain(raw)`` — and, in
place of a bare ``__call__``/``inverse``, implements

* ``xs`` / ``ys``               knot positions and values (from ``self.params``),
* ``_forward_in_range``        the interpolant, called only with x in [a, b],
* ``_inverse_in_range``        its inverse, called only with y in [a, b],
* ``num_params`` (a property of the static configuration) and ``constrain``.

The constructor takes ``num_bins`` and ``xy_range`` (plus floors), and ``raw`` as an
optional keyword; omitted, the spline is the identity.

``knot_derivs`` (``f'(xs)``) is provided by the splines that are at least C^1; tests use
it to check the C^1 join.

[1] C. Durkan et al. Neural Spline Flows. NeurIPS (2019).
[2] S. Hong and S. Y. Chun. Neural Diffeomorphic Non-uniform B-spline Flows.
    AAAI (2023).
"""

import abc
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..base import AbstractScalarBijection


class AbstractSpline[P: tuple](AbstractScalarBijection[P]):
    xy_range: eqx.AbstractVar[tuple[float, float]]
    num_bins: eqx.AbstractVar[int]

    @staticmethod
    def _check_range(xy_range: Sequence[float]) -> tuple[float, float]:
        """``(lo, hi)`` as floats from any two-element sequence (configs give lists)."""
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
        """Knot positions; ``xs[0], xs[-1] == xy_range``."""

    @property
    @abc.abstractmethod
    def ys(self) -> Float[Array, " {self.num_bins+1}"]:
        """Knot values ``f(xs)``; ``ys[0], ys[-1] == xy_range``."""

    @property
    def num_knots(self) -> int:
        return self.num_bins + 1

    @property
    def range_width(self) -> float:
        lo, hi = self.xy_range
        return hi - lo

    # ------------------------------------------------------ interpolant ----------
    @abc.abstractmethod
    def _forward_in_range(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """``f(x)`` for ``x`` in ``xy_range``. Must be finite (and differentiable) on
        the closed range, since it is also evaluated at the clipped tail inputs."""

    @abc.abstractmethod
    def _inverse_in_range(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """``f^{-1}(y)`` for ``y`` in ``xy_range``; same finiteness requirement."""

    # --------------------------------------------------------- bijection ---------
    def is_inrange(self, v: Float[Array, ""]) -> Bool[Array, ""]:
        lo, hi = self.xy_range
        return (v > lo) & (v < hi)

    def clip_to_range(self, v: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.clip(v, *self.xy_range)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        # The interpolant is evaluated at the *clipped* input, so that the branch not
        # selected by `where` is still finite for tail inputs. Otherwise a NaN in the
        # unselected branch leaks into the gradient (jax FAQ, "gradients contain NaN
        # where using where").
        y_in = self._forward_in_range(self.clip_to_range(x))
        return jnp.where(self.is_inrange(x), y_in, x)

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        x_in = self._inverse_in_range(self.clip_to_range(y))
        return jnp.where(self.is_inrange(y), x_in, y)

    # ------------------------------------------------------------- helpers -------
    def get_bin_and_offset(
        self, v: Float[Array, ""], edges: Float[Array, " {self.num_bins+1}"]
    ) -> tuple[Int[Array, ""], Float[Array, ""]]:
        """Bin index of ``v`` and its offset ``v - edges[k]`` from the bin's left edge
        (cf. ``_interpret_t`` in ``diffrax.AbstractGlobalInterpolation``).

        Index ``k`` is computed to satisfy ``edges[k] <= v < edges[k+1]``, clipped to
        the ``n - 1`` bins (so ``v == edges[-1]`` lands in the last bin, out-of-range
        values land in the first or last bin)."""
        k = jnp.searchsorted(edges, v, side="right", method="compare_all") - 1
        k = jnp.clip(k, 0, edges.shape[0] - 2)
        return k, v - edges[k]

    @staticmethod
    def _knots_from_widths(
        widths: Float[Array, " n"], lo: float, hi: float
    ) -> Float[Array, " n+1"]:  # symbolic axis in a return annotation is fine
        """``lo + cumsum(widths)`` with the last knot snapped to ``hi`` exactly, as
        nflows does, so the range endpoint is a knot regardless of rounding."""
        knots = jnp.cumulative_sum(widths, include_initial=True) + lo
        return knots.at[-1].set(hi)
