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

A concrete spline stores its *constrained* parameters (positive widths, positive
derivatives, ...) as its trainable leaves and implements

* ``xs`` / ``ys``               knot positions and values,
* ``_forward_in_range``        the interpolant, called only with x in [a, b],
* ``_inverse_in_range``        its inverse, called only with y in [a, b],
* ``num_params`` and ``from_unconstrained``   (the ``AbstractScalarBijection``
    interface).

``knot_derivs`` (``f'(xs)``) is provided by the splines that are at least C^1; tests use
it to check the C^1 join.

[1] C. Durkan et al. Neural Spline Flows. NeurIPS (2019).
[2] S. Hong and S. Y. Chun. Neural Diffeomorphic Non-uniform B-spline Flows.
    AAAI (2023).
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from ..base import AbstractScalarBijection
from ..constraints import Widths


def constrain_widths(
    raw: Float[Array, " n"], total: Float[ArrayLike, ""], min_rel_width: float
) -> Float[Array, " n"]:
    """Map ``n`` unconstrained values to ``n`` positive widths summing to ``total``,
    each at least ``min_rel_width * total``. Thin alias of ``constraints.Widths``; to be
    removed when the splines move to the ``raw``/``constrain`` contract."""
    return Widths(total, min_rel_width)(raw)


def check_positive(name: str, values: Float[Array, " n"]) -> None:
    """Raise if any entry is non-positive. Skipped under tracing (``jit``/``vmap``,
    e.g. inside a coupling layer), where ``from_unconstrained`` guarantees
    positivity anyway; the check is for direct construction with concrete arrays."""
    if isinstance(values, jax.core.Tracer):
        return
    if jnp.any(values <= 0):
        raise ValueError(f"{name} must be positive.")


class AbstractSpline(AbstractScalarBijection):
    xy_range: eqx.AbstractVar[tuple[float, float]]
    num_bins: eqx.AbstractVar[int]

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
