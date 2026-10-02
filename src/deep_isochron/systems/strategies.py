r"""Integration strategies for ``AbstractNormalForm.flow``.

The same normal form can be integrated in different coordinates; which one is best
(accuracy near the origin, cost, behaviour under ``vmap``) is an experimental question,
so the choice is an object passed to ``flow(..., strategy=...)`` — the diffrax pattern
of passing a solver instance. Each strategy is an ``eqx.Module`` with no array leaves,
hence static under ``eqx.filter_jit``/``filter_vmap``: every strategy traces separately
and the dispatch costs nothing at run time. The names in ``STRATEGIES`` are accepted as
a shorthand for the default instances.

All strategies take and return **cartesian** coordinates.

* ``CartesianIntegration`` — ``rhs`` on $\mathbb R^2$. Smooth at the origin; the only
  strategy that can start *at* the origin.
* ``PolarIntegration`` — ``rhs_polar`` on $(r, \theta)$; ``theta`` accumulates without
  wrapping. Singular at $r = 0$ (``theta`` undefined), fine elsewhere.
* ``RadiusSquaredIntegration`` — integrates $s = r^2$ ($\dot s = 2 s\rho(s)$, a
  polynomial for the Hopf/Bautin forms, no square root) together with the phase
  quadrature $\theta(t) = \theta_0 + \int_0^t \omega(s)\,dt$ as a second state;
  recovers $r = \sqrt s$ on output. The former ``BautinNormalForm.solve``. Also
  singular at the origin (through $\theta_0$ and $\sqrt s$'s derivative).
"""

import abc
from typing import Any, TYPE_CHECKING

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import diffeqsolve, SolverConfig


if TYPE_CHECKING:
    from .normal_form import AbstractNormalForm


class AbstractFlowStrategy(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        nf: "AbstractNormalForm",
        ts: Float[Array, " time"],
        u0: Float[Array, " 2"],
        args: Any,
        config: SolverConfig,
    ) -> tuple[Float[Array, "time 2"], dfx.RESULTS]:
        """Cartesian trajectory through cartesian ``u0`` at ``ts``, and the diffrax
        result."""


class CartesianIntegration(AbstractFlowStrategy):
    def __call__(self, nf, ts, u0, args, config):
        return diffeqsolve(nf.rhs, ts, u0, args, config)


class PolarIntegration(AbstractFlowStrategy):
    def __call__(self, nf, ts, u0, args, config):
        zs, result = diffeqsolve(nf.rhs_polar, ts, nf.to_chart(u0), args, config)
        return jax.vmap(nf.from_chart)(zs), result


class RadiusSquaredIntegration(AbstractFlowStrategy):
    def __call__(self, nf, ts, u0, args, config):
        def rhs_s(t, state, args):
            del t, args
            s, _ = state
            return jnp.stack((2 * s * nf.radial_rate(s), nf.angular_rate(s)))

        r0, theta0 = nf.to_chart(u0)
        state0 = jnp.stack((r0**2, theta0))
        ys, result = diffeqsolve(rhs_s, ts, state0, args, config)
        r, theta = jnp.sqrt(ys[:, 0]), ys[:, 1]
        return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), axis=-1), result


STRATEGIES: dict[str, AbstractFlowStrategy] = {
    "cartesian": CartesianIntegration(),
    "polar": PolarIntegration(),
    "r_squared": RadiusSquaredIntegration(),
}


def resolve_strategy(strategy: "AbstractFlowStrategy | str") -> AbstractFlowStrategy:
    if isinstance(strategy, AbstractFlowStrategy):
        return strategy
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown flow strategy {strategy!r}; expected an AbstractFlowStrategy or "
            f"one of {sorted(STRATEGIES)}."
        ) from None
