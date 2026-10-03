r"""Integration methods for ``AbstractNormalForm.flow``.

The same normal form can be integrated in different coordinates; which one is best
(accuracy near the origin, cost, behaviour under ``vmap``) is an experimental question,
so the choice is an object passed to ``flow(..., integration=...)`` — the diffrax
pattern of passing a solver instance. Each integration is an ``eqx.Module`` with no
array leaves, hence static under ``eqx.filter_jit``/``filter_vmap``: every integration
traces separately and the dispatch costs nothing at run time. The names in
``INTEGRATIONS`` are accepted as a shorthand for the default instances.

All integrations take cartesian ``u0`` and return a ``diffrax.Solution`` whose ``.ys``
is the **cartesian** trajectory; the other fields (``result``, ``stats``, ``ts``) are
diffrax's. (Dense output or saved solver state, if ever requested through
``SolverConfig``, would be in the integration's own chart.)

* ``CartesianIntegration`` — ``rhs`` on $\mathbb R^2$. Smooth at the origin; the only
  method that can start *at* the origin.
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

from ..base import diffeqsolve, SolverConfig


if TYPE_CHECKING:
    from .base import AbstractNormalForm


def _with_ys(sol: dfx.Solution, ys: Array) -> dfx.Solution:
    return eqx.tree_at(lambda s: s.ys, sol, ys)


class AbstractFlowIntegration(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self,
        normal_form: "AbstractNormalForm",
        ts: Float[Array, " time"],
        u0: Float[Array, " 2"],
        args: Any,
        config: SolverConfig,
    ) -> dfx.Solution:
        """Solution through cartesian ``u0`` at ``ts`` with cartesian ``.ys``."""


class CartesianIntegration(AbstractFlowIntegration):
    def __call__(self, normal_form, ts, u0, args, config):
        return diffeqsolve(normal_form.rhs, ts, u0, args, config)


class PolarIntegration(AbstractFlowIntegration):
    def __call__(self, normal_form, ts, u0, args, config):
        z0 = normal_form.to_chart(u0)
        sol = diffeqsolve(normal_form.rhs_polar, ts, z0, args, config)
        assert sol.ys is not None
        return _with_ys(sol, jax.vmap(normal_form.from_chart)(sol.ys))


class RadiusSquaredIntegration(AbstractFlowIntegration):
    def __call__(self, normal_form, ts, u0, args, config):
        def rhs_s(t, state, args):
            del t, args
            s, _ = state
            return jnp.stack(
                (2 * s * normal_form.radial_rate(s), normal_form.angular_rate(s))
            )

        r0, theta0 = normal_form.to_chart(u0)
        sol = diffeqsolve(rhs_s, ts, jnp.stack((r0**2, theta0)), args, config)
        assert sol.ys is not None
        r, theta = jnp.sqrt(sol.ys[:, 0]), sol.ys[:, 1]
        return _with_ys(sol, jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), -1))


INTEGRATIONS: dict[str, AbstractFlowIntegration] = {
    "cartesian": CartesianIntegration(),
    "polar": PolarIntegration(),
    "r_squared": RadiusSquaredIntegration(),
}


def resolve_integration(
    integration: "AbstractFlowIntegration | str",
) -> AbstractFlowIntegration:
    if isinstance(integration, AbstractFlowIntegration):
        return integration
    try:
        return INTEGRATIONS[integration]
    except KeyError:
        raise ValueError(
            f"Unknown integration {integration!r}; expected an AbstractFlowIntegration "
            f"or one of {sorted(INTEGRATIONS)}."
        ) from None
