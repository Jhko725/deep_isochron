r"""Integration methods for ``AbstractNormalForm.flow``.

The same normal form can be integrated in different coordinates, and which one is best
near the origin, in cost, or under ``vmap`` is an experimental question. The choice is
therefore an object passed to ``flow(..., integration=...)``, as diffrax passes a solver
instance: each integration is an ``eqx.Module`` with no array leaves, so it is static
under the filter transforms, every integration traces separately, and the dispatch
costs nothing at run time. The names in ``INTEGRATIONS`` stand for the default
instances.

Every integration takes a cartesian ``u0`` and returns a ``diffrax.Solution`` whose
``.ys`` is the cartesian trajectory; the other fields are diffrax's. There are four:
``CartesianIntegration`` solves ``rhs`` on $\mathbb R^2$ and is the only one that can
start at the origin; ``PolarIntegration`` solves ``rhs_polar``;
``RadiusSquaredIntegration`` solves in $s = r^2$ with the even family's data;
``ClosedFormIntegration`` solves no ODE at all and is the reference the other three
are tested against. The design document §6 derives the closed form and compares the
numerical charts.
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
    """`sol` with its ``.ys`` replaced by `ys`."""
    return eqx.tree_at(lambda s: s.ys, sol, ys)


class AbstractFlowIntegration(eqx.Module):
    """One way to integrate a normal form: a callable with no array leaves."""

    @abc.abstractmethod
    def __call__(
        self,
        normal_form: "AbstractNormalForm",
        ts: Float[Array, " time"],
        u0: Float[Array, " 2"],
        args: Any,
        config: SolverConfig,
    ) -> dfx.Solution:
        """The trajectory through cartesian `u0` sampled at `ts`.

        **Arguments:**

        - `normal_form`: The system.
        - `ts`: The save times, shape ``(time,)``.
        - `u0`: The initial cartesian point, shape ``(2,)``.
        - `args`: Passed through to the vector field.
        - `config`: The solver settings.

        **Returns:**

        A ``diffrax.Solution`` with cartesian ``.ys``, shape ``(time, 2)``.
        """


class CartesianIntegration(AbstractFlowIntegration):
    r"""A diffrax solve of ``rhs`` on $\mathbb R^2$. Smooth at the origin, so the
    only integration that can start there."""

    def __call__(self, normal_form, ts, u0, args, config):
        """Solve ``rhs`` from `u0`; see `AbstractFlowIntegration.__call__`."""
        return diffeqsolve(normal_form.rhs, ts, u0, args, config)


class PolarIntegration(AbstractFlowIntegration):
    r"""A diffrax solve of ``rhs_polar`` on $(r, \theta)$, with $\theta$ accumulating
    without wrapping. Singular at $r = 0$, where $\theta$ is undefined."""

    def __call__(self, normal_form, ts, u0, args, config):
        """Solve ``rhs_polar`` from the polar image of `u0` and map the trajectory
        back; see `AbstractFlowIntegration.__call__`."""
        z0 = normal_form.to_polar(u0)
        sol = diffeqsolve(normal_form.rhs_polar, ts, z0, args, config)
        assert sol.ys is not None
        return _with_ys(sol, jax.vmap(normal_form.from_polar)(sol.ys))


class RadiusSquaredIntegration(AbstractFlowIntegration):
    r"""A diffrax solve in $s = r^2$ with the even family's ``*_sq`` data.

    The state is $(s, \theta)$ with $\dot s = 2s\tilde\rho(s)$, a polynomial for the
    Hopf and Bautin forms, and the phase quadrature
    $\dot\theta = \tilde\omega(s)$; $r = \sqrt s$ is recovered on output. This was
    ``BautinNormalForm.solve`` before the integrations became objects. Singular at the
    origin, through $\theta_0$ and the derivative of $\sqrt s$.
    """

    def __call__(self, normal_form, ts, u0, args, config):
        """Solve in $(r^2, \\theta)$ and map back; see
        `AbstractFlowIntegration.__call__`.

        **Raises:**

        `TypeError` for a normal form that is not an ``AbstractEvenNormalForm``.
        """
        from .base import AbstractEvenNormalForm

        if not isinstance(normal_form, AbstractEvenNormalForm):
            raise TypeError(
                "the r_squared integration needs an AbstractEvenNormalForm (data even "
                f"in r); got {type(normal_form).__name__}."
            )

        r0, theta0 = normal_form.to_polar(u0)
        sol = diffeqsolve(
            _rhs_r_squared, ts, jnp.stack((r0**2, theta0)), (normal_form, args), config
        )
        assert sol.ys is not None
        r, theta = jnp.sqrt(sol.ys[:, 0]), sol.ys[:, 1]
        return _with_ys(sol, jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), -1))


def _rhs_r_squared(t, state, args):
    r"""The field in $(s, \theta)$, $s = r^2$: $(2s\tilde\rho(s), \tilde\omega(s))$.

    A module-level function with the normal form in `args`, as diffrax passes
    parameters: a closure built per call was a new function object each time, and
    diffrax's jit of ``diffeqsolve`` compiled again on every call.
    """
    del t
    normal_form, _ = args
    s, _ = state
    return jnp.stack(
        (2 * s * normal_form._log_growth_rate_sq(s), normal_form._angular_rate_sq(s))
    )


class ClosedFormIntegration(AbstractFlowIntegration):
    r"""No ODE solve: $(\Theta, \Psi)$ evolve linearly,
    $\Theta(t) = \Theta_0 + t/T$ and $\Psi(t) = \Psi_0 e^{\kappa t}$, and
    ``from_phase_amplitude`` recovers the point, with one monotone root solve per
    output time (explicit for Hopf). Exact up to the root solve, so the reference for
    the three numerical integrations (design document §6.1). `config` is ignored.

    The phase is in turns and increases with time for either sign of $\omega_1$
    (ADR-0020).
    """

    @eqx.filter_jit
    def __call__(self, normal_form, ts, u0, args, config):
        """Evolve the phase–amplitude coordinates of `u0` and map back; see
        `AbstractFlowIntegration.__call__`. `args` and `config` are ignored.

        Jitted as ``diffeqsolve`` is for the other integrations: run op by op, the
        root solve's ``fori_loop``s compiled their closure on every call."""
        del args, config
        Theta0, Psi0 = normal_form.to_phase_amplitude(u0)
        dt = ts - ts[0]
        Theta = Theta0 + dt / normal_form.period()
        Psi = Psi0 * jnp.exp(normal_form.floquet_exponent() * dt)
        ys = jax.vmap(normal_form.from_phase_amplitude)(jnp.stack((Theta, Psi), -1))
        return dfx.Solution(
            t0=ts[0],
            t1=ts[-1],
            ts=ts,
            ys=ys,
            interpolation=None,
            stats={},
            result=dfx.RESULTS.successful,
            solver_state=None,
            controller_state=None,
            made_jump=None,
            event_mask=None,
        )


INTEGRATIONS: dict[str, AbstractFlowIntegration] = {
    # the default instances, by the names flow() accepts
    "cartesian": CartesianIntegration(),
    "polar": PolarIntegration(),
    "r_squared": RadiusSquaredIntegration(),
    "closed_form": ClosedFormIntegration(),
}


def resolve_integration(
    integration: "AbstractFlowIntegration | str",
) -> AbstractFlowIntegration:
    """The integration object for `integration`, an instance or a name in
    ``INTEGRATIONS``.

    **Raises:**

    `ValueError` for an unknown name.
    """
    if isinstance(integration, AbstractFlowIntegration):
        return integration
    try:
        return INTEGRATIONS[integration]
    except KeyError:
        raise ValueError(
            f"Unknown integration {integration!r}; expected an AbstractFlowIntegration "
            f"or one of {sorted(INTEGRATIONS)}."
        ) from None
