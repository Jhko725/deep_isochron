r"""ODE systems.

``AbstractODE``
    Any ODE used anywhere in the study — the observed oscillators (FitzHugh–Nagumo,
    Hodgkin–Huxley) and the normal forms alike. It carries the vector field ``rhs`` and
    the numerical flow ``flow``/``flow_result`` (diffrax, with every solver setting in a
    ``SolverConfig``). Anything *numerical* about an ODE — locating its limit cycle,
    monodromy, asymptotic phase by long integration — is a function over ``AbstractODE``
    in ``deep_isochron.analysis``, not a method.

``AbstractNormalForm`` The subset that can serve as the normal-form template of a
conjugacy: planar oscillators with a stable limit cycle whose phase–amplitude structure
is available in
    *closed form*. These expose what the observed system does not — ``period``,
    ``floquet_exponent``, the asymptotic ``phase`` and isostable ``amplitude`` of any
    point, ``limit_cycle``, ``isochron`` — plus the polar chart (``to_chart``,
    ``from_chart``) in which they are naturally written. Their flow can be integrated by
    several strategies (``systems.strategies``), all returning the trajectory in
    cartesian coordinates.

Vmapping: ``flow`` is written for one initial condition; batch with
``eqx.filter_vmap(ode.flow, in_axes=(None, 0))(ts, u0s)``. ``SolverConfig`` and the
strategies have no array leaves, so they are static under the filter transforms.
"""

import abc
from typing import Any

import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, ArrayLike, Float


class SolverConfig(eqx.Module):
    """Every numerical setting of a flow, in one hashable object (no array leaves).

    **Arguments:** as for ``diffrax.diffeqsolve``. ``throw=False`` makes a failed batch
    element report through ``flow_result`` instead of raising for the whole batch (under
    ``vmap``, ``throw=True`` raises if *any* element fails).
    """

    solver: dfx.AbstractSolver = eqx.field(static=True, default=dfx.Tsit5())
    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=4096)
    adjoint: dfx.AbstractAdjoint = eqx.field(
        static=True, default=dfx.RecursiveCheckpointAdjoint()
    )
    throw: bool = eqx.field(static=True, default=True)

    def controller(self) -> dfx.PIDController:
        return dfx.PIDController(rtol=self.rtol, atol=self.atol)


DEFAULT_SOLVER_CONFIG = SolverConfig()


def diffeqsolve(
    rhs, ts: Float[Array, " time"], u0, args, config: SolverConfig
) -> tuple[Array, dfx.RESULTS]:
    """``(ys, result)`` of ``diffrax.diffeqsolve`` of ``rhs`` saved at ``ts``,
    configured by ``config``. ``ys`` is never ``None`` with ``SaveAt(ts=...)``."""
    sol = dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        config.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=u0,
        args=args,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=config.controller(),
        max_steps=config.max_steps,
        adjoint=config.adjoint,
        throw=config.throw,
    )
    assert sol.ys is not None
    return sol.ys, sol.result


class AbstractODE(eqx.Module):
    """An autonomous ODE ``du/dt = rhs(t, u)`` on ``R^dim``."""

    dim: eqx.AbstractVar[int]
    """Implemented as ``eqx.field(static=True, default=..., init=False)`` (ADR-0004)."""

    @abc.abstractmethod
    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " {self.dim}"], args: Any = None
    ) -> Float[Array, " {self.dim}"]:
        """The vector field, in the diffrax signature (``t`` may be a Python float)."""

    def flow_result(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " {self.dim}"],
        args: Any = None,
        *,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> tuple[Float[Array, "time {self.dim}"], dfx.RESULTS]:
        """Trajectory through ``u0`` sampled at ``ts``, and the diffrax result
        (compare with ``diffrax.RESULTS.successful``; elementwise under ``vmap``). Use
        with ``config.throw=False`` to detect failures in a batch instead of raising."""
        return diffeqsolve(self.rhs, ts, u0, args, config)

    def flow(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " {self.dim}"],
        args: Any = None,
        *,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> Float[Array, "time {self.dim}"]:
        """Trajectory through ``u0`` sampled at ``ts``."""
        return self.flow_result(ts, u0, args, config=config)[0]

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{cls}({args})"
