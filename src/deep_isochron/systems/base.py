r"""ODE systems.

``AbstractODE``
    Any ODE used anywhere in the study — the observed oscillators (FitzHugh–Nagumo,
    Hodgkin–Huxley) and the normal forms alike. It carries the vector field ``rhs``, the
    constrained parameter values ``params()`` (for provenance), and the numerical flow
    ``flow`` (diffrax, every solver setting in a ``SolverConfig``). Anything *numerical*
    about an ODE beyond integrating it — locating its limit cycle, monodromy, asymptotic
    phase by long integration — is a function over ``AbstractODE`` in
    ``deep_isochron.analysis`` (Phase E), not a method.

``AbstractNormalForm`` (``systems/normal_forms/``)
    The subset that can serve as the normal-form template of a conjugacy: planar
    oscillators with a stable limit cycle whose phase–amplitude structure is available
    in *closed form*. See that subpackage.

Vmapping: ``flow`` is written for one initial condition; batch with
``eqx.filter_vmap(ode.flow, in_axes=(None, 0))(ts, u0s)``. ``SolverConfig`` has no array
leaves, so it is static under the filter transforms.
"""

import abc
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float


class SolverConfig(eqx.Module):
    """Every numerical setting of a flow, in one hashable object (no array leaves).

    **Arguments:** as for ``diffrax.diffeqsolve``. ``throw=False`` makes a failed batch
    element report through ``Solution.result`` instead of raising for the whole batch
    (under ``vmap``, ``throw=True`` raises if *any* element fails).
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

    def params(self) -> dict[str, Any]:
        """JSON-able description (for dataset metadata)."""
        return {
            "solver": type(self.solver).__name__,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }


DEFAULT_SOLVER_CONFIG = SolverConfig()


def diffeqsolve(
    rhs, ts: Float[Array, " time"], u0, args, config: SolverConfig
) -> dfx.Solution:
    """``diffrax.diffeqsolve`` of ``rhs`` saved at ``ts``, configured by ``config``."""
    return dfx.diffeqsolve(
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


def _jsonable(v):
    v = np.asarray(v)
    return v.item() if v.ndim == 0 else v.tolist()


class AbstractODE(eqx.Module):
    """An autonomous ODE ``du/dt = rhs(t, u)`` on ``R^dim``."""

    dim: eqx.AbstractVar[int]
    """Implemented as ``eqx.field(static=True, default=..., init=False)`` (ADR-0004)."""

    @abc.abstractmethod
    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " {self.dim}"], args: Any = None
    ) -> Float[Array, " {self.dim}"]:
        """The vector field, in the diffrax signature (``t`` may be a Python float)."""

    def in_basin(self, u: Float[Array, " {self.dim}"]) -> Bool[Array, ""]:
        """Whether ``u`` lies in the basin of attraction of the system's limit cycle —
        the domain on which an asymptotic phase exists. ``True`` everywhere by default
        (a basin whose complement has measure zero, like FitzHugh–Nagumo's single
        phaseless point); a normal form with a phaseless *set* overrides it, and
        ``data.generate`` uses it to reject or resample initial conditions."""
        del u
        return jnp.asarray(True)

    def params(self) -> dict[str, Any]:
        """The system's parameters as JSON-able *constrained* values, keyed by their
        mathematical names — what a dataset's metadata records. The default is every
        non-static field; a system that stores unconstrained leaves overrides this to
        report the constrained values (``AbstractNormalForm`` does)."""
        return {
            f.name: _jsonable(getattr(self, f.name))
            for f in self.__dataclass_fields__.values()
            if not f.metadata.get("static", False)
        }

    def flow(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " {self.dim}"],
        args: Any = None,
        *,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> dfx.Solution:
        """The diffrax ``Solution`` through ``u0`` sampled at ``ts``: ``.ys`` is the
        trajectory ``(time, dim)``, ``.result`` the outcome (compare with
        ``diffrax.RESULTS.successful``; elementwise under ``vmap`` with
        ``config.throw=False``), ``.stats`` the step counts."""
        return diffeqsolve(self.rhs, ts, u0, args, config)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{cls}({args})"


__all__ = ["AbstractODE", "DEFAULT_SOLVER_CONFIG", "SolverConfig", "diffeqsolve"]
