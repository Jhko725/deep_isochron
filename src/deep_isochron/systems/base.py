r"""The ODE systems of the study, and the one class they all share.

``AbstractODE`` is any ODE used anywhere in the study: the observed oscillators
(FitzHugh–Nagumo, Hodgkin–Huxley), the normal forms, and the level-set test orbits.
It carries the vector field ``rhs``, the constrained parameter values ``params()``,
which dataset metadata records, and the numerical flow ``flow``, a diffrax solve whose
every setting sits in a ``SolverConfig``. Anything numerical about an ODE beyond
integrating it, such as locating its limit cycle or its Floquet multipliers, is a
function over ``AbstractODE`` in ``deep_isochron.analysis``, not a method.

``AbstractNormalForm`` (``systems/normal_forms/``) is the subset that can be the
normal-form template of a conjugacy: planar oscillators with a stable limit cycle whose
phase–amplitude structure is in closed form. ``AbstractSymbolicODE`` (``symbolic.py``)
is the subset declared as SymPy expressions, from which ``rhs`` and the AUTO-07p
Fortran are derived.

``flow`` is written for one initial condition. Batch it with
``eqx.filter_vmap(ode.flow, in_axes=(None, 0))(ts, u0s)``; ``SolverConfig`` has no
array leaves, so it is static under the filter transforms.

ADR-0007 has the hierarchy and the flow strategies; ADR-0016 the symbolic layer.
"""

import abc
import copy
import dataclasses
from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float


class SolverConfig(eqx.Module):
    """Every numerical setting of a flow, in one hashable object with no array leaves.

    **Attributes:** as for ``diffrax.diffeqsolve``.

    - `solver`: The diffrax solver; ``Tsit5`` by default, diffrax's own recommendation
        for a non-stiff problem.
    - `rtol`, `atol`: The tolerances of the ``PIDController``.
    - `max_steps`: The step budget of one solve.
    - `adjoint`: The adjoint; the default ``RecursiveCheckpointAdjoint`` supports
        reverse mode only, so a solve differentiated with ``jax.jacfwd`` takes
        ``ForwardMode`` through `with_adjoint`.
    - `throw`: With ``False``, a failed batch element reports through
        ``Solution.result`` instead of raising for the whole batch; under ``vmap``,
        ``True`` raises if any element fails.
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
        """The step-size controller: a ``PIDController`` at `rtol` and `atol`."""
        return dfx.PIDController(rtol=self.rtol, atol=self.atol)

    def with_adjoint(self, adjoint: dfx.AbstractAdjoint) -> "SolverConfig":
        """The same configuration with another adjoint.

        **Arguments:**

        - `adjoint`: The replacement, for instance ``diffrax.ForwardMode()`` for a
            solve that ``jax.jacfwd`` will differentiate.

        **Returns:** A new ``SolverConfig``; `self` is unchanged.
        """
        return copy.replace(self, adjoint=adjoint)

    def params(self) -> dict[str, Any]:
        """The solver name, tolerances and step budget as JSON-able values, for
        dataset metadata."""
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
    """Solve ``du/dt = rhs(t, u, args)`` from ``ts[0]`` to ``ts[-1]``, saving at ``ts``.

    **Arguments:**

    - `rhs`: The vector field, in the diffrax signature.
    - `ts`: The save times, increasing or decreasing; the solve runs from the first to
        the last.
    - `u0`: The initial state.
    - `args`: Passed through to `rhs`.
    - `config`: The solver, tolerances, step budget, adjoint and `throw` flag.

    **Returns:** The ``diffrax.Solution``.
    """
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
    """An autonomous ODE ``du/dt = rhs(t, u)`` on ``R^dim``.

    **Attributes:**

    - `dim`: The state dimension, implemented by every subclass as
        ``eqx.field(static=True, default=..., init=False)`` (ADR-0004).
    """

    dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " {self.dim}"], args: Any = None
    ) -> Float[Array, " {self.dim}"]:
        """The vector field at state `u`, in the diffrax signature.

        **Arguments:**

        - `t`: The time; unused by an autonomous system, and may be a Python float.
        - `u`: The state, shape ``(dim,)``.
        - `args`: Unused; present for diffrax.

        **Returns:** ``du/dt``, shape ``(dim,)``.
        """

    def in_basin(self, u: Float[Array, " {self.dim}"]) -> Bool[Array, ""]:
        """Whether `u` lies in the basin of attraction of the limit cycle, the domain
        on which an asymptotic phase exists.

        The default is ``True`` everywhere, for a basin whose complement has measure
        zero, such as FitzHugh–Nagumo's single phaseless point. A normal form with a
        phaseless set overrides it, and ``data.generate`` uses it to reject or resample
        initial conditions (ADR-0013).

        **Arguments:**

        - `u`: The state, shape ``(dim,)``.

        **Returns:** A scalar boolean array.
        """
        del u
        return jnp.asarray(True)

    def params(self) -> dict[str, Any]:
        """The parameters as JSON-able constrained values keyed by their mathematical
        names, which is what a dataset's metadata records.

        The default reports every non-static field. A system that stores unconstrained
        leaves overrides it to report the constrained values, as the learnable normal
        forms do.
        """
        return {
            f.name: _jsonable(getattr(self, f.name))
            for f in dataclasses.fields(self)  # excludes ClassVar pseudo-fields
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
        """The trajectory through `u0` sampled at `ts`, by a diffrax solve.

        The normal forms override this with a choice of integration, closed form by
        default (``normal_forms.integration``).

        **Arguments:**

        - `ts`: The save times, shape ``(time,)``; the solve runs from ``ts[0]`` to
            ``ts[-1]``, so a decreasing `ts` integrates backward.
        - `u0`: The initial state, shape ``(dim,)``.
        - `args`: Passed through to `rhs`.
        - `config`: The solver settings; ``DEFAULT_SOLVER_CONFIG`` when omitted.

        **Returns:** The ``diffrax.Solution``: ``.ys`` is the trajectory, shape
        ``(time, dim)``; ``.result`` the outcome, to compare with
        ``diffrax.RESULTS.successful`` (elementwise under ``vmap`` when
        ``config.throw`` is ``False``); ``.stats`` the step counts.
        """
        return diffeqsolve(self.rhs, ts, u0, args, config)

    def flow_to(
        self,
        t: Float[ArrayLike, ""],
        u0: Float[Array, " {self.dim}"],
        args: Any = None,
        *,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> Float[Array, " {self.dim}"]:
        """The state at time `t` from `u0`: ``flow(ts=[0, t]).ys[-1]``.

        **Arguments:**

        - `t`: The time; negative integrates the reversed flow. May be traced.
        - `u0`: The initial state, shape ``(dim,)``.
        - `args`: Passed through to `rhs`.
        - `config`: The solver settings.

        **Returns:** The state, shape ``(dim,)``, differentiable in `u0` and `t` under
        the mode that ``config.adjoint`` supports.
        """
        ts = jnp.stack([jnp.zeros_like(t), jnp.asarray(t)])
        ys = cast(Array, self.flow(ts, u0, args, config=config).ys)
        return ys[-1]

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{cls}({args})"


__all__ = ["AbstractODE", "DEFAULT_SOLVER_CONFIG", "SolverConfig", "diffeqsolve"]
