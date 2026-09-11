import abc
from typing import Any

import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, Float


class AbstractODE(eqx.Module):
    """Abstract base class for dynamical systems governed by ordinary differential
    equations.

    Ordinary differential equations represented by subclasses of AbstractODE are meant
    to be numerically solved using either `diffrax.diffeqsolve` or the `solve_ode`
    function defined in this module.
    """

    dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        """Describes the right hand side of the differential equation.

        The method signature is chosen to match the requirements for diffrax.
        """
        ...

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
        return f"{cls}({args})"

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " {self.dim}"],
        args: Any = None,
        *,
        solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps=4096,
        **kwargs,
    ) -> Float[Array, "time {self.dim}"]:
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),  # ty: ignore
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=u0,
            args=args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(atol=atol, rtol=rtol),
            max_steps=max_steps,
        )
        return sol.ys
