import abc

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
    def rhs(self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None):
        """Describes the right hand side of the differential equation.

        The method signature is chosen to match the requirements for diffrax.
        """
        ...

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
        return f"{cls}({args})"
