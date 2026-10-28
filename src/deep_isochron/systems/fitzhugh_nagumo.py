r"""The FitzHugh–Nagumo model in Winfree's parametrization, with his time scaling.

The equations are those of Langfield et al. 2014, Eq. 15, with their parameters
$a = 0.7$, $b = 0.8$, $c = 3$, $z = -0.4$:

$$\dot x = c\bigl(y + x - x^3/3 + z\bigr), \qquad \dot y = -(x - a + b\thinspace y)/c.$$

It is a planar slow–fast system: $x$ is the fast variable, $c^2$ times faster than $y$.
A source at $(0.9066, -0.2582)$ has eigenvalues $0.1339 \pm 0.9163\thinspace i$, and
the attracting cycle has period $T \approx 11.2279$; its isochrons make sharp turns
near the repelling slow manifold (Langfield et al. 2014, §4). The vector field is
declared symbolically, so ``rhs`` and the AUTO Fortran are derived (ADR-0016).
"""

import equinox as eqx
import sympy as sp

from .symbolic import AbstractSymbolicODE, SymbolicODE


class FitzhughNagumo(AbstractSymbolicODE):
    r"""$\dot x = c(y + x - x^3/3 + z)$, $\dot y = -(x - a + by)/c$.

    **Attributes:**

    - `a`, `b`, `c`, `z`: The parameters, plain floats: this is an observed system,
        not a learnable one.
    """

    dim: int = eqx.field(static=True, default=2, init=False)

    a: float = 0.7
    b: float = 0.8
    c: float = 3.0
    z: float = -0.4

    @classmethod
    def _symbolic_template(cls) -> SymbolicODE:
        """The FitzHugh–Nagumo vector field as SymPy (`SymbolicODE`)."""
        x, y = sp.symbols("x y", real=True)
        a, b, c, z = sp.symbols("a b c z", real=True)
        return SymbolicODE(
            states=(x, y),
            params={"a": a, "b": b, "c": c, "z": z},
            rhs=(c * (y + x - x**3 / 3 + z), -(x - a + b * y) / c),
        )
