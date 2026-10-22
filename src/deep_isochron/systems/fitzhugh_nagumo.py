r"""The FitzHugh–Nagumo model in Winfree's parametrization (Langfield, Krauskopf &
Osinga 2014, Eq. 15; their $a = 0.7$, $b = 0.8$, $c = 3$, $z = -0.4$), with the
time scaling of Winfree's original:

$$\dot x = c\bigl(y + x - x^3/3 + z\bigr), \qquad \dot y = -(x - a + b\thinspace y)/c.$$

A planar slow–fast system (fast $x$, slow $y$; $x$ is $c^2$ times faster): a source at
$(0.9066, -0.2582)$ with eigenvalues $0.1339 \pm 0.9163\thinspace i$ and an attracting
cycle of period $T \approx 11.2279$, whose isochrons make sharp turns near the repelling
slow manifold (Langfield et al. 2014, §4). The vector field is declared symbolically
(``AbstractSymbolicODE``); ``rhs`` and the AUTO Fortran are derived.
"""

import equinox as eqx
import sympy as sp

from .symbolic import AbstractSymbolicODE, SymbolicODE


class FitzhughNagumo(AbstractSymbolicODE):
    r"""$\dot x = c(y + x - x^3/3 + z)$, $\dot y = -(x - a + by)/c$.

    **Attributes:**

    - `a`, `b`, `c`, `z`: The parameters (plain floats; an observed system).
    """

    dim: int = eqx.field(static=True, default=2, init=False)

    a: float = 0.7
    b: float = 0.8
    c: float = 3.0
    z: float = -0.4

    @classmethod
    def _symbolic_template(cls) -> SymbolicODE:
        x, y = sp.symbols("x y", real=True)
        a, b, c, z = sp.symbols("a b c z", real=True)
        return SymbolicODE(
            states=(x, y),
            params={"a": a, "b": b, "c": c, "z": z},
            rhs=(c * (y + x - x**3 / 3 + z), -(x - a + b * y) / c),
        )
