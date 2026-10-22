r"""The Hodgkin–Huxley model as used by Yawata et al. (2024) for their phase
autoencoder. The equations in the paper have typos; the rates below are those of
the authors' source code
(https://github.com/kyoukuntaro/PhaseAutoencoder/blob/main/utils/limitcycle.py).
Declared symbolically (``AbstractSymbolicODE``); ``rhs`` is derived. The two
removable singularities of $\alpha_m$, $\alpha_n$ are written with ``expm1`` so the
compiled field keeps its precision near $V = -40$ and $V = -55$.
"""

import equinox as eqx
import sympy as sp
from sympy.codegen.cfunctions import expm1

from .symbolic import AbstractSymbolicODE, SymbolicODE


class HodgekinHuxley(AbstractSymbolicODE):
    r"""Four-dimensional Hodgkin–Huxley: $(V, m, h, n)$ with constant current $I$.

    **Attributes:**

    - `I`: Applied current.
    - `C`: Membrane capacitance.
    - `G_Na`, `G_K`, `G_L`: Maximal conductances.
    - `E_Na`, `E_K`, `E_L`: Reversal potentials.
    """

    dim: int = eqx.field(static=True, default=4, init=False)

    I: float = 30.0  # noqa: E741 (the model's own symbol)
    C: float = 1.0
    G_Na: float = 120.0
    G_K: float = 36.0
    G_L: float = 0.3
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.4

    @classmethod
    def _symbolic_template(cls) -> SymbolicODE:
        V, m, h, n = sp.symbols("V m h n", real=True)
        I, C, G_Na, G_K, G_L, E_Na, E_K, E_L = sp.symbols(  # noqa: E741
            "I C G_Na G_K G_L E_Na E_K E_L", real=True
        )
        alpha_m = -sp.Rational(1, 10) * (V + 40) / expm1(-(V + 40) / 10)
        beta_m = 4 * sp.exp(-(V + 65) / 18)
        alpha_h = sp.Rational(7, 100) * sp.exp(-(V + 65) / 20)
        beta_h = 1 / (1 + sp.exp(-(V + 35) / 10))
        alpha_n = -sp.Rational(1, 100) * (V + 55) / expm1(-(V + 55) / 10)
        beta_n = sp.Rational(1, 8) * sp.exp(-(V + 65) / 80)
        dV = (
            G_Na * m**3 * h * (E_Na - V) + G_K * n**4 * (E_K - V) + G_L * (E_L - V) + I
        ) / C
        return SymbolicODE(
            states=(V, m, h, n),
            params={
                "I": I,
                "C": C,
                "G_Na": G_Na,
                "G_K": G_K,
                "G_L": G_L,
                "E_Na": E_Na,
                "E_K": E_K,
                "E_L": E_L,
            },
            rhs=(
                dV,
                alpha_m * (1 - m) - beta_m * m,
                alpha_h * (1 - h) - beta_h * h,
                alpha_n * (1 - n) - beta_n * n,
            ),
        )
