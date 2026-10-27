r"""The Bautin (generalized Hopf) normal form: $\rho(r) = a(1 - r^2)(1 + b r^2)$ and
$\omega(r) = \omega_0 + (\omega_1 - \omega_0) r^2$.

The quintic factor makes the radial contraction depend on $r$, which decouples the
cycle's Floquet exponent $\kappa = -2a(1 + b)$ from the focus's instability $a$. The
closed forms $h(r)$ and $\Psi(r)$ are in the design document §3.2; they are written
here in $r^{-2}$ so that they stay finite up to the float range, where the textbook
forms give ``inf/inf``. For $b \ge 0$ the cycle attracts the punctured plane; for
$-1 < b < 0$ an unstable outer cycle at $r^2 = -1/b$ bounds the basin, the two-cycle
regime of the Bautin bifurcation (Kuznetsov 2004, Ch. 8). As a conjugacy target use
$b \ge 0$.

The parameters $a > 0$ and $b > -1$ are constrained leaves through ``A_CONSTRAINT`` and
``B_CONSTRAINT``; $b > -1$ is exactly $\rho'(1) < 0$. $\omega_1$ (``w``) and
$\omega_0$ (``w0``) are free. The constructor takes constrained values.
"""

import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, Float

from ...model.invertible.constraints import GreaterThan
from ..symbolic import SymbolicODE
from .base import AbstractEvenNormalForm
from .hopf import A_CONSTRAINT


B_CONSTRAINT = GreaterThan(-1.0)
"""The constraint ``b > -1``, the stability of the cycle, with ``raw = 0`` mapped to
``b = 0``, the Hopf form."""


class BautinNormalForm(AbstractEvenNormalForm):
    r"""$\dot r = a r (1 - r^2)(1 + b r^2)$,
    $\dot\theta = \omega_0 + (\omega_1 - \omega_0) r^2$.

    **Arguments:**

    (constrained values)

    - `a`: The radial rate, positive.
    - `b`: The quintic coefficient, above $-1$.
    - `w`: $\omega_1$, the frequency on the cycle.
    - `w0`: $\omega_0$, the frequency at the origin; `w` when omitted.
    - `integration`: The default integration of `flow`.

    **Attributes:**

    - `raw_a`, `raw_b`: The unconstrained leaves of `a` and `b` (ADR-0001).
    - `w`, `w0`: As above, as scalar arrays.

    **Raises:**

    `ValueError` for ``a <= 0`` or ``b <= -1``.
    """

    raw_a: Float[Array, ""]
    raw_b: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""]

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.1,
        w: float = 1.0,
        w0: float | None = None,
        integration: str = "closed_form",
    ):
        self._set_integration(integration)
        if a <= 0:
            raise ValueError("a must be positive.")
        if b <= -1:
            raise ValueError("b must exceed -1 (stability of the cycle).")
        self.raw_a = A_CONSTRAINT.inverse(jnp.asarray(a, dtype=float))
        self.raw_b = B_CONSTRAINT.inverse(jnp.asarray(b, dtype=float))
        self.w = jnp.asarray(w, dtype=float)
        self.w0 = jnp.asarray(w if w0 is None else w0, dtype=float)

    @property
    def a(self) -> Float[Array, ""]:
        """The constrained radial rate, ``A_CONSTRAINT(raw_a)``."""
        return A_CONSTRAINT(self.raw_a)

    @property
    def b(self) -> Float[Array, ""]:
        """The constrained quintic coefficient, ``B_CONSTRAINT(raw_b)``."""
        return B_CONSTRAINT(self.raw_b)

    @classmethod
    def _symbolic_template_polar(cls) -> SymbolicODE:
        r"""$\dot r = a r (1 - r^2)(1 + b r^2)$,
        $\dot\theta = \omega_0 + (\omega_1 - \omega_0) r^2$."""
        r = sp.Symbol("r", positive=True)
        theta = sp.Symbol("theta", real=True)
        a, b, w, w0 = sp.symbols("a b w w0", real=True)
        return SymbolicODE(
            states=(r, theta),
            params={"a": a, "b": b, "w": w, "w0": w0},
            rhs=(a * r * (1 - r**2) * (1 + b * r**2), w0 + (w - w0) * r**2),
        )

    # The closed forms are written in 1/s = r⁻² so that they stay finite for r up to
    # the float range: the textbook forms (design document §3.2) pair an overflowing
    # numerator with an overflowing denominator (inf/inf = nan) from r ≈ 1e150 on,
    # which a learned bijection can reach before anything else goes wrong.
    def phase_shift(self, r):
        r"""$h(r) = c\thinspace[\ln r - \tfrac12\ln((1 + br^2)/(1 + b))]$ with
        $c = (\omega_1 - \omega_0)/a$, evaluated as
        $\tfrac12 c\thinspace[\ln(1 + b) - \ln(r^{-2} + b)]$."""
        b = self.b
        inv_s = 1.0 / (r * r)
        return 0.5 * (self.w - self.w0) / self.a * (jnp.log1p(b) - jnp.log(inv_s + b))

    def isostable(self, r):
        r"""$\Psi(r) = (r^2 - 1)(1 + br^2)^b / (2(1 + b)^b r^{2(1 + b)})$, evaluated as
        $(1 - r^{-2})(r^{-2} + b)^b / (2(1 + b)^b)$; it tends to
        $\tfrac12(b/(1 + b))^b$ as $r \to \infty$ for $b \ge 0$."""
        b = self.b
        inv_s = 1.0 / (r * r)
        return (1.0 - inv_s) * (inv_s + b) ** b / (2 * (1 + b) ** b)
