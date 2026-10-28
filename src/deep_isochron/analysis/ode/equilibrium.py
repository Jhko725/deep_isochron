"""The equilibrium of a vector field near a guess, by Newton's method on the field.

``find_equilibrium(ode, guess)`` solves ``f(x) = 0`` with ``optimistix.Newton`` from
`guess` and returns the point with a convergence flag; a reference analysis that needs
the phaseless set's center, such as the log-polar view of an isochron, calls it with the
cycle's centroid as the guess. It is jit- and vmap-compatible and left unjitted.
"""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool, Float

from ...systems import AbstractODE


RESIDUAL_TOL = 1e-12
"""Newton stops when its step is below this, and the point counts as converged only when
``|f(x)|`` is below it too: at round-off for a field of order one."""


def find_equilibrium(
    ode: AbstractODE, guess: Float[Array, " dim"], *, max_steps: int = 30
) -> tuple[Float[Array, " dim"], Bool[Array, ""]]:
    """The equilibrium of `ode` nearest `guess`, by Newton's method on ``rhs``.

    **Arguments:**

    - `ode`: The vector field.
    - `guess`: A point near the equilibrium; for a planar cycle's phaseless point, the
        centroid of the sampled cycle.
    - `max_steps`: The Newton steps allowed.

    **Returns:**

    The point Newton stopped at, and whether it converged: a step below
    `RESIDUAL_TOL` within `max_steps` and ``|f(x)| < RESIDUAL_TOL`` there. The caller
    decides what an unconverged point means.
    """

    def field(x, args):
        del args
        return ode.rhs(0.0, x)

    sol = optx.root_find(
        field,
        optx.Newton(rtol=0.0, atol=RESIDUAL_TOL),
        jnp.asarray(guess, dtype=float),
        max_steps=max_steps,
        throw=False,
    )
    x = sol.value
    small = jnp.linalg.norm(field(x, None)) < RESIDUAL_TOL
    return x, (sol.result == optx.RESULTS.successful) & small


__all__ = ["find_equilibrium", "RESIDUAL_TOL"]
