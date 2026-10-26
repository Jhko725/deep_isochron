r"""Multiple shooting for a periodic orbit, on the time-$T$ flow.

Unknowns: $m$ points $x_1, \dots, x_m$ and the period $T$. Residual (Guckenheimer &
Meloon 2000, §2, "forward multiple shooting"; Parker & Chua 1989, §5.5 for $m = 1$):

$$
r = \bigl(\Phi_{T/m}(x_1) - x_2,\ \dots,\ \Phi_{T/m}(x_m) - x_1,
\ \langle n, x_1 - p\rangle\bigr)
$$

$2m + 1$ equations in $2m + 1$ unknowns (planar case). The last one is the phase
condition — the first point is pinned to the cross-section $\Sigma = (p, n)$ — which
removes the one-dimensional null direction of the matching equations along the flow
(sliding every point along the orbit at fixed $T$); it is admissible iff $\Sigma$ is
transversal, which the flow-normal section guarantees. Splitting the period into $m$
equal segments is the time rescaling of the BVP form $\dot x = T f(x)$ on $[0, 1]$
(Doedel; Dednam & Botha 2015, Eq. 7).

Why multiple shooting (Morrison et al. 1962; Keller 1968; Deuflhard 1980):
the error amplification over one segment is $e^{\kappa_u T/m}$ instead of
$e^{\kappa_u T}$, so the Newton problem stays well conditioned on long or weakly
attracting orbits; $m = 1$ is single shooting. The segments are independent initial
value problems, so they are ``vmap``ped over.

Why the Jacobian is exact: it is forward-mode AD through the solver
(``diffrax.ForwardMode``), the derivative of the *numerical* flow the solver computed —
the property Guckenheimer & Meloon identify as the source of their accuracy ("no finite
difference calculations of derivatives", §6). Why Levenberg–Marquardt: Dednam & Botha
2015 use it to globalize shooting; near the solution it is Gauss–Newton, so nothing is
lost against Newton, and a poorer initial guess is survived.

The solver's tolerances are the floor of the residual: shooting needs them well below
the tolerance asked of the result. ``SHOOTING_SOLVER_CONFIG`` uses ``Dopri8``: at
tolerances of $10^{-10}$–$10^{-12}$ an eighth-order method needs far fewer steps than a
fifth-order one (the classical rule that high order pays at stringent tolerances,
Hairer, Nørsett & Wanner I, §II.4); ``Tsit5`` at the same tolerances gives the same
answers more slowly.
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from .sections import CrossSection


SHOOTING_SOLVER_CONFIG = SolverConfig(
    solver=dfx.Dopri8(),
    rtol=1e-10,
    atol=1e-12,
    max_steps=65536,
    adjoint=dfx.ForwardMode(),
)
"""A shooting-grade configuration: eighth order, tolerances ≈ 1e-3 of a 1e-8 target,
forward-mode adjoint for ``jax.jacfwd``."""


class ShootingResult(NamedTuple):
    """``points (m, dim)`` on the orbit at times ``i T / m``, ``period``, the final
    residual norm and the solver's step count (arrays: the result of a jittable
    function)."""

    points: Float[Array, "m dim"]
    period: Float[Array, ""]
    residual: Float[Array, ""]
    steps: Array


class ShootingArgs(NamedTuple):
    """The fixed data of the shooting residual."""

    ode: AbstractODE
    section: CrossSection
    config: SolverConfig
    reverse_time: bool


def shooting_residual(
    y: tuple[Float[Array, "m dim"], Float[Array, ""]], args: ShootingArgs
) -> tuple[Float[Array, "m dim"], Float[Array, ""]]:
    """``optimistix`` residual: the matching defects ``Φ_{T/m}(x_i) − x_{i+1}``
    (cyclic) and the phase condition ``⟨n, x_1 − p⟩``."""
    points, period = y
    ode, section, config, reverse_time = args
    m = points.shape[0]
    duration = (-1.0 if reverse_time else 1.0) * period / m
    ends = eqx.filter_vmap(lambda x: ode.flow_to(duration, x, config=config))(points)
    matching = ends - jnp.roll(points, -1, axis=0)
    phase = section.signed_distance(points[0])
    return matching, phase


def multiple_shooting(
    ode: AbstractODE,
    points: Float[ArrayLike, "m dim"],
    period: Float[ArrayLike, ""],
    section: CrossSection,
    *,
    config: SolverConfig = SHOOTING_SOLVER_CONFIG,
    reverse_time: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100,
    throw: bool = True,
) -> ShootingResult:
    """Solve the shooting equations from the guess ``(points, period)`` by
    Levenberg–Marquardt; ``points[0]`` should lie on ``section`` (the phase condition).
    Jit-compatible, left unjitted (``find_limit_cycle`` jits its stages); a failure
    to converge is a runtime error when ``throw`` (as in optimistix), otherwise the
    result carries ``nan`` residual. ``config.adjoint`` is replaced by
    ``diffrax.ForwardMode`` if needed."""
    if not isinstance(config.adjoint, dfx.ForwardMode):
        config = config.with_adjoint(dfx.ForwardMode())
    pts = jnp.asarray(points, dtype=float)
    T = jnp.asarray(period, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 1:
        raise ValueError("points must be (m, dim) with m >= 1.")
    args = ShootingArgs(ode, section, config, reverse_time)
    sol = optx.least_squares(
        shooting_residual,
        optx.LevenbergMarquardt(rtol=rtol, atol=atol),
        (pts, T),
        args=args,
        max_steps=max_steps,
        throw=throw,
    )
    pts, T = sol.value
    matching, phase = shooting_residual((pts, T), args)
    residual = jnp.sqrt(jnp.sum(matching**2) + phase**2)
    residual = jnp.where(sol.result == optx.RESULTS.successful, residual, jnp.nan)
    return ShootingResult(pts, T, residual, sol.stats["num_steps"])


__all__ = [
    "SHOOTING_SOLVER_CONFIG",
    "ShootingArgs",
    "ShootingResult",
    "multiple_shooting",
    "shooting_residual",
]
