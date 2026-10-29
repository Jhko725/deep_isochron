r"""Multiple shooting for a periodic orbit, on the time-$T$ flow.

The unknowns are $m$ points $x_1, \dots, x_m$ and the period $T$, and the residual is
the $m$ matching defects $\Phi_{T/m}(x_i) - x_{i+1}$, cyclic, with the phase condition
$\langle n, x_1 - p\rangle$ pinning the first point to the cross-section (Guckenheimer
& Meloon 2000, §2, "forward multiple shooting"; Parker & Chua 1989, §5.5 for
$m = 1$). The phase condition removes the null direction of the matching equations
along the flow; it is admissible iff $\Sigma$ is transversal, which the flow-normal
section guarantees. ``multiple_shooting`` solves the equations by Levenberg–Marquardt
with the exact Jacobian, forward-mode AD through the solver; the segments are
independent initial value problems and are ``vmap``ped. ``SHOOTING_SOLVER_CONFIG`` is
the shooting-grade solver configuration, whose tolerances are the floor of the
residual.

``docs/design/limit-cycle.md`` has the survey: why multiple shooting (the error
amplification over one segment is $e^{\kappa_u T/m}$ rather than $e^{\kappa_u T}$,
so the Newton problem stays well conditioned on long or weakly attracting orbits),
why the exact Jacobian, and why Levenberg–Marquardt rather than Newton. ADR-0014
has the decisions.
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
"""The shooting-grade configuration: ``Dopri8`` at tolerances about 1e-3 of a 1e-8
target, with the forward-mode adjoint for ``jax.jacfwd``. At tolerances of
$10^{-10}$–$10^{-12}$ an eighth-order method needs far fewer steps than a fifth-order
one, the classical rule that high order pays at stringent tolerances (Hairer et al.
1993, §II.4); ``Tsit5`` at the same tolerances gives the same answers more slowly."""


class ShootingResult(NamedTuple):
    """The result of `multiple_shooting`, as arrays, since the function is jittable.

    **Attributes:**

    - `points`: The points on the orbit at times ``i T / m``, shape ``(m, dim)``.
    - `period`: $T$.
    - `residual`: The final residual norm; ``nan`` when the solve did not converge.
    - `steps`: The solver's step count.
    """

    points: Float[Array, "m dim"]
    period: Float[Array, ""]
    residual: Float[Array, ""]
    steps: Array


class ShootingArgs(NamedTuple):
    """The fixed data of the shooting residual: the system, the section of the phase
    condition, the solver settings and the time direction."""

    ode: AbstractODE
    section: CrossSection
    config: SolverConfig
    reverse_time: bool


def shooting_residual(
    y: tuple[Float[Array, "m dim"], Float[Array, ""]], args: ShootingArgs
) -> tuple[Float[Array, "m dim"], Float[Array, ""]]:
    """The shooting residual in the ``optimistix`` form: the matching defects
    ``Φ_{T/m}(x_i) − x_{i+1}``, cyclic, and the phase condition ``⟨n, x_1 − p⟩``.

    **Arguments:**

    - `y`: The unknowns ``(points, period)``, shapes ``(m, dim)`` and ``()``.
    - `args`: The fixed data.

    **Returns:**

    ``(matching, phase)``, shapes ``(m, dim)`` and ``()``.
    """
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
    Levenberg–Marquardt. Jit-compatible and left unjitted; ``find_limit_cycle`` jits
    it as a stage.

    **Arguments:**

    - `ode`: The system.
    - `points`: The initial mesh, shape ``(m, dim)``; ``points[0]`` should lie on
        `section`, which is the phase condition.
    - `period`: The initial period.
    - `section`: The cross-section of the phase condition.
    - `config`: The solver settings; the adjoint is replaced by
        ``diffrax.ForwardMode`` if it is anything else.
    - `reverse_time`: Whether to shoot in the time-reversed flow.
    - `rtol`, `atol`: The Levenberg–Marquardt tolerances.
    - `max_steps`: The bound on its iterations.
    - `throw`: Whether a failure to converge is a runtime error, as in optimistix;
        otherwise the result carries a ``nan`` residual.

    **Returns:**

    The `ShootingResult`.

    **Raises:**

    `ValueError` unless `points` has shape ``(m, dim)`` with ``m >= 1``.
    """
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
