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
(Doedel; Dednam & Botha 2015, Eq. (7)).

Why multiple shooting (Morrison, Riley & Zancanaro 1962; Keller 1968; Deuflhard 1980):
the error amplification over one segment is $e^{\kappa_u T/m}$ instead of
$e^{\kappa_u T}$, so the Newton problem stays well conditioned on long or weakly
attracting orbits; $m = 1$ is single shooting. The segments are independent initial
value problems, so they are ``vmap``ped over.

Why the Jacobian is exact: it is forward-mode AD through the solver
(``diffrax.ForwardMode``), the derivative of the *numerical* flow the solver computed —
the property Guckenheimer & Meloon identify as the source of their accuracy ("no finite
difference calculations of derivatives", §6). Why Levenberg–Marquardt: Dednam & Botha
2015 use it to globalize shooting; near the solution it is Gauss–Newton, so nothing is
lost against Newton, and a poorer initial guess is survived. Grid coarsening (solve at
$m$, halve, finish at $m = 1$) follows BoundaryValueDiffEq.jl's ``MultipleShooting``.

The solver's tolerances are the floor of the residual: shooting needs them well below
the tolerance asked of the result (``SHOOTING_SOLVER_CONFIG``).
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from ..errors import NotConvergedError
from .sections import Hyperplane


SHOOTING_SOLVER_CONFIG = SolverConfig(
    solver=dfx.Dopri8(),
    rtol=1e-10,
    atol=1e-12,
    max_steps=1 << 16,
    adjoint=dfx.ForwardMode(),
)
"""A shooting-grade configuration: eighth order, tolerances ≈ 1e-3 of a 1e-8 target,
forward-mode adjoint for ``jax.jacfwd``."""


@dataclass(frozen=True)
class ShootingResult:
    """``points (m, dim)`` on the orbit at times ``i T / m``, ``period``, the final
    residual norm and the solver's step count."""

    points: Float[np.ndarray, "m dim"]
    period: float
    residual: float
    steps: int


def segment_flow(
    ode: AbstractODE,
    x: Float[Array, " dim"],
    duration: Float[Array, ""],
    config: SolverConfig,
    time_sign: float = 1.0,
) -> Float[Array, " dim"]:
    """``Φ_{±duration}(x)``, differentiable in ``x`` and ``duration`` under
    forward-mode AD when ``config.adjoint`` is ``diffrax.ForwardMode()``."""
    sol = dfx.diffeqsolve(
        dfx.ODETerm(ode.rhs),
        config.solver,
        t0=0.0,
        t1=time_sign * duration,
        dt0=None,
        y0=x,
        stepsize_controller=config.controller(),
        max_steps=config.max_steps,
        adjoint=config.adjoint,
        throw=config.throw,
    )
    return sol.ys[0]


def shooting_residual(y, args):
    """``optimistix`` residual: ``y = (points (m, dim), period)``, ``args = (ode,
    section, config, time_sign)``. Returns ``(matching (m, dim), phase ())``."""
    points, period = y
    ode, section, config, time_sign = args
    m = points.shape[0]
    ends = eqx.filter_vmap(
        lambda x: segment_flow(ode, x, period / m, config, time_sign)
    )(points)
    matching = ends - jnp.roll(points, -1, axis=0)
    phase = section(points[0])
    return matching, phase


def multiple_shooting(
    ode: AbstractODE,
    points: Float[ArrayLike, "m dim"],
    period: float,
    section: Hyperplane,
    *,
    config: SolverConfig = SHOOTING_SOLVER_CONFIG,
    reverse_time: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100,
    coarsen: bool = True,
) -> ShootingResult:
    """Solve the shooting equations from the guess ``(points, period)``; ``points[0]``
    should lie on ``section`` (the phase condition). With ``coarsen`` the problem is
    solved at ``m``, then at ``m // 2``, ..., and finally at ``m = 1`` (the points
    re-sampled from the current solution by the flow each time), so the result is a
    single-shooting solution started from a well-conditioned multiple-shooting one.
    Raises ``NotConvergedError`` with the solver's last iterate in ``last``."""
    if not isinstance(config.adjoint, dfx.ForwardMode):
        config = config.with_adjoint(dfx.ForwardMode())
    time_sign = -1.0 if reverse_time else 1.0
    pts = jnp.asarray(points, dtype=float)
    T = jnp.asarray(period, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 1:
        raise ValueError("points must be (m, dim) with m >= 1.")
    if not float(T) > 0:
        raise ValueError("period must be positive.")

    schedule = [pts.shape[0]]
    if coarsen:
        while schedule[-1] > 1:
            schedule.append(max(schedule[-1] // 2, 1))
    args = (ode, section, config, time_sign)
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    steps = 0
    for m in schedule:
        if m != pts.shape[0]:
            times = T * jnp.arange(m) / m
            pts = _resample(ode, pts[0], times, config, time_sign)
        sol = optx.least_squares(
            shooting_residual,
            solver,
            (pts, T),
            args=args,
            max_steps=max_steps,
            throw=False,
        )
        steps += int(sol.stats["num_steps"])
        pts, T = sol.value
        if sol.result != optx.RESULTS.successful:
            raise NotConvergedError(
                "shooting",
                f"Levenberg–Marquardt did not converge at m = {m}: {sol.result}",
                last=ShootingResult(np.asarray(pts), float(T), float("nan"), steps),
            )
    matching, phase = shooting_residual((pts, T), args)
    residual = float(jnp.sqrt(jnp.sum(matching**2) + phase**2))
    return ShootingResult(np.asarray(pts), float(T), residual, steps)


def _resample(ode, x0, times, config, time_sign):
    """Points of the orbit through ``x0`` at ``times`` (``times[0] == 0``)."""
    if times.shape[0] == 1:
        return x0[None]
    rest = jax.vmap(lambda t: segment_flow(ode, x0, t, config, time_sign))(times[1:])
    return jnp.concatenate([x0[None], rest], axis=0)


__all__ = [
    "SHOOTING_SOLVER_CONFIG",
    "ShootingResult",
    "multiple_shooting",
    "segment_flow",
    "shooting_residual",
]
