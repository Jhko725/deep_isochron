r"""``find_limit_cycle``: the attracting (or, in reverse time, repelling) periodic orbit
of a planar ODE and its period, as a composition of stages.

1. ``settle`` (Parker & Chua 1989, §5.1.1, brute force): integrate from ``x0`` for
   ``t_settle``; the end point $p$ is near the cycle.
2. ``initial_guess``: $\Sigma$ through $p$ normal to the flow
   (``CrossSection.normal_to_flow``); the one-sided map $P_+$ on it (``PoincareMap``)
   iterated to its fixed point (their minimum-period test, §5.5), giving a point
   $\hat x \in \Sigma$ on the cycle to ``settle_tol``, the return time $\hat T$ (over
   the $k$ map steps of a period-$k$ orbit), and the brute-force multiplier estimate.
3. ``polish`` (``multiple_shooting``): $m$ points of the orbit through $\hat x$ at times
   $i\hat T/m$ and $\hat T$, refined by Levenberg–Marquardt on the shooting equations
   with AD Jacobians.
4. Guards (Parker & Chua §5.5, "Extraneous solutions"): the shooting equations are
   also solved by an equilibrium (``is_equilibrium``) and by $T \to 0$
   (``period_consistent`` against $\hat T$); both are rejected.
5. ``sample_orbit``: the orbit at ``n_points`` uniform times in $[0, T)$ (a
   phase-uniform parametrization), and ``orientation`` as the sign of its signed area.

The orbit's own properties beyond its location — the monodromy, the Floquet multipliers,
their cross-checks — are functions *of* a ``LimitCycle`` (roadmap O2), not products of
the search, and are not computed here. ``find_limit_cycle`` is a host-side driver (it
runs once per system and raises informative errors); every stage it calls is jitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from ..errors import NotConvergedError
from .poincare import FixedPoint, PoincareMap
from .sections import CrossSection
from .shooting import multiple_shooting, SHOOTING_SOLVER_CONFIG, ShootingResult


@dataclass(frozen=True)
class LimitCycle:
    """A periodic orbit located by ``find_limit_cycle`` (host-side NumPy).

    - ``x0 (dim,)``: a point of the orbit (on ``section``); ``period`` $T > 0$.
    - ``times (n,)``, ``points (n, dim)``: the orbit at ``n`` uniform times in
      $[0, T)$, ``points[0] == x0``; phase-uniform, so ``2π times / period`` is the
      asymptotic phase on the cycle up to a constant.
    - ``orientation``: $+1$ counterclockwise (positive signed area), $-1$ clockwise.
    - ``section``: the cross-section used; ``residual``: the shooting residual norm;
      ``settle``: the fixed-point iteration that produced the initial guess;
      ``reverse_time``: whether the orbit was found in the time-reversed flow (then it
      is a *repelling* orbit of the flow).
    """

    x0: Float[np.ndarray, " dim"]
    period: float
    times: Float[np.ndarray, " n"]
    points: Float[np.ndarray, "n dim"]
    orientation: int
    section: CrossSection
    residual: float
    settle: FixedPoint
    reverse_time: bool = False


# ------------------------------------------------------------------ the stages ----
def settle(
    ode: AbstractODE,
    x0: Float[ArrayLike, " dim"],
    t_settle: float,
    *,
    config: SolverConfig,
    reverse_time: bool = False,
) -> Float[Array, " dim"]:
    """The state after ``t_settle`` (Parker & Chua §5.1.1): near the cycle if ``x0`` is
    in its basin and ``t_settle`` covers the transient."""
    sign = -1.0 if reverse_time else 1.0
    return ode.flow_to(sign * t_settle, jnp.asarray(x0, dtype=float), config=config)


def initial_guess(
    ode: AbstractODE,
    p: Float[ArrayLike, " dim"],
    *,
    config: SolverConfig,
    t_max: float,
    settle_tol: float,
    max_returns: int,
    reverse_time: bool = False,
) -> tuple[CrossSection, FixedPoint]:
    """The flow-normal section through ``p`` and the fixed point of its one-sided
    return map: a point on the cycle to ``settle_tol`` and the return time."""
    section = CrossSection.normal_to_flow(ode, p)
    pmap = PoincareMap(ode, section, config, t_max=t_max, reverse_time=reverse_time)
    return section, pmap.fixed_point(p, tol=settle_tol, max_iter=max_returns)


def polish(
    ode: AbstractODE,
    x_hat: Float[ArrayLike, " dim"],
    period_hat: float,
    section: CrossSection,
    *,
    m: int,
    config: SolverConfig,
    tol: float,
    reverse_time: bool = False,
) -> ShootingResult:
    """Multiple shooting from ``m`` points of the orbit through ``x_hat`` at times
    ``i period_hat / m``."""
    sign = -1.0 if reverse_time else 1.0
    x_hat = jnp.asarray(x_hat, dtype=float)
    times = sign * period_hat * jnp.arange(m) / m
    guess = (
        cast(Array, ode.integrate(times, x_hat, config=config).ys)
        if m > 1
        else x_hat[None]
    )
    return multiple_shooting(
        ode,
        guess,
        period_hat,
        section,
        config=config,
        reverse_time=reverse_time,
        rtol=tol,
        atol=tol * 1e-2,
    )


def is_equilibrium(
    ode: AbstractODE, x: Float[ArrayLike, " dim"], rel_tol: float = 1e-6
) -> bool:
    """Whether the vector field vanishes at ``x`` (relative to ``max(1, |x|)``) — the
    first extraneous solution of shooting (Parker & Chua §5.5)."""
    x = jnp.asarray(x, dtype=float)
    speed = float(jnp.linalg.norm(ode.rhs(0.0, x)))
    return speed <= rel_tol * max(1.0, float(jnp.linalg.norm(x)))


def period_consistent(period: float, period_hat: float, factor: float = 10.0) -> bool:
    """Whether the polished period is within ``factor`` of the return-map estimate —
    rejects the collapse to $T \to 0$, the second extraneous solution of shooting."""
    return period_hat / factor <= period <= factor * period_hat


def sample_orbit(
    ode: AbstractODE,
    x0: Float[ArrayLike, " dim"],
    period: float,
    n_points: int,
    *,
    config: SolverConfig,
    reverse_time: bool = False,
) -> tuple[Float[Array, " n"], Float[Array, "n dim"]]:
    """The orbit through ``x0`` at ``n_points`` uniform times in ``[0, period)``, in
    the forward time order (a reverse-time sampling is re-ordered)."""
    x0 = jnp.asarray(x0, dtype=float)
    times = period * jnp.arange(n_points) / n_points
    if not reverse_time:
        return times, cast(Array, ode.integrate(times, x0, config=config).ys)
    # Φ_{-t}(x0) = Φ_{T-t}(x0): sample backward, then reverse the loop after x0
    back = cast(Array, ode.integrate(-times, x0, config=config).ys)
    return times, jnp.concatenate([back[:1], back[1:][::-1]], axis=0)


def orientation(points: Float[ArrayLike, "n 2"]) -> int:
    """+1 for a counterclockwise closed curve (positive signed area), −1 clockwise."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return 1 if area > 0 else -1


# ----------------------------------------------------------------- the driver ----
def find_limit_cycle(
    ode: AbstractODE,
    x0: Float[ArrayLike, " dim"],
    *,
    t_settle: float,
    m: int = 8,
    n_points: int = 256,
    config: SolverConfig = SHOOTING_SOLVER_CONFIG,
    tol: float = 1e-8,
    settle_tol: float = 1e-6,
    max_returns: int = 100,
    t_max: float | None = None,
    reverse_time: bool = False,
) -> LimitCycle:
    """Locate the limit cycle reached from ``x0``; see the module docstring.

    ``t_settle``: integration time before the section is placed (a few transient
    times); ``m``: shooting segments; ``tol``: the shooting tolerance (Levenberg–
    Marquardt ``rtol``, ``atol = 1e-2 tol``) — the accuracy asked of the result;
    ``settle_tol``: the fixed-point tolerance of the return map, i.e. how good the
    initial guess must be before the polish (loosen it for weakly attracting cycles,
    whose returns contract slowly); ``max_returns``: bound on the map iterations;
    ``t_max``: bound on one return time (default ``10 t_settle``); ``reverse_time``:
    find a repelling cycle by iterating the reversed flow (the basin check is skipped,
    since the forward basin is not the relevant set).

    Raises ``ValueError`` if ``x0`` is outside ``ode.in_basin`` (forward time) or the
    settled point is an equilibrium, ``NotConvergedError`` from any stage.
    """
    x0 = jnp.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a vector.")
    if not reverse_time and not bool(ode.in_basin(x0)):
        raise ValueError("x0 is outside the basin of attraction of the limit cycle.")
    if not t_settle > 0:
        raise ValueError("t_settle must be positive.")
    t_max = 10.0 * t_settle if t_max is None else t_max

    p = settle(ode, x0, t_settle, config=config, reverse_time=reverse_time)
    if not bool(jnp.all(jnp.isfinite(p))):
        raise NotConvergedError("settle", "the trajectory left the finite domain.", p)
    section, fp = initial_guess(
        ode,
        p,
        config=config,
        t_max=t_max,
        settle_tol=settle_tol,
        max_returns=max_returns,
        reverse_time=reverse_time,
    )
    shot = polish(
        ode,
        fp.x,
        fp.tau,
        section,
        m=m,
        config=config,
        tol=tol,
        reverse_time=reverse_time,
    )
    x_star, period = shot.points[0], float(shot.period)
    if not bool(jnp.isfinite(shot.residual)):
        raise NotConvergedError(
            "shooting", "Levenberg–Marquardt did not converge.", shot
        )
    if is_equilibrium(ode, x_star):
        raise NotConvergedError(
            "shooting", "converged to an equilibrium (the vector field vanishes).", shot
        )
    if not period_consistent(period, fp.tau):
        raise NotConvergedError(
            "shooting",
            f"the period collapsed or exploded: T = {period:.6g} vs the return-map "
            f"estimate {fp.tau:.6g}.",
            shot,
        )
    times, points = sample_orbit(
        ode, x_star, period, n_points, config=config, reverse_time=reverse_time
    )
    return LimitCycle(
        x0=np.asarray(x_star),
        period=period,
        times=np.asarray(times),
        points=np.asarray(points),
        orientation=orientation(points),
        section=section,
        residual=float(shot.residual),
        settle=fp,
        reverse_time=reverse_time,
    )


__all__ = [
    "LimitCycle",
    "find_limit_cycle",
    "initial_guess",
    "is_equilibrium",
    "orientation",
    "period_consistent",
    "polish",
    "sample_orbit",
    "settle",
]
