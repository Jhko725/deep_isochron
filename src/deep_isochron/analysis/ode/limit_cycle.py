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

The result is a ``LimitCycle``: the converged shooting mesh ``x0s`` (the first point
on the section), the period, and the ODE and solver configuration it was found with,
so that every later analysis of the orbit (its sampling, ``LimitCycle.sample``; its
Floquet multipliers, ``floquet``) integrates the same system the same way. The orbit's
properties beyond its location are functions *of* a ``LimitCycle`` (roadmap O2–O6),
not products of the search.

**Jit.** The stages (``settle``, ``polish``, ``LimitCycle.sample``) and the map are
naturally jit- and vmap-compatible and are left unjitted — the caller chooses the
level. ``find_limit_cycle`` and ``PoincareMap.fixed_point`` are host-side drivers (they
run once per system and raise informative errors) and jit the stages they call. A
``LimitCycle`` is a fixed-shape pytree (its ``settle`` record is the light
``FixedPoint``), so functions of a cycle can be jitted and vmapped as they stand.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from ..data.cycle import orientation
from ..errors import NotConvergedError
from .poincare import FixedPoint, PoincareMap
from .sections import CrossSection
from .shooting import multiple_shooting, SHOOTING_SOLVER_CONFIG, ShootingResult


class LimitCycle(eqx.Module):
    """A periodic orbit located by ``find_limit_cycle``.

    **Attributes:**

    - `ode`: The `AbstractODE` the orbit belongs to.
    - `x0s`: Starting points `(m, dim)` of the converged shooting segments, at times
        `i T / m` along the orbit; `x0s[0]` lies on `section`.
    - `period`: The period $T > 0$.
    - `residual`: The norm of the shooting residual at the solution.
    - `settle`: The `FixedPoint` of the return map that gave the initial guess (its
        return time `tau` is the brute-force period, its `contraction` the
        brute-force multiplier estimate).
    - `section`: The `CrossSection` through `x0s[0]`, normal to the flow there.
    - `config`: The `SolverConfig` the orbit was found with, used by every method and
        analysis of the orbit (static).
    - `reverse_time`: Whether the orbit was found in the time-reversed flow — then it
        is a *repelling* orbit of `ode` (static).
    """

    ode: AbstractODE
    x0s: Float[Array, "m dim"]
    period: Float[Array, ""]
    residual: Float[Array, ""]
    settle: FixedPoint
    section: CrossSection
    config: SolverConfig = eqx.field(static=True)
    reverse_time: bool = eqx.field(static=True, default=False)

    @property
    def dim(self) -> int:
        return self.x0s.shape[1]

    @property
    def m(self) -> int:
        """The number of shooting segments."""
        return self.x0s.shape[0]

    def sample(
        self, n_points: int = 256
    ) -> tuple[Float[Array, " n"], Float[Array, "n dim"]]:
        """The orbit at `n_points` uniform times in $[0, T)$, by integrating from
        `x0s[0]` with `config`: a phase-uniform parametrization, so
        `2π times / period` is the asymptotic phase on the orbit up to a constant.

        **Arguments:**

        - `n_points`: The number of sample times.

        **Returns:**

        `(times, points)` with `times (n,)` and `points (n, dim)`,
        `points[0] == x0s[0]`, in forward time order (a reverse-time orbit is sampled
        backward and re-ordered). Jit- and vmap-compatible.
        """
        x0 = self.x0s[0]
        times = jnp.linspace(0.0, 1.0, n_points, endpoint=False) * self.period
        if not self.reverse_time:
            return times, cast(Array, self.ode.flow(times, x0, config=self.config).ys)
        # Φ_{-t}(x0) = Φ_{T-t}(x0): sample backward, then reverse the loop after x0
        back = cast(Array, self.ode.flow(-times, x0, config=self.config).ys)
        return times, jnp.concatenate([back[:1], back[1:][::-1]], axis=0)

    def orientation(self, n_points: int = 64) -> int:
        """$+1$ for a counterclockwise orbit (positive signed area of the sampled
        polygon), $-1$ for a clockwise one. Planar only."""
        if self.dim != 2:
            raise ValueError("orientation is defined for planar orbits only.")
        _, points = self.sample(n_points)
        return orientation(points)


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
    in its basin and ``t_settle`` covers the transient. Jit-compatible."""
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
    return map: a point on the cycle to ``settle_tol`` and the return time (the
    iteration record is dropped; call ``PoincareMap.fixed_point`` for it)."""
    section = CrossSection.normal_to_flow(ode, p)
    pmap = PoincareMap(ode, section, config, t_max=t_max, reverse_time=reverse_time)
    fp, _ = pmap.fixed_point(p, tol=settle_tol, max_iter=max_returns)
    return section, fp


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
    ``i period_hat / m``. Jit-compatible."""
    sign = -1.0 if reverse_time else 1.0
    x_hat = jnp.asarray(x_hat, dtype=float)
    times = sign * jnp.linspace(0.0, period_hat, m, endpoint=False)
    guess = (
        cast(Array, ode.flow(times, x_hat, config=config).ys) if m > 1 else x_hat[None]
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


# ----------------------------------------------------------------- the driver ----
_settle = eqx.filter_jit(settle)
_polish = eqx.filter_jit(polish)
"""The stages as the host-side driver runs them."""


def find_limit_cycle(
    ode: AbstractODE,
    x0: Float[ArrayLike, " dim"],
    *,
    t_settle: float,
    m: int = 8,
    reverse_time: bool = False,
    config: SolverConfig = SHOOTING_SOLVER_CONFIG,
    t_max: float | None = None,
    max_returns: int = 100,
    settle_tol: float = 1e-6,
    tol: float = 1e-8,
) -> LimitCycle:
    """Locate the limit cycle reached from `x0`; see the module docstring.

    **Main arguments:**

    - `ode`: The `AbstractODE` instance whose limit cycle is sought.
    - `x0`: Initial condition for the limit cycle search. If the target is the stable
        limit cycle of `ode`, `x0` must be in the basin of attraction. This is
        explicitly checked for using `ode.in_basin`.
    - `t_settle`: Initial integration time for `x0` to settle to the vicinity of the
        limit cycle (duration of the transient dynamics from `x0`). Insufficient
        convergence to the limit cycle is caught by inspecting the iterates of the
        one-sided Poincaré map (see `initial_guess`). If `reverse_time` is True,
        corresponds to the backward time integration duration.
    - `m`: Number of shooting segments. `m=1` corresponds to single shooting.
    - `reverse_time`: Whether to detect limit cycles using the reverse-time flow.
        Relevant for finding repelling cycles. If True, the `ode.in_basin` check is
        skipped since the forward basin of attraction is irrelevant for the repelling
        cycle.
    - `config`: `SolverConfig` used for the numerical integration of flows during all
        stages of the algorithm, and recorded on the result for later analyses.

    **Other arguments:**

    - `t_max`: Bound on one return time of the Poincaré map. None corresponds to the
        default choice of `10 t_settle`.
    - `max_returns`: Maximum iterations of the Poincaré map used while searching for
        its fixed point.
    - `settle_tol`: Tolerance of the fixed-point iteration of the Poincaré map. Does
        not have to be stringent due to the subsequent polishing step. For weakly
        attracting cycles the value must be loosened, since the return map contracts
        slowly.
    - `tol`: Tolerance of the Levenberg–Marquardt optimizer used in multiple shooting
        (`rtol = tol`, `atol = 1e-2 tol`). Corresponds to the accuracy of the shooting
        result.

    **Returns:**

    A `LimitCycle`: the converged shooting mesh `x0s`, the `period`, the shooting
    `residual`, the `settle` record, the `section`, and the `ode`, `config` and
    `reverse_time` the orbit was found with.

    **Raises:**

    - `ValueError` if `x0` is outside `ode.in_basin` (forward time) or the settled
        point is an equilibrium.
    - `NotConvergedError` if a stage fails to converge (the stage is named).
    """
    x0 = jnp.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a vector.")
    if not reverse_time and not bool(ode.in_basin(x0)):
        raise ValueError("x0 is outside the basin of attraction of the limit cycle.")
    if not t_settle > 0:
        raise ValueError("t_settle must be positive.")
    t_max = 10.0 * t_settle if t_max is None else t_max

    p = _settle(ode, x0, t_settle, config=config, reverse_time=reverse_time)
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
    period_hat = float(fp.tau)
    shot = _polish(
        ode,
        fp.x,
        period_hat,
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
    if not period_consistent(period, period_hat):
        raise NotConvergedError(
            "shooting",
            f"the period collapsed or exploded: T = {period:.6g} vs the return-map "
            f"estimate {period_hat:.6g}.",
            shot,
        )
    return LimitCycle(
        ode=ode,
        x0s=shot.points,
        period=shot.period,
        residual=shot.residual,
        settle=fp,
        section=section,
        config=config,
        reverse_time=reverse_time,
    )


__all__ = [
    "LimitCycle",
    "find_limit_cycle",
    "initial_guess",
    "is_equilibrium",
    "period_consistent",
    "polish",
    "settle",
]
