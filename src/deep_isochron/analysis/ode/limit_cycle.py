r"""``find_limit_cycle``: the attracting periodic orbit of a planar ODE and its period,
or the repelling one in reverse time, as a composition of stages.

``settle`` integrates from ``x0`` for ``t_settle`` to a point $p$ near the cycle
(Parker & Chua 1989, §5.1.1, brute force). ``initial_guess`` puts the section
$\Sigma$ through $p$ normal to the flow and iterates the one-sided return map
``PoincareMap`` on it to its fixed point, with their minimum-period test (§5.5): a
point $\hat x \in \Sigma$ on the cycle to ``settle_tol``, the return time $\hat T$ over
the $k$ map steps of a period-$k$ orbit, and a brute-force multiplier estimate.
``polish`` refines $m$ points of the orbit through $\hat x$ at times $i\hat T/m$, and
$\hat T$, by ``multiple_shooting``. Two guards reject the extraneous solutions of the
shooting equations (Parker & Chua §5.5): an equilibrium (``is_equilibrium``) and the
collapse $T \to 0$ (``period_consistent`` against $\hat T$).

The result is a ``LimitCycle``: the converged shooting mesh ``x0s``, the period, and
the ODE and solver configuration it was found with, so that every later analysis of
the orbit integrates the same system the same way. An orbit's properties beyond its
location are functions of a ``LimitCycle`` (roadmap O2–O6), not products of the
search.

``nearest_cycle_time`` finds the cycle point nearest a given point, as a time from
``x0s[0]``, and ``reanchor`` moves a cycle's phase origin there (roadmap O7, which
anchors its isochrons at phase 0) without solving the orbit again.

The stages and the map are jit- and vmap-compatible and left unjitted; the host-side
driver ``find_limit_cycle`` jits the stages it calls and raises informative errors.
Times reach the jitted stages as arrays, never as Python floats, which would be static
and compile a new program for every value. A
``LimitCycle`` is a fixed-shape pytree, so functions of a cycle can be jitted and
vmapped as they stand. ADR-0014 has the decisions; ``docs/design/limit-cycle.md`` the
survey of shooting methods.
"""

from __future__ import annotations

import copy
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike, Bool, Float

from ...systems import AbstractODE, SolverConfig
from ..data.cycle import orientation
from ..errors import NotConvergedError
from .poincare import FixedPoint, PoincareMap
from .sections import CrossSection
from .shooting import (
    multiple_shooting,
    shooting_residual,
    SHOOTING_SOLVER_CONFIG,
    ShootingArgs,
    ShootingResult,
)


TIME_TOL = 1e-14
"""The absolute tolerance of the Newton solve in time of `nearest_cycle_time`, in the
units of the period: Optimistix declares convergence when a step is below it."""


class LimitCycle(eqx.Module):
    """A periodic orbit located by ``find_limit_cycle``.

    **Attributes:**

    - `ode`: The `AbstractODE` the orbit belongs to.
    - `x0s`: The starting points of the converged shooting segments, shape
        ``(m, dim)``, at times ``i T / m`` along the orbit; ``x0s[0]`` lies on
        `section`.
    - `period`: The period $T > 0$.
    - `residual`: The norm of the shooting residual at the solution.
    - `settle`: The `FixedPoint` of the return map that gave the initial guess; its
        `tau` is the brute-force period and its `contraction` the brute-force
        multiplier estimate.
    - `section`: The `CrossSection` through ``x0s[0]``, normal to the flow there.
    - `config`: The `SolverConfig` the orbit was found with, used by every method and
        analysis of the orbit (static).
    - `reverse_time`: Whether the orbit was found in the time-reversed flow, in which
        case it is a repelling orbit of `ode` (static).
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
        """The state dimension."""
        return self.x0s.shape[1]

    @property
    def m(self) -> int:
        """The number of shooting segments."""
        return self.x0s.shape[0]

    def sample(
        self, n_points: int = 256
    ) -> tuple[Float[Array, " n"], Float[Array, "n dim"]]:
        """The orbit at `n_points` uniform times in $[0, T)$, by integrating from
        ``x0s[0]`` with `config`.

        The parametrization is uniform in phase, so ``2π times / period`` is the
        asymptotic phase on the orbit up to a constant.

        **Arguments:**

        - `n_points`: The number of sample times.

        **Returns:**

        ``(times, points)`` with shapes ``(n,)`` and ``(n, dim)``, ``points[0] ==
        x0s[0]``, in forward time order; a reverse-time orbit is sampled backward and
        reordered.
        """
        x0 = self.x0s[0]
        times = jnp.linspace(0.0, 1.0, n_points, endpoint=False) * self.period
        if not self.reverse_time:
            return times, cast(Array, self.ode.flow(times, x0, config=self.config).ys)
        # Φ_{-t}(x0) = Φ_{T-t}(x0): sample backward, then reverse the loop after x0
        back = cast(Array, self.ode.flow(-times, x0, config=self.config).ys)
        return times, jnp.concatenate([back[:1], back[1:][::-1]], axis=0)

    def orientation(self, n_points: int = 64) -> int:
        """$+1$ for a counterclockwise orbit, by the signed area of the polygon of
        `n_points` samples, and $-1$ for a clockwise one.

        **Raises:**

        `ValueError` for a non-planar orbit.
        """
        if self.dim != 2:
            raise ValueError("orientation is defined for planar orbits only.")
        _, points = self.sample(n_points)
        return orientation(points)


# ------------------------------------------------------------------ the stages ----
def settle(
    ode: AbstractODE,
    x0: Float[ArrayLike, " dim"],
    t_settle: Float[ArrayLike, ""],
    *,
    config: SolverConfig,
    reverse_time: bool = False,
) -> Float[Array, " dim"]:
    """The state reached from `x0` after `t_settle` (Parker & Chua 1989, §5.1.1):
    near the cycle if `x0` is in its basin and `t_settle` covers the transient.

    **Arguments:**

    - `ode`: The system.
    - `x0`: The initial condition, shape ``(dim,)``.
    - `t_settle`: The integration time.
    - `config`: The solver settings.
    - `reverse_time`: Whether to integrate the time-reversed flow.

    **Returns:**

    The state, shape ``(dim,)``.
    """
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
    """The flow-normal section through `p` and the fixed point of its one-sided
    return map: a point on the cycle to `settle_tol`, and the return time.

    The iteration record is dropped; call ``PoincareMap.fixed_point`` for it.

    **Arguments:**

    - `ode`: The system.
    - `p`: A point near the cycle, shape ``(dim,)``.
    - `config`: The solver settings.
    - `t_max`: The bound on one return time of the map.
    - `settle_tol`: The distance to an earlier iterate that counts as a return.
    - `max_returns`: The bound on the number of map iterations.
    - `reverse_time`: Whether to iterate the map of the time-reversed flow.

    **Returns:**

    ``(section, fixed_point)``.

    **Raises:**

    `ValueError` at an equilibrium; `NotConvergedError` from the map.
    """
    section = CrossSection.normal_to_flow(ode, p)
    pmap = PoincareMap(ode, section, config, t_max=t_max, reverse_time=reverse_time)
    fp, _ = pmap.fixed_point(p, tol=settle_tol, max_iter=max_returns)
    return section, fp


def polish(
    ode: AbstractODE,
    x_hat: Float[ArrayLike, " dim"],
    period_hat: Float[ArrayLike, ""],
    section: CrossSection,
    *,
    m: int,
    config: SolverConfig,
    tol: float,
    reverse_time: bool = False,
) -> ShootingResult:
    """Multiple shooting from `m` points of the orbit through `x_hat` at times
    ``i period_hat / m``.

    **Arguments:**

    - `ode`: The system.
    - `x_hat`: A point on the cycle, on `section`, shape ``(dim,)``.
    - `period_hat`: The period estimate.
    - `section`: The section of the phase condition.
    - `m`: The number of shooting segments; 1 is single shooting.
    - `config`: The solver settings.
    - `tol`: The Levenberg–Marquardt tolerance: ``rtol = tol``,
        ``atol = 1e-2 tol``.
    - `reverse_time`: Whether to shoot in the time-reversed flow.

    **Returns:**

    The `ShootingResult`.
    """
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
    """Whether the vector field of `ode` vanishes at `x`, to `rel_tol` relative to
    ``max(1, |x|)``: the first extraneous solution of shooting (Parker & Chua 1989,
    §5.5)."""
    x = jnp.asarray(x, dtype=float)
    speed = float(jnp.linalg.norm(ode.rhs(0.0, x)))
    return speed <= rel_tol * max(1.0, float(jnp.linalg.norm(x)))


def period_consistent(period: float, period_hat: float, factor: float = 10.0) -> bool:
    """Whether the polished `period` is within `factor` of the return-map estimate
    `period_hat`, which rejects the collapse $T \to 0$, the second extraneous solution
    of shooting."""
    return period_hat / factor <= period <= factor * period_hat


# ----------------------------------------------------------------- the driver ----
_settle = eqx.filter_jit(settle)
_polish = eqx.filter_jit(polish)  # the stages as the host-side drivers run them


def _check_shooting(ode: AbstractODE, shot: ShootingResult, period_hat: float) -> None:
    """Raise `NotConvergedError` unless `shot` converged to a periodic orbit whose
    period is consistent with `period_hat` (Parker & Chua 1989, §5.5)."""
    period = float(shot.period)
    if not bool(jnp.isfinite(shot.residual)):
        raise NotConvergedError(
            "shooting", "Levenberg–Marquardt did not converge.", shot
        )
    if is_equilibrium(ode, shot.points[0]):
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
    """Locate the limit cycle reached from `x0`: settle, return map, shooting, guards.

    **Main arguments:**

    - `ode`: The system whose limit cycle is sought.
    - `x0`: The initial condition, shape ``(dim,)``. For the stable cycle it must lie
        in the basin of attraction, which ``ode.in_basin`` checks.
    - `t_settle`: The integration time for `x0` to settle near the cycle, the
        duration of the transient; in reverse time, the backward integration time.
        Insufficient settling is caught by the iterates of the one-sided map
        (`initial_guess`).
    - `m`: The number of shooting segments; 1 is single shooting.
    - `reverse_time`: Whether to search in the time-reversed flow, for a repelling
        cycle. The ``in_basin`` check is then skipped, since the forward basin is
        irrelevant to a repelling cycle.
    - `config`: The solver settings for every stage, recorded on the result for the
        later analyses.

    **Other arguments:**

    - `t_max`: The bound on one return time of the map; ``10 t_settle`` when
        ``None``.
    - `max_returns`: The bound on the number of map iterations in the fixed-point
        search.
    - `settle_tol`: The tolerance of that search. It need not be stringent, since
        shooting polishes the result; a weakly attracting cycle needs it loosened,
        because its return map contracts slowly.
    - `tol`: The Levenberg–Marquardt tolerance of the shooting, ``rtol = tol`` and
        ``atol = 1e-2 tol``: the accuracy of the result.

    **Returns:**

    A `LimitCycle`: the converged mesh `x0s`, the `period`, the shooting `residual`, the
    `settle` record, the `section`, and the `ode`, `config` and `reverse_time` the orbit
    was found with.

    **Raises:**

    `ValueError` for an `x0` that is not a vector or lies outside ``ode.in_basin`` in
    forward time, or for a non-positive `t_settle`; `NotConvergedError`, naming the
    stage, when the settled point leaves the finite domain, the map does not return,
    shooting does not converge, or a guard rejects the solution.
    """
    x0 = jnp.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a vector.")
    if not reverse_time and not bool(ode.in_basin(x0)):
        raise ValueError("x0 is outside the basin of attraction of the limit cycle.")
    if not t_settle > 0:
        raise ValueError("t_settle must be positive.")
    t_max = 10.0 * t_settle if t_max is None else t_max

    # Floats are static under filter_jit, and a new value is a new program; the times
    # go in as arrays so that a new t_settle or period estimate does not recompile.
    p = _settle(
        ode, x0, jnp.asarray(t_settle, dtype=float), config=config,
        reverse_time=reverse_time,
    )  # fmt: skip
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
    period_hat = jnp.asarray(fp.tau, dtype=float)
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
    _check_shooting(ode, shot, float(period_hat))
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


def nearest_cycle_time(
    cycle: LimitCycle,
    y: Float[ArrayLike, " dim"],
    *,
    n_samples: int = 1024,
    newton_steps: int = 8,
) -> tuple[Float[Array, ""], Float[Array, " dim"], Bool[Array, ""]]:
    r"""The time $t^\ast$ from ``cycle.x0s[0]`` of the cycle point nearest `y`, that
    point, and whether the solve converged.

    The nearest of `n_samples` samples is $x_j$ at time $t_j$; an Optimistix Newton
    root solve finds the offset $\delta$ from the *previous* sample $x_{j-1}$ at which
    the stationarity condition $g(\delta) = (\gamma - y)\cdot f(\gamma)$ vanishes,
    $\gamma = \varphi_\delta(x_{j-1})$, differentiating through the flow, from the
    seed $\delta = T/$`n_samples`, the spacing. The solve starts one spacing back
    because diffrax's derivative of a flow in its end time, $f(\gamma)$ elsewhere, is
    identically zero for a solve of zero length, so a Newton seeded at the nearest
    sample itself would never move. The answer is within one spacing of the seed, so
    the solve converges quadratically; it runs at most `newton_steps` steps and never
    raises. Jit- and vmap-compatible; a caller that needs convergence checks the
    returned flag (`reanchor` does).

    **Arguments:**

    - `cycle`: The orbit.
    - `y`: The point, shape ``(dim,)``, on or near the orbit.
    - `n_samples`: The number of samples for the starting guess.
    - `newton_steps`: The bound on the solver's steps.

    **Returns:**

    ``(t, point, converged)``: $t^\ast$ in $[0, T)$, $\gamma(t^\ast)$, and whether the
    Newton solve reached `TIME_TOL`.
    """
    ode, config, period = cycle.ode, cycle.config, cycle.period
    y = jnp.asarray(y, dtype=float)
    times, points = cycle.sample(n_samples)
    j = jnp.argmin(jnp.linalg.norm(points - y, axis=1))
    spacing = period / n_samples
    t_prev, x_prev = times[j] - spacing, points[j - 1]  # j - 1 wraps around at 0

    def stationarity(delta, args):
        del args
        x = ode.flow_to(delta, x_prev, config=config)
        return jnp.dot(x - y, ode.rhs(0.0, x))

    sol = optx.root_find(
        stationarity, optx.Newton(rtol=0.0, atol=TIME_TOL), spacing,
        max_steps=newton_steps, throw=False,
    )  # fmt: skip
    delta = sol.value
    x = ode.flow_to(delta, x_prev, config=config)
    converged = sol.result == optx.RESULTS.successful
    return jnp.mod(t_prev + delta, period), x, converged


def reanchor(
    cycle: LimitCycle,
    point: Float[ArrayLike, " dim"],
    *,
    n_points: int = 1024,
    newton_steps: int = 8,
) -> LimitCycle:
    r"""The same orbit with its phase origin ``x0s[0]`` moved to the orbit point nearest
    `point`.

    The nearest orbit point comes from `nearest_cycle_time`, which must report
    convergence. The orbit is already converged, so it is not
    solved again: the new mesh is the flow from the new origin at times $iT/m$, and the
    residual is the shooting residual evaluated once at the new mesh (it is the old
    orbit's, moved). The section is the one through the new origin, normal to the flow
    there, as `find_limit_cycle` builds through its own origin. Phase 0 is where the
    isochron pipeline anchors its junction (`docs/design/isochrons.md` §6).

    **Arguments:**

    - `cycle`: The orbit, from `find_limit_cycle`.
    - `point`: Where phase 0 should be, shape ``(dim,)``; on or near the orbit.
    - `n_points`: The number of samples for the starting guess.
    - `newton_steps`: The bound on the Newton steps in time.

    **Returns:**

    A `LimitCycle` with the same `ode`, `config`, `reverse_time`, `settle` and
    `period`, and new `x0s`, `residual` and `section`.

    **Raises:**

    `ValueError` for a `point` of the wrong shape; `NotConvergedError` when the Newton
    solve does not settle within `newton_steps`.
    """
    ode = cycle.ode
    p = jnp.asarray(point, dtype=float)
    if p.shape != (cycle.dim,):
        raise ValueError(f"point must have shape ({cycle.dim},), not {p.shape}.")
    _, x, converged = nearest_cycle_time(
        cycle, p, n_samples=n_points, newton_steps=newton_steps
    )
    if not bool(converged):
        raise NotConvergedError(
            "reanchor",
            f"the nearest orbit point did not settle in {newton_steps} Newton steps.",
            x,
        )
    section = CrossSection.normal_to_flow(ode, x)
    x0s, residual = _moved_mesh(cycle, x, section)
    return copy.replace(cycle, x0s=x0s, residual=residual, section=section)


@eqx.filter_jit
def _moved_mesh(
    cycle: LimitCycle, origin: Float[Array, " dim"], section: CrossSection
) -> tuple[Float[Array, "m dim"], Float[Array, ""]]:
    """The cycle's mesh from `origin` at times ``i T / m``, and the shooting residual
    at that mesh with the phase condition on `section`."""
    sign = -1.0 if cycle.reverse_time else 1.0
    times = sign * jnp.linspace(0.0, cycle.period, cycle.m, endpoint=False)
    x0s = cycle.ode.flow(times, origin, config=cycle.config).ys
    x0s = cast(Array, x0s)
    args = ShootingArgs(cycle.ode, section, cycle.config, cycle.reverse_time)
    matching, phase = shooting_residual((x0s, cycle.period), args)
    return x0s, jnp.sqrt(jnp.sum(matching**2) + phase**2)


__all__ = [
    "LimitCycle",
    "find_limit_cycle",
    "initial_guess",
    "is_equilibrium",
    "nearest_cycle_time",
    "period_consistent",
    "polish",
    "reanchor",
    "settle",
]
