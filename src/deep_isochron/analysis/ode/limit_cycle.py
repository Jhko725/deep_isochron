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

The stages and the map are jit- and vmap-compatible and left unjitted; the host-side
driver ``find_limit_cycle`` jits the stages it calls and raises informative errors. A
``LimitCycle`` is a fixed-shape pytree, so functions of a cycle can be jitted and
vmapped as they stand. ADR-0014 has the decisions; ``docs/design/analysis.md`` §2 the
survey of shooting methods.
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
    t_settle: float,
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
    period_hat: float,
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
_polish = eqx.filter_jit(polish)  # the stages as the host-side driver runs them


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
