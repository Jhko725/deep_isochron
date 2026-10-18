r"""``limit_cycle``: the attracting (or, in reverse time, repelling) periodic orbit of a
planar ODE and its period, by settle → first-return map → multiple shooting.

1. **Settle** (Parker & Chua 1989, §5.1.1, brute force): integrate from ``x0`` for
   ``t_settle``; the end point $p$ is near the cycle.
2. **Section and return map**: $\Sigma$ through $p$ normal to the flow
   (``Hyperplane.through``); the one-sided map $P_+$ on it (``PoincareMap``) is
   iterated to its fixed point (their minimum-period test, §5.5), giving a point
   $\hat x \in \Sigma$ on the cycle to ``tol``, the return time $\hat T$ (over the
   $k$ map steps of a period-$k$ orbit), and the brute-force multiplier estimate.
3. **Polish** (``multiple_shooting``): $m$ points sampled from the orbit through
   $\hat x$ at times $i\hat T/m$ and $\hat T$ are refined by Levenberg–Marquardt on
   the shooting equations with AD Jacobians, coarsening to $m = 1$.
4. **Guards** (Parker & Chua §5.5, "Extraneous solutions"): the shooting equations are
   also solved by an equilibrium and by $T \to 0$; both are rejected.
5. **Diagnostics**: the residual norm, and the monodromy $M = \partial_x\Phi_T(x^\ast)$,
   whose eigenvalue nearest $1$ measures the accuracy of the orbit (Guckenheimer &
   Meloon 2000, §5.4) — the other eigenvalue is the nontrivial Floquet multiplier,
   whose use is the business of the Floquet item (roadmap O2). In reverse time the
   monodromy of the reversed flow is inverted, so ``monodromy`` is always the
   forward-time one.
6. **Output**: the orbit sampled at ``n_points`` uniform times in $[0, T)$ (a
   phase-uniform parametrization), the period, the orientation (sign of the signed
   area, $+1$ counterclockwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from ..errors import NotConvergedError
from .poincare import FixedPoint, PoincareMap
from .sections import Hyperplane
from .shooting import multiple_shooting, segment_flow, SHOOTING_SOLVER_CONFIG


@dataclass(frozen=True)
class LimitCycle:
    r"""A periodic orbit located by ``limit_cycle`` (host-side NumPy).

    - ``x0 (dim,)``: a point of the orbit (on ``section``); ``period`` $T > 0$.
    - ``times (n,)``, ``points (n, dim)``: the orbit at ``n`` uniform times in
      $[0, T)$, ``points[0] == x0``; phase-uniform, so ``2π times / period`` is the
      asymptotic phase on the cycle up to a constant.
    - ``orientation``: $+1$ counterclockwise (positive signed area), $-1$ clockwise.
    - ``section``: the cross-section used; ``monodromy (dim, dim)``: forward-time
      $\partial_x\Phi_T(x_0)$; ``trivial_multiplier_error``:
      $\min_i\vert\lambda_i - 1\vert$ over its eigenvalues, the accuracy readout;
      ``residual``: the shooting residual norm; ``settle``: the fixed-point iteration
      that produced the initial guess; ``reverse_time``: whether the orbit was found
      in the time-reversed flow (then it is a *repelling* orbit of the flow).
    """

    x0: Float[np.ndarray, " dim"]
    period: float
    times: Float[np.ndarray, " n"]
    points: Float[np.ndarray, "n dim"]
    orientation: int
    section: Hyperplane
    monodromy: Float[np.ndarray, "dim dim"]
    trivial_multiplier_error: float
    residual: float
    settle: FixedPoint
    reverse_time: bool = False


def limit_cycle(
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

    Raises ``ValueError`` if ``x0`` is outside ``ode.in_basin`` (forward time) or at
    an equilibrium, ``NotConvergedError`` from any stage.
    """
    x0 = jnp.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a vector.")
    if not reverse_time and not bool(ode.in_basin(x0)):
        raise ValueError("x0 is outside the basin of attraction of the limit cycle.")
    if not t_settle > 0:
        raise ValueError("t_settle must be positive.")
    t_max = 10.0 * t_settle if t_max is None else t_max
    time_sign = -1.0 if reverse_time else 1.0

    # 1. settle
    p = segment_flow(ode, x0, jnp.asarray(t_settle), config, time_sign)
    if not bool(jnp.all(jnp.isfinite(p))):
        raise NotConvergedError("settle", "the trajectory left the finite domain.", p)

    # 2. section, return map, fixed point
    section = Hyperplane.through(ode, p)
    pmap = PoincareMap(ode, section, config, t_max=t_max, reverse_time=reverse_time)
    fp = pmap.fixed_point(p, tol=settle_tol, max_iter=max_returns)
    x_hat, T_hat = jnp.asarray(fp.x), fp.tau

    # 3. polish
    times = T_hat * jnp.arange(m) / m
    guess = (
        jnp.concatenate(
            [
                x_hat[None],
                jax.vmap(lambda t: segment_flow(ode, x_hat, t, config, time_sign))(
                    times[1:]
                ),
            ]
        )
        if m > 1
        else x_hat[None]
    )
    shot = multiple_shooting(
        ode,
        guess,
        T_hat,
        section,
        config=config,
        reverse_time=reverse_time,
        rtol=tol,
        atol=tol * 1e-2,
    )
    x_star = jnp.asarray(shot.points[0])
    T = shot.period

    # 4. guards
    f_star = ode.rhs(0.0, x_star)
    speed = float(jnp.linalg.norm(f_star))
    if speed <= 1e-6 * max(1.0, float(jnp.linalg.norm(x_star))):
        raise NotConvergedError(
            "shooting", "converged to an equilibrium (the vector field vanishes).", shot
        )
    if not (0.1 * T_hat <= T <= 10.0 * T_hat):
        raise NotConvergedError(
            "shooting",
            f"the period collapsed or exploded: T = {T:.6g} vs the return-map estimate "
            f"{T_hat:.6g}.",
            shot,
        )

    # 5. diagnostics
    fwd_config = config.with_adjoint(dfx.ForwardMode())
    J = jax.jacfwd(
        lambda z: segment_flow(ode, z, jnp.asarray(T), fwd_config, time_sign)
    )(x_star)
    monodromy = jnp.linalg.inv(J) if reverse_time else J
    eig = jnp.linalg.eigvals(monodromy)
    trivial_error = float(jnp.min(jnp.abs(eig - 1.0)))

    # 6. output
    ts = T * jnp.arange(n_points) / n_points
    pts = jnp.concatenate(
        [
            x_star[None],
            jax.vmap(lambda t: segment_flow(ode, x_star, t, config, time_sign))(ts[1:]),
        ]
    )
    # the orbit as traversed by the *forward* flow: reverse-time sampling runs the
    # loop backward, so flip it to keep the parametrization forward in time
    if reverse_time:
        pts = jnp.concatenate([pts[:1], pts[1:][::-1]])
    area = 0.5 * float(
        jnp.sum(
            pts[:, 0] * jnp.roll(pts[:, 1], -1) - jnp.roll(pts[:, 0], -1) * pts[:, 1]
        )
    )
    orientation = 1 if area > 0 else -1

    return LimitCycle(
        x0=np.asarray(x_star),
        period=float(T),
        times=np.asarray(ts),
        points=np.asarray(pts),
        orientation=orientation,
        section=section,
        monodromy=np.asarray(monodromy),
        trivial_multiplier_error=trivial_error,
        residual=shot.residual,
        settle=fp,
        reverse_time=reverse_time,
    )


__all__ = ["LimitCycle", "limit_cycle"]
