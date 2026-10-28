r"""The Poincaré (first-return) map of a flow on a cross-section.

``PoincareMap`` turns the flow of an ``AbstractODE`` and a ``CrossSection`` $\Sigma$
into a discrete map: $P(x)$ is the point where the trajectory through $x$ next crosses
$\Sigma$ in the selected direction, and $\tau(x)$ the time it takes. Parker & Chua
1989 call the one-directional version the one-sided map $P_+$ (§2.4, §5.5); the settle
stage of ``find_limit_cycle`` iterates $P_+$ to its fixed point, their brute force
(§5.1.1) read on the section, and a cycle that cuts $\Sigma$ in $2K$ points is a
period-$K$ orbit of $P_+$ (§5.4), which is how the minimum-period test is expressed.

The map has four settings. ``direction`` selects which crossings count:
``"positive"`` those with $\langle n, v\rangle > 0$ for the field $v$ being
integrated, ``"negative"`` the opposite ones, ``"both"`` every crossing; only the
one-sided maps have fixed points. ``reverse_time`` integrates the time-reversed field,
so the first-return map then finds a repelling cycle by iteration. ``departure`` is a
short window after the start in which crossings are ignored, because diffrax reads a
start on the section as a crossing at $t = 0$. ``throw`` chooses a runtime error or
``tau = inf`` when no crossing is found within ``t_max``.

The map is differentiable in $x$ through a custom JVP that never differentiates
through the event: the fixed-time flow derivative over the known return time,
projected onto $\Sigma$ along the flow (Parker & Chua Eq. 5.21). The eigenvalues of
$DP$ are $0$, along the flow, and the nontrivial Floquet multipliers of a cycle
through $x$ (their §5.3). ``__call__`` and ``iterate`` are jit- and vmap-compatible and
left unjitted; ``fixed_point`` is a host-side driver over the jitted return solve.

ADR-0014 has the decisions, including why the departure guard is a separate solve;
``docs/design/limit-cycle.md`` has the survey and the crossing-location literature.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from ..errors import NotConvergedError
from .sections import CrossSection


Direction = Literal["positive", "negative", "both"]
"""Which crossings of the section a map counts."""


class FixedPoint(eqx.Module):
    """A fixed point of a ``PoincareMap`` as settled by ``fixed_point``: a light,
    fixed-shape pytree. The iteration that produced it is returned separately, as a
    ``FixedPointIteration``.

    **Attributes:**

    - `x`: The last iterate, on the section.
    - `tau`: The time of one full return to it, the sum of the last `k` map times.
    - `k`: The minimal period of the map at the fixed point, 1 for a cycle cutting
        the section twice (static).
    - `contraction`: The ratio of the last two distances ``|x_i - x_{i-k}|``, the
        brute-force estimate of the nontrivial Floquet multiplier; ``nan`` if fewer
        than ``2k`` steps were taken.
    - `converged`: Whether the tolerance was met (static).
    """

    x: Float[Array, " dim"]
    tau: Float[Array, ""]
    k: int = eqx.field(static=True)
    contraction: Float[Array, ""]
    converged: bool = eqx.field(static=True)


class FixedPointIteration(NamedTuple):
    """The record of a ``fixed_point`` search, as host-side NumPy arrays.

    **Attributes:**

    - `taus`: The return times, shape ``(n,)``.
    - `iterates`: The iterates, shape ``(n+1, dim)``; ``iterates[0]`` is the starting
        point.
    """

    taus: Float[np.ndarray, " n"]
    iterates: Float[np.ndarray, "n+1 dim"]


class PoincareMap(eqx.Module):
    """The first-return map of the flow of `ode` to `section`.

    **Attributes:**

    - `ode`: The system.
    - `section`: The cross-section.
    - `config`: The solver settings; its `rtol` and `atol` also set the crossing root
        finder's.
    - `t_max`: The bound on the integration time of one return, an array leaf so that
        a new bound does not recompile the map (a float was static, and every
        `find_limit_cycle` with a new `t_settle` compiled the return map again).
    - `direction`: Which crossings count; ``"positive"`` by default (static).
    - `reverse_time`: Whether to integrate the time-reversed field (static).
    - `departure`: The window after the start in which crossings are ignored, as a
        fraction of `t_max` (static).
    - `throw`: Whether a missing crossing is a runtime error, as in diffrax, or
        reported as ``tau = inf`` (static).

    **Raises:**

    `ValueError` at construction for an unknown `direction`, a non-positive `t_max`, or
    a `departure` outside ``(0, 1)``.
    """

    ode: AbstractODE
    section: CrossSection
    config: SolverConfig
    t_max: Float[Array, ""] = eqx.field(converter=lambda t: jnp.asarray(t, dtype=float))
    direction: Direction = eqx.field(static=True, default="positive")
    reverse_time: bool = eqx.field(static=True, default=False)
    departure: float = eqx.field(static=True, default=1e-3)
    throw: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        """Validate `direction`, `t_max` and `departure`."""
        if self.direction not in ("positive", "negative", "both"):
            raise ValueError("direction must be 'positive', 'negative' or 'both'.")
        if not float(self.t_max) > 0:
            raise ValueError("t_max must be positive.")
        if not 0 < self.departure < 1:
            raise ValueError("departure is a fraction of t_max in (0, 1).")

    # ------------------------------------------------------------- properties ----
    @property
    def time_sign(self) -> float:
        """$-1$ in reverse time, $+1$ otherwise."""
        return -1.0 if self.reverse_time else 1.0

    @property
    def event_direction(self) -> bool | None:
        """diffrax's event ``direction`` for this map: up, down or either crossing of
        the signed distance. The map's direction is relative to the field being
        integrated, so in reverse time a positive crossing of the reversed field is a
        down-crossing of the signed distance."""
        if self.direction == "both":
            return None
        up = self.direction == "positive"
        return up if not self.reverse_time else not up

    # ----------------------------------------------------------------- the map ----
    def __call__(
        self, x: Float[ArrayLike, " dim"]
    ) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
        """The next counted crossing from `x` and the positive time to it.

        **Arguments:**

        - `x`: The starting point, shape ``(dim,)``.

        **Returns:**

        ``(P(x), tau(x))``, shapes ``(dim,)`` and ``()``; ``tau = inf`` when no crossing
        is found and `throw` is ``False``. Differentiable in `x`.
        """
        y, tau = _first_return(self, jnp.asarray(x, dtype=float))
        if self.throw:
            tau = eqx.error_if(
                tau,
                ~jnp.isfinite(tau),
                "PoincareMap: no counted crossing of the section within t_max; the "
                "point may be outside the basin, or converging to an equilibrium.",
            )
        return y, tau

    def iterate(self, x: Float[ArrayLike, " dim"], k: int) -> tuple[Array, Array]:
        """`k` applications of the map from `x`, by ``lax.scan``.

        **Returns:**

        ``(iterates, taus)`` with shapes ``(k+1, dim)`` and ``(k,)``; ``iterates[0]`` is
        `x`.
        """
        x = jnp.asarray(x, dtype=float)

        def step(carry, _):
            y, tau = self(carry)
            return y, (y, tau)

        _, (ys, taus) = jax.lax.scan(step, x, None, length=k)
        return jnp.concatenate([x[None], ys], axis=0), taus

    def fixed_point(
        self, x: Float[ArrayLike, " dim"], *, tol: float = 1e-8, max_iter: int = 100
    ) -> tuple[FixedPoint, FixedPointIteration]:
        """Iterate the one-sided map from `x` until it returns to within `tol` of an
        earlier iterate: a period-`k` orbit of the map with `k` minimal, the
        minimum-period test of Parker & Chua 1989, §5.4, where ``k = 1`` for a cycle
        that cuts the section twice. A host-side driver over the jitted return solve.

        **Arguments:**

        - `x`: The starting point, on the section.
        - `tol`: The distance to an earlier iterate that counts as a return.
        - `max_iter`: The bound on the number of map iterations.

        **Returns:**

        ``(fixed_point, iteration)``, the `FixedPoint` and the `FixedPointIteration`
        that produced it.

        **Raises:**

        `ValueError` for ``direction="both"``, since consecutive crossings of a
        two-sided map alternate sides and a cycle is never its fixed point;
        `NotConvergedError` after `max_iter` steps or when a return is missing, with the
        unfinished ``(fixed_point, iteration)`` in ``last``.
        """
        if self.direction == "both":
            raise ValueError("fixed_point needs a one-sided map; use iterate.")
        xs = [np.asarray(jnp.asarray(x, dtype=float))]
        taus: list[float] = []

        def unfinished() -> tuple[FixedPoint, FixedPointIteration]:
            """The iteration so far, for ``NotConvergedError.last``."""
            fp = FixedPoint(
                x=jnp.asarray(xs[-1]),
                tau=jnp.asarray(float("nan")),
                k=0,
                contraction=jnp.asarray(float("nan")),
                converged=False,
            )
            return fp, FixedPointIteration(np.asarray(taus), np.stack(xs))

        for i in range(1, max_iter + 1):
            xn, tau = _first_return_jit(self, jnp.asarray(xs[-1]))
            if not bool(jnp.isfinite(tau)):
                raise NotConvergedError(
                    "return",
                    f"no {self.direction} crossing of the section within t_max = "
                    f"{self.t_max} from iterate {i - 1}; the point may be outside the "
                    "basin, or converging to an equilibrium.",
                    last=unfinished(),
                )
            xs.append(np.asarray(xn))
            taus.append(float(tau))
            for k in range(1, i + 1):
                d = float(np.linalg.norm(xs[i] - xs[i - k]))
                if d < tol:
                    prev = (
                        float(np.linalg.norm(xs[i - k] - xs[i - 2 * k]))
                        if i - 2 * k >= 0
                        else 0.0
                    )
                    fp = FixedPoint(
                        x=jnp.asarray(xs[i]),
                        tau=jnp.asarray(float(np.sum(taus[i - k : i]))),
                        k=k,
                        contraction=jnp.asarray(d / prev if prev > 0 else float("nan")),
                        converged=True,
                    )
                    return fp, FixedPointIteration(np.asarray(taus), np.stack(xs))
        raise NotConvergedError(
            "settle",
            f"no return to within {tol} of an earlier crossing in {max_iter} "
            "iterations of the map; the transient is slower than max_iter allows, or "
            "the orbit is not periodic.",
            last=unfinished(),
        )


# ------------------------------------------------------- the differentiable core --
@eqx.filter_custom_jvp
def _first_return(
    pmap: PoincareMap, x: Float[Array, " dim"]
) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
    """``(P(x), tau(x))``, with ``tau = inf`` when no counted crossing occurs within
    ``t_max``: a departure solve without the event, then an event-terminated solve
    with a Newton root finder at the configuration's tolerances."""
    sign_t = pmap.time_sign
    delta = pmap.departure * pmap.t_max
    x_dep = pmap.ode.flow_to(sign_t * delta, x, config=pmap.config)

    def cond(t, y, args, **kwargs):
        del t, args, kwargs
        return pmap.section.signed_distance(y)

    root_finder = optx.Newton(rtol=pmap.config.rtol, atol=pmap.config.atol)
    sol = dfx.diffeqsolve(
        dfx.ODETerm(pmap.ode.rhs),
        pmap.config.solver,
        t0=sign_t * delta,
        t1=sign_t * pmap.t_max,
        dt0=None,
        y0=x_dep,
        stepsize_controller=pmap.config.controller(),
        max_steps=pmap.config.max_steps,
        adjoint=pmap.config.adjoint,
        event=dfx.Event(cond, root_finder, direction=pmap.event_direction),
        throw=pmap.config.throw,
    )
    assert sol.ys is not None and sol.ts is not None
    tau = jnp.where(sol.event_mask, jnp.abs(sol.ts[0]), jnp.inf)
    return sol.ys[0], tau


@_first_return.def_jvp
def _first_return_jvp(primals, tangents):
    r"""The derivative of the map in $x$ without differentiating through the event:
    with $y = P(x)$, $\tau$ known, $J = \partial_x\Phi_\tau(x)$ and $v$ the velocity at
    $y$, differentiating $\langle n, \Phi_\tau(x) - p\rangle = 0$ gives
    $d\tau = -\langle n, J\thinspace dx\rangle / \langle n, v\rangle$ and
    $dy = J\thinspace dx + v\thinspace d\tau$ (Parker & Chua 1989, Eq. 5.21).

    **Raises:**

    `NotImplementedError` for a tangent in the map itself: the map is differentiable in
    the point only, not in the ODE's parameters or the section.
    """
    pmap, x = primals
    pmap_dot, x_dot = tangents
    if pmap_dot is not None and any(
        leaf is not None for leaf in jax.tree_util.tree_leaves(pmap_dot)
    ):
        raise NotImplementedError(
            "PoincareMap is differentiable in the point x only, not in the ODE's "
            "parameters or the section."
        )
    y, tau = _first_return(pmap, x)
    x_dot = jnp.zeros_like(x) if x_dot is None else x_dot
    # J dx: the fixed-horizon flow derivative over the known return time
    _, j_dx = jax.jvp(
        lambda z: pmap.ode.flow_to(
            pmap.time_sign * tau, z, config=pmap.config.with_adjoint(dfx.ForwardMode())
        ),
        (x,),
        (x_dot,),
    )
    v = pmap.time_sign * pmap.ode.rhs(0.0, y)  # velocity at the crossing
    n = pmap.section.normal
    dtau = -jnp.dot(n, j_dx) / jnp.dot(n, v)
    dy = j_dx + v * dtau
    return (y, tau), (dy, dtau)


_first_return_jit = eqx.filter_jit(_first_return)  # what fixed_point drives


__all__ = ["Direction", "FixedPoint", "FixedPointIteration", "PoincareMap"]
