r"""The Poincaré (first-return) map of a flow on a cross-section.

``PoincareMap`` turns the continuous flow of an ``AbstractODE`` and a ``Hyperplane``
$\Sigma$ into a discrete map: $P(x)$ is the point where the trajectory through $x$ next
crosses $\Sigma$ in the selected direction, together with the time $\tau(x)$ it takes.
Parker & Chua (1989) call the one-directional version the *one-sided* map $P_+$ (§2.4,
§5.5) and build their Poincaré-map shooting method on it; the settle stage of
``limit_cycle`` is the iteration of $P_+$ to its fixed point (their brute force, §5.1.1,
read on the section), and a cycle that cuts $\Sigma$ in $2K$ points is a period-$K$
orbit of $P_+$ (their §5.4), which is how the minimum-period test is expressed here.

**Direction.** ``"positive"`` counts crossings with $\langle n, v\rangle > 0$ for the
field $v$ being integrated — Parker & Chua's positive direction (footnote 4 of §5.5),
with $v = -f$ in reverse time so that a point of a cycle on the section returns to
itself after one period in either time direction; ``"negative"`` the opposite ones,
``"both"`` every crossing. The one-sided maps are the ones with fixed points; $P_-$ on
$(p, n)$ is $P_+$ on $(p, -n)$.

**Reverse time** integrates the time-reversed field, so attracting and repelling
cycles swap roles: the first-return map of the reversed flow finds a *repelling* cycle
by iteration (Parker & Chua §5.1.1, footnote 1: brute force "can locate an unstable
limit set of a system that can run in reverse time").

**Crossing location.** diffrax's ``Event`` with a real-valued condition and a Newton
root finder (Shampine & Thompson 2000; the classical remedy for the error at the final
step is Hénon 1982, see Tucker 2002). Starting *on* the section would trigger the event
at $t = 0$ (diffrax reads the zero as a sign change), so the map first integrates for
a short ``departure`` time (a fraction of ``t_max``) without the event, and detects
crossings from there; a crossing inside the departure window is missed, which only
matters for a starting point just short of the section. The guard is deliberately a
separate solve rather than a time test inside the condition: in a reverse-time solve
diffrax 0.7.2 passes the condition its *internal* (increasing) time, not the signed
one, so a condition that reads ``t`` would be wrong there.

**Derivative.** $DP(x)$ is obtained *without* differentiating through the event time:
with $y = P(x)$ and $\tau = \tau(x)$ already known,

$$
DP(x) = \Bigl[I - \frac{f(y)\, n^{\top}}{\langle n, f(y)\rangle}\Bigr]
\partial_x\Phi_\tau(x),
$$

Parker & Chua Eq. (5.21): the fixed-time flow derivative projected onto $\Sigma$ along
the flow. Its eigenvalues are $0$ (along the flow) and the nontrivial Floquet
multipliers of a cycle through $x$ (their §5.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Literal

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


Direction = Literal["positive", "negative", "both"]
_EVENT_DIRECTION: dict[str, bool | None] = {
    "positive": True,
    "negative": False,
    "both": None,
}


@dataclass(frozen=True)
class FixedPoint:
    """The settled state of ``PoincareMap.fixed_point``.

    - ``x``: the last iterate, on the section; ``tau``: the time of one full return
      to it (the sum of the last ``k`` map times); ``k``: the minimal period of the
      map at the fixed point (1 for a cycle cutting the section twice).
    - ``taus (n,)``, ``iterates (n+1, dim)``: the whole iteration, ``iterates[0]`` the
      starting point.
    - ``contraction``: ratio of the last two distances ``|x_i - x_{i-k}|`` — the
      brute-force estimate of the nontrivial Floquet multiplier (``nan`` if fewer than
      ``2k`` steps were taken); ``converged``: whether the tolerance was met.
    """

    x: Float[np.ndarray, " dim"]
    tau: float
    k: int
    taus: Float[np.ndarray, " n"]
    iterates: Float[np.ndarray, "n+1 dim"]
    contraction: float
    converged: bool


class PoincareMap(eqx.Module):
    """First-return map of ``ode``'s flow to ``section``; see the module docstring.

    ``t_max`` bounds the integration time of one return (``NotConvergedError`` if the
    trajectory has not crossed by then); ``departure`` (fraction of ``t_max``) is the
    guard after the start during which crossings are ignored; ``config`` sets the
    solver, and its ``rtol``/``atol`` also set the crossing root finder's.
    """

    ode: AbstractODE
    section: Hyperplane
    config: SolverConfig
    t_max: float = eqx.field(static=True)
    direction: Direction = eqx.field(static=True, default="positive")
    reverse_time: bool = eqx.field(static=True, default=False)
    departure: float = eqx.field(static=True, default=1e-3)

    def __check_init__(self):
        if self.direction not in _EVENT_DIRECTION:
            raise ValueError(f"direction must be one of {list(_EVENT_DIRECTION)}.")
        if not self.t_max > 0:
            raise ValueError("t_max must be positive.")
        if not 0 < self.departure < 1:
            raise ValueError("departure is a fraction of t_max in (0, 1).")

    # ----------------------------------------------------------------- the map ----
    @property
    def time_sign(self) -> float:
        return -1.0 if self.reverse_time else 1.0

    def __call__(
        self, x: Float[ArrayLike, " dim"]
    ) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
        """``(P(x), tau(x))``: the next crossing and the (positive) time to reach it."""
        x = jnp.asarray(x, dtype=float)
        sol = _return_solve(self, x)
        ys, ts = cast(Array, sol.ys), cast(Array, sol.ts)
        if not bool(sol.event_mask):
            raise NotConvergedError(
                "return",
                f"no {self.direction} crossing of the section within t_max = "
                f"{self.t_max}; the point may be outside the basin, or converging "
                "to an equilibrium.",
                last=np.asarray(ys[0]),
            )
        return ys[0], jnp.abs(ts[0])

    def iterate(self, x: Float[ArrayLike, " dim"], k: int) -> tuple[Array, Array]:
        """``k`` applications of the map: the iterates ``(k+1, dim)`` (starting with
        ``x``) and the return times ``(k,)``."""
        xs = [jnp.asarray(x, dtype=float)]
        taus = []
        for _ in range(k):
            xn, tau = self(xs[-1])
            xs.append(xn)
            taus.append(tau)
        return jnp.stack(xs), jnp.stack(taus) if taus else jnp.zeros((0,))

    def fixed_point(
        self, x: Float[ArrayLike, " dim"], *, tol: float = 1e-8, max_iter: int = 100
    ) -> FixedPoint:
        """Iterate the one-sided map until it returns to within ``tol`` of an earlier
        iterate — a period-``k`` orbit of the map, ``k`` minimal (Parker & Chua §5.4,
        the minimum-period test), ``k = 1`` for a cycle that cuts the section twice.
        Raises ``NotConvergedError`` (with the iteration so far in ``last``) after
        ``max_iter`` steps. Not defined for ``direction="both"``: consecutive crossings
        of a two-sided map alternate sides, so a cycle is never its fixed point."""
        if self.direction == "both":
            raise ValueError("fixed_point needs a one-sided map; use iterate.")
        xs = [np.asarray(jnp.asarray(x, dtype=float))]
        taus: list[float] = []
        for i in range(1, max_iter + 1):
            xn, tau = self(xs[-1])
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
                    contraction = d / prev if prev > 0 else float("nan")
                    return FixedPoint(
                        x=xs[i],
                        tau=float(np.sum(taus[i - k : i])),
                        k=k,
                        iterates=np.stack(xs),
                        taus=np.asarray(taus),
                        contraction=contraction,
                        converged=True,
                    )
        raise NotConvergedError(
            "settle",
            f"no return to within {tol} of an earlier crossing in {max_iter} "
            "iterations of the map; the transient is slower than max_iter allows, or "
            "the orbit is not periodic.",
            last=FixedPoint(
                x=xs[-1],
                tau=float("nan"),
                k=0,
                iterates=np.stack(xs),
                taus=np.asarray(taus),
                contraction=float("nan"),
                converged=False,
            ),
        )

    # -------------------------------------------------------------- derivative ----
    def jacobian(self, x: Float[ArrayLike, " dim"]) -> Float[Array, "dim dim"]:
        r"""$DP(x)$ by Parker & Chua Eq. (5.21): the fixed-time flow derivative over
        $\tau(x)$ (forward-mode AD through the solver), projected onto the section
        along the flow at $P(x)$. One eigenvalue is $0$; the others are the
        multipliers of the return map."""
        x = jnp.asarray(x, dtype=float)
        y, tau = self(x)
        J = jax.jacfwd(lambda z: _flow_to(self, z, tau))(x)
        f = self.ode.rhs(0.0, y)
        n = self.section.normal
        proj = jnp.eye(x.shape[0]) - jnp.outer(f, n) / jnp.dot(n, f)
        return proj @ J


# ------------------------------------------------------------------ the solves ----
@eqx.filter_jit
def _return_solve(pmap: PoincareMap, x: Float[Array, " dim"]) -> dfx.Solution:
    """One event-terminated solve from ``x`` to the next counted crossing, after the
    departure window; ``ts`` is the crossing time measured from ``x``."""
    sign_t = pmap.time_sign
    delta = pmap.departure * pmap.t_max
    x_dep = _flow_to(pmap, x, jnp.asarray(delta))

    def cond(t, y, args, **kwargs):
        del t, args, kwargs
        return pmap.section(y)

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
        event=dfx.Event(cond, root_finder, direction=_event_direction(pmap)),
        throw=pmap.config.throw,
    )
    return sol


def _event_direction(pmap: PoincareMap) -> bool | None:
    """diffrax's ``direction`` (up / down / either crossing of the signed distance)
    for the map's direction, relative to the field being integrated: in reverse time the
    signed distance decreases along a positive crossing of the reversed field."""
    d = _EVENT_DIRECTION[pmap.direction]
    return d if d is None or not pmap.reverse_time else not d


def _flow_to(pmap: PoincareMap, x: Float[Array, " dim"], tau) -> Float[Array, " dim"]:
    """``Φ_{±tau}(x)`` with a forward-mode-differentiable adjoint."""
    config = pmap.config.with_adjoint(dfx.ForwardMode())
    sol = dfx.diffeqsolve(
        dfx.ODETerm(pmap.ode.rhs),
        config.solver,
        t0=0.0,
        t1=pmap.time_sign * tau,
        dt0=None,
        y0=x,
        stepsize_controller=config.controller(),
        max_steps=config.max_steps,
        adjoint=config.adjoint,
        throw=config.throw,
    )
    return sol.ys[0]


__all__ = ["Direction", "FixedPoint", "PoincareMap"]
