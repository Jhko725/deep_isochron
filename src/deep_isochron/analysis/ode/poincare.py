r"""The Poincaré (first-return) map of a flow on a cross-section.

``PoincareMap`` turns the continuous flow of an ``AbstractODE`` and a ``CrossSection``
$\Sigma$ into a discrete map: $P(x)$ is the point where the trajectory through $x$ next
crosses $\Sigma$ in the selected direction, together with the time $\tau(x)$ it takes.
Parker & Chua (1989) call the one-directional version the *one-sided* map $P_+$ (§2.4,
§5.5) and build their Poincaré-map shooting method on it; the settle stage of
``find_limit_cycle`` is the iteration of $P_+$ to its fixed point (their brute force,
§5.1.1, read on the section), and a cycle that cuts $\Sigma$ in $2K$ points is a
period-$K$ orbit of $P_+$ (their §5.4), which is how the minimum-period test is
expressed here.

**Direction.** ``"positive"`` counts crossings with $\langle n, v\rangle > 0$ for the
field $v$ being integrated — Parker & Chua's positive direction (§2.1.2; footnote 4 of
§5.5), with $v = -f$ in reverse time so that a point of a cycle on the section returns
to itself after one period in either time direction; ``"negative"`` the opposite ones,
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

**Derivative.** The map is a differentiable function of $x$ (``jax.jacfwd(P)(x)``,
``jax.jvp``) through a custom JVP that never differentiates through the event: with
$y = P(x)$ and $\tau$ known, differentiating the crossing condition
$\langle n, \Phi_\tau(x) - p\rangle = 0$ gives

$$
d\tau = -\frac{\langle n, J\,dx\rangle}{\langle n, v\rangle}, \qquad
dy = \Bigl[I - \frac{v\, n^{\top}}{\langle n, v\rangle}\Bigr] J\,dx,
$$

with $J = \partial_x\Phi_\tau(x)$ (forward-mode AD through the solver) and $v$ the
velocity at $y$ — Parker & Chua Eq. (5.21), the fixed-time flow derivative projected
onto $\Sigma$ along the flow. $DP$'s eigenvalues are $0$ (along the flow) and the
nontrivial Floquet multipliers of a cycle through $x$ (their §5.3).

**Jit.** ``__call__`` and ``iterate`` are jit- and vmap-compatible; a missing crossing
is a runtime error (``throw=True``, as in diffrax) or ``tau = inf`` (``throw=False``).
``fixed_point`` is a host-side driver over the jitted map (its state — a growing history
searched for the minimal period — buys nothing from compilation for a stage that runs
once per system), and raises ``NotConvergedError`` with the iteration so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class FixedPoint:
    """The settled state of ``PoincareMap.fixed_point`` (host-side NumPy).

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

    ``t_max`` bounds the integration time of one return; ``departure`` (fraction of
    ``t_max``) is the window after the start during which crossings are ignored;
    ``config`` sets the solver, and its ``rtol``/``atol`` also set the crossing root
    finder's; ``throw`` selects a runtime error (default) or ``tau = inf`` when no
    crossing is found within ``t_max``.
    """

    ode: AbstractODE
    section: CrossSection
    config: SolverConfig
    t_max: float = eqx.field(static=True)
    direction: Direction = eqx.field(static=True, default="positive")
    reverse_time: bool = eqx.field(static=True, default=False)
    departure: float = eqx.field(static=True, default=1e-3)
    throw: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        if self.direction not in ("positive", "negative", "both"):
            raise ValueError("direction must be 'positive', 'negative' or 'both'.")
        if not self.t_max > 0:
            raise ValueError("t_max must be positive.")
        if not 0 < self.departure < 1:
            raise ValueError("departure is a fraction of t_max in (0, 1).")

    # ------------------------------------------------------------- properties ----
    @property
    def time_sign(self) -> float:
        return -1.0 if self.reverse_time else 1.0

    @property
    def event_direction(self) -> bool | None:
        """diffrax's ``direction`` (up / down / either crossing of the signed
        distance) for this map: relative to the field being integrated, so in reverse
        time a positive crossing of the reversed field is a *down*-crossing of the
        signed distance."""
        if self.direction == "both":
            return None
        up = self.direction == "positive"
        return up if not self.reverse_time else not up

    @property
    def forward_config(self) -> SolverConfig:
        """``config`` with a forward-mode adjoint, for the derivative of the flow."""
        return self.config.with_adjoint(dfx.ForwardMode())

    # ----------------------------------------------------------------- the map ----
    def __call__(
        self, x: Float[ArrayLike, " dim"]
    ) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
        """``(P(x), tau(x))``: the next counted crossing and the positive time to it.
        Jit- and vmap-compatible; differentiable in ``x``."""
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
        """``k`` applications of the map by ``lax.scan``: the iterates ``(k+1, dim)``
        (starting with ``x``) and the return times ``(k,)``."""
        x = jnp.asarray(x, dtype=float)

        def step(carry, _):
            y, tau = self(carry)
            return y, (y, tau)

        _, (ys, taus) = jax.lax.scan(step, x, None, length=k)
        return jnp.concatenate([x[None], ys], axis=0), taus

    def fixed_point(
        self, x: Float[ArrayLike, " dim"], *, tol: float = 1e-8, max_iter: int = 100
    ) -> FixedPoint:
        """Iterate the one-sided map until it returns to within ``tol`` of an earlier
        iterate — a period-``k`` orbit of the map, ``k`` minimal (Parker & Chua §5.4,
        the minimum-period test), ``k = 1`` for a cycle that cuts the section twice.
        Host-side; raises ``NotConvergedError`` (with the iteration so far in ``last``)
        after ``max_iter`` steps or when a return is missing. Not defined for
        ``direction="both"``: consecutive crossings of a two-sided map alternate
        sides, so a cycle is never its fixed point."""
        if self.direction == "both":
            raise ValueError("fixed_point needs a one-sided map; use iterate.")
        xs = [np.asarray(jnp.asarray(x, dtype=float))]
        taus: list[float] = []

        def unfinished(converged: bool) -> FixedPoint:
            return FixedPoint(
                x=xs[-1],
                tau=float("nan"),
                k=0,
                taus=np.asarray(taus),
                iterates=np.stack(xs),
                contraction=float("nan"),
                converged=converged,
            )

        for i in range(1, max_iter + 1):
            xn, tau = _first_return(self, jnp.asarray(xs[-1]))
            if not bool(jnp.isfinite(tau)):
                raise NotConvergedError(
                    "return",
                    f"no {self.direction} crossing of the section within t_max = "
                    f"{self.t_max} from iterate {i - 1}; the point may be outside the "
                    "basin, or converging to an equilibrium.",
                    last=unfinished(False),
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
                    return FixedPoint(
                        x=xs[i],
                        tau=float(np.sum(taus[i - k : i])),
                        k=k,
                        taus=np.asarray(taus),
                        iterates=np.stack(xs),
                        contraction=d / prev if prev > 0 else float("nan"),
                        converged=True,
                    )
        raise NotConvergedError(
            "settle",
            f"no return to within {tol} of an earlier crossing in {max_iter} "
            "iterations of the map; the transient is slower than max_iter allows, or "
            "the orbit is not periodic.",
            last=unfinished(False),
        )


# ------------------------------------------------------- the differentiable core --
@eqx.filter_custom_jvp
def _first_return(
    pmap: PoincareMap, x: Float[Array, " dim"]
) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
    """``(P(x), tau(x))`` with ``tau = inf`` when no counted crossing occurs within
    ``t_max``: a departure solve, then an event-terminated solve."""
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
        lambda z: pmap.ode.flow_to(pmap.time_sign * tau, z, config=pmap.forward_config),
        (x,),
        (x_dot,),
    )
    v = pmap.time_sign * pmap.ode.rhs(0.0, y)  # velocity at the crossing
    n = pmap.section.normal
    dtau = -jnp.dot(n, j_dx) / jnp.dot(n, v)
    dy = j_dx + v * dtau
    return (y, tau), (dy, dtau)


__all__ = ["Direction", "FixedPoint", "PoincareMap"]
