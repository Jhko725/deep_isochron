r"""The asymptotic phase on the basin of an attracting cycle, by forward integration and
a read-off on the cycle (roadmap O3; route (a) of ``docs/design/phase-field.md`` §3.1).

The asymptotic phase $\Theta(x)$ of a point $x$ in the basin of an attracting cycle
$\gamma$ of period $T$ is the phase of the cycle point whose orbit the orbit of $x$
converges to (Guckenheimer 1975; Langfield et al. 2014, §I.B). It satisfies
$\Theta(\Phi_t x) = \Theta(x) + t/T \pmod 1$ (their Eq. 2, with the phase in turns), so
$\Theta(x) = \Theta(\Phi_{t_{\rm end}} x) - t_{\rm end}/T$ for every $t_{\rm end}$.
After a long enough $t_{\rm end}$ the end point $y$ lies within about $\mu^n d_0$ of
the cycle, with $\mu$ the nontrivial Floquet multiplier and $n$ the number of periods,
and its phase is read off the cycle as $t^\ast/T$, where $\gamma(t^\ast)$ is the cycle
point nearest $y$ (`nearest_cycle_time`). The read-off errs at first order in the
distance from the cycle, because an isochron crosses the cycle along the stable
Floquet direction and not along the normal (Langfield et al. 2014, §I.B); the error is
negligible once that distance is near rounding, which is what $t_{\rm end}$ has to buy.

Phase 0 is at ``cycle.x0s[0]``, where `reanchor` puts it; the phase is a fraction of
the period in $[0, 1)$ and increases along the flow (ADR-0020), as in
``model/base.py``. `asymptotic_phase` reads the phase off at $t_{\rm end}$ and again
one period later and reports the gap between the two readings and the distance from
the cycle, so that a point whose orbit has not converged is flagged rather than given
a wrong phase. `phase_grid` evaluates it on a grid in chunks and masks the flagged
points.
"""

from __future__ import annotations

import copy
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from ...utils.numerics import phase_difference, wrap_phase
from .limit_cycle import LimitCycle, nearest_cycle_time


class AsymptoticPhase(eqx.Module):
    r"""The asymptotic phase of one or more points, with the two diagnostics of its
    read-off.

    **Attributes:**

    - `phase`: The phase in turns, in $[0, 1)$ with 0 at ``cycle.x0s[0]`` and
        increasing with time, read off at $t_{\rm end} + T$; ``nan``
        where the solve failed.
    - `distance`: The distance from the orbit's point at $t_{\rm end} + T$ to the
        nearest cycle point.
    - `discrepancy`: The unsigned shortest arc, in turns, between the readings at
        $t_{\rm end}$ and $t_{\rm end} + T$; it vanishes as the orbit converges to the
        cycle.
    """

    phase: Float[Array, "*shape"]
    distance: Float[Array, "*shape"]
    discrepancy: Float[Array, "*shape"]


def asymptotic_phase(
    cycle: LimitCycle,
    x: Float[ArrayLike, " dim"],
    *,
    t_end: Float[ArrayLike, ""],
    n_samples: int = 1024,
    newton_steps: int = 4,
) -> AsymptoticPhase:
    r"""The asymptotic phase of `x`, read off the cycle after flowing for `t_end` and
    again one period later.

    The orbit is integrated with ``cycle.config`` but with ``throw=False``, so that a
    failed solve gives ``nan`` for this point instead of an error for a whole batch.
    Jit- and vmap-compatible.

    **Arguments:**

    - `cycle`: An attracting orbit, with phase 0 at ``cycle.x0s[0]``.
    - `x`: The point, shape ``(dim,)``, in the basin.
    - `t_end`: The time before the first reading. It must cover the escape from the
        slowest part of the basin and enough periods $n$ for $\mu^n$ to reach rounding;
        the isochron configs give it per system, with the reason.
    - `n_samples`, `newton_steps`: As for `nearest_cycle_time`.

    **Returns:**

    An `AsymptoticPhase` with scalar fields.

    **Raises:**

    `ValueError` for a cycle found in reverse time, which is repelling; its phase
    field belongs to the reversed flow.
    """
    if cycle.reverse_time:
        raise ValueError(
            "asymptotic_phase needs an attracting cycle; this one was found in reverse "
            "time."
        )
    cycle = copy.replace(cycle, config=copy.replace(cycle.config, throw=False))
    period = cycle.period
    t_reads = jnp.stack([jnp.asarray(t_end, dtype=float), t_end + period])
    ts = jnp.concatenate([jnp.zeros((1,)), t_reads])
    solution = cycle.ode.flow(ts, jnp.asarray(x, dtype=float), config=cycle.config)
    # type-level only: ys is None only for a solve that saves nothing
    ends = cast(Array, solution.ys)[1:]
    finite = jnp.all(jnp.isfinite(ends))
    ends = jnp.where(finite, ends, cycle.x0s[0])  # keep a failed solve out of Newton

    def read(y, t_read):
        t, point, _ = nearest_cycle_time(
            cycle, y, n_samples=n_samples, newton_steps=newton_steps
        )
        phase = wrap_phase(jnp.mod(t - t_read, period) / period)
        return phase, jnp.linalg.norm(y - point)

    phases, distances = jax.vmap(read)(ends, t_reads)
    nan = jnp.asarray(jnp.nan, dtype=phases.dtype)
    return AsymptoticPhase(
        phase=jnp.where(finite, phases[1], nan),
        distance=jnp.where(finite, distances[1], nan),
        discrepancy=jnp.where(
            finite, jnp.abs(phase_difference(phases[1], phases[0])), nan
        ),
    )


def phase_grid(
    cycle: LimitCycle,
    xs: Float[ArrayLike, " nx"],
    ys: Float[ArrayLike, " ny"],
    *,
    t_end: float,
    batch_size: int = 4096,
    tol_phase: float = 1e-6,
    tol_distance: float = 1e-6,
    n_samples: int = 1024,
    newton_steps: int = 4,
) -> tuple[Float[Array, "ny nx"], Bool[Array, "ny nx"], AsymptoticPhase]:
    r"""The asymptotic phase on the grid ``xs`` $\times$ ``ys`` of a planar cycle,
    with the points whose read-off is not trusted masked.

    A host-side driver: it jits `asymptotic_phase` over chunks of `batch_size` points
    (``jax.lax.map``), so the memory of one chunk bounds the memory of the grid.

    **Arguments:**

    - `cycle`: An attracting planar orbit, with phase 0 at ``cycle.x0s[0]``.
    - `xs`, `ys`: The grid coordinates, shapes ``(nx,)`` and ``(ny,)``.
    - `t_end`: As for `asymptotic_phase`.
    - `batch_size`: The number of points per chunk.
    - `tol_phase`: A point is kept when its two readings differ by less than this.
    - `tol_distance`: A point is kept when its orbit ends closer to the cycle than this.
    - `n_samples`, `newton_steps`: As for `nearest_cycle_time`.

    **Returns:**

    ``(field, mask, result)``: the phase with ``nan`` where masked, shape
    ``(ny, nx)`` with ``field[i, j]`` at ``(xs[j], ys[i])``; the mask of the kept
    points; and the `AsymptoticPhase` of every point, masked or not.

    **Raises:**

    `RuntimeError` when float64 is off, since the read-off needs the cycle to rounding;
    `ValueError` for a cycle that is not planar or was found in reverse time.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "phase_grid needs float64: set jax.config.update('jax_enable_x64', True)."
        )
    if cycle.dim != 2:
        raise ValueError(
            f"phase_grid needs a planar cycle; this one has dim {cycle.dim}."
        )
    grid_x, grid_y = jnp.meshgrid(jnp.asarray(xs, float), jnp.asarray(ys, float))
    points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    @eqx.filter_jit
    def evaluate(cycle, points, t_end):
        def phase_of(x):
            return asymptotic_phase(
                cycle, x, t_end=t_end, n_samples=n_samples, newton_steps=newton_steps
            )

        return jax.lax.map(phase_of, points, batch_size=batch_size)

    result = evaluate(cycle, points, jnp.asarray(t_end, dtype=float))
    result = jax.tree.map(lambda a: a.reshape(grid_x.shape), result)
    mask = (result.discrepancy < tol_phase) & (result.distance < tol_distance)
    field = jnp.where(mask, result.phase, jnp.nan)
    return field, mask, result


__all__ = ["AsymptoticPhase", "asymptotic_phase", "phase_grid"]
