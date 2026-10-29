r"""Analyses of a trained model against a reference cycle.

A model's phase has its own origin and may run either way round the cycle, so before a
learned phase field, isochron or metric is compared with a reference, the two phases
have to be aligned: ``phase_offset(model, cycle)`` gives the model's phase ``offset``
where its learned cycle crosses the reference cycle's section in the direction of the
flow, and the ``sense`` in which the model's phase advances in time, so that the
reference phase $\vartheta$ corresponds to the model phase
$\mathrm{sense}\cdot\vartheta + \mathrm{offset}$. The aligned error itself is
``training.evaluation.align_phases``, which seeds from this offset when a cycle is at
hand.

This module is the seam between ``model/``, which knows no reference, and
``analysis.ode``, which knows no model (ADR-0012).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from ..model.base import AbstractPhaseAmplitudeModel
from ..utils.numerics import phase_difference, wrap_phase
from .ode.limit_cycle import LimitCycle, TIME_TOL


def phase_sense(
    model: AbstractPhaseAmplitudeModel, cycle: LimitCycle
) -> Float[Array, ""]:
    """``+1`` when the model's phase increases along its own prediction, ``−1`` when it
    decreases: the sign of the shortest-arc phase change over a thousandth of the
    reference period from the reference cycle's phase origin."""
    ts = jnp.array([0.0, 1e-3]) * cycle.period
    xt, _ = model(ts, cycle.x0s[0])
    return jnp.sign(phase_difference(model.phase(xt[1]), model.phase(xt[0])))


def phase_offset(
    model: AbstractPhaseAmplitudeModel,
    cycle: LimitCycle,
    *,
    n_samples: int = 1024,
    newton_steps: int = 8,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""The model's phase at the reference cycle's phase 0, and the sense of its phase.

    The learned cycle is sampled at `n_samples` phases; among its crossings of
    ``cycle.section`` in the direction of the flow (the signed distance increasing
    with the phase, in the model's sense), the one nearest ``cycle.x0s[0]`` is taken
    and polished by ``optimistix.Newton`` on the signed distance.

    **Arguments:**

    - `model`: The trained model.
    - `cycle`: The reference cycle, whose section and phase origin define phase 0.
    - `n_samples`: The phases at which the learned cycle is sampled.
    - `newton_steps`: The Newton steps allowed for the polish.

    **Returns:**

    ``(offset, sense)``: the model phase of the crossing, in turns, wrapped to
    $[0, 1)$, and `phase_sense`. A reference phase $\vartheta$ is the model phase
    $\mathrm{sense}\cdot\vartheta + \mathrm{offset}$.

    **Raises:**

    `ValueError` when the learned cycle does not cross the section forward.
    """
    sense = phase_sense(model, cycle)

    def distance(theta, args):
        del args
        return cycle.section.signed_distance(model.cycle_point(theta))

    thetas = jnp.linspace(0.0, 1.0, n_samples, endpoint=False)
    values = jax.vmap(lambda t: distance(t, None))(thetas)
    following = jnp.roll(values, -1)
    forward = (values * following <= 0) & (sense * (following - values) > 0)
    if not bool(forward.any()):
        raise ValueError("the learned cycle does not cross the reference section.")
    points = jax.vmap(model.cycle_point)(thetas)
    gap = jnp.linalg.norm(points - cycle.x0s[0], axis=1)
    seed = thetas[jnp.argmin(jnp.where(forward, gap, jnp.inf))]
    sol = optx.root_find(
        distance,
        optx.Newton(rtol=0.0, atol=TIME_TOL),
        seed,
        max_steps=newton_steps,
        throw=False,
    )
    return wrap_phase(sol.value), sense


__all__ = ["phase_offset", "phase_sense"]
