"""The synchrony animation: `animate_isochron_sync` shows that points seeded on one
isochron fall into step on the limit cycle. The phase plane is on the left and the
horizontal coordinate against time on the right.

Every drawn object comes from the normal form itself: its closed-form `flow`,
`isochron`, `limit_cycle`, `period` and `in_basin`. `_sync_scene` computes the
trajectories and the phase of each group's isochron at every frame, and
`animate_isochron_sync` draws them with the helpers of `drawing.py`.

The design and its decisions are in `docs/changes/2026-11-02-ksiam-animations.md`.
"""

import math
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float
from matplotlib.animation import FuncAnimation

from ...systems.normal_forms.base import AbstractNormalForm
from .drawing import (
    _clock,
    _clock_label,
    _draw_cycle,
    _fading_tail,
    _frame_sequence,
    _portrait_figure,
    _SAMPLES_PER_FRAME,
    _set_tail,
    _time_axis,
)
from .style import PortraitStyle


_ISOCHRON_SAMPLES = 512
"""The radii at which each drawn isochron is sampled, so that its spiral shows no
corners."""

_INNER_RADIUS = 0.15
"""The smallest radius of each drawn isochron, where the isochrons crowd together."""

_LABEL_INSET = 0.85
"""The φ label sits at the outermost sample of its isochron within this fraction of
`style.box` from the center, so that the label stays inside the view."""


class _SyncScene(eqx.Module):
    """The trajectories of `animate_isochron_sync`: groups of points seeded on one
    isochron each, sampled `_SAMPLES_PER_FRAME` times per frame.

    **Attributes:**

    - `ts`: The sample times, shape `(samples,)`, from 0 in steps of
        `T / (frames_per_period × _SAMPLES_PER_FRAME)`; every `_SAMPLES_PER_FRAME`-th
        sample is a frame.
    - `isochron_phases`: The phase of each group's isochron at each frame, in turns in
        [0, 1), `(θ₀ + t/T) mod 1` for the seeded phase θ₀; shape `(m, frames)`. The
        points of a group stay on the isochron of this phase.
    - `trajectories`: The points of each group, one per radius, shape
        `(m, n, samples, 2)`.
    """

    ts: Float[Array, " samples"]
    isochron_phases: Float[Array, "m frames"]
    trajectories: Float[Array, "m n samples 2"]


def _sync_scene(
    system: AbstractNormalForm,
    *,
    phases: Float[ArrayLike, " m"],
    radii: Float[ArrayLike, " n"],
    n_periods: int,
    frames_per_period: int,
) -> _SyncScene:
    """Compute the scene of `animate_isochron_sync`: points seeded on the isochrons of
    `phases`, one per radius, and their flow.

    The points of a group start at `system.isochron(phase, radii)`, and every
    trajectory is `system.flow`. The isochron of a group at time `t` is the one of phase
    `(phase + t/T) mod 1`, since the flow maps each isochron onto the one a time `t`
    later in phase.

    **Arguments:**

    - `system`: The normal form.
    - `phases`: The phases of the seeded isochrons, in turns, one group per phase.
    - `radii`: The radii of the seeded points on each isochron.
    - `n_periods`: The length of the scene, in periods.
    - `frames_per_period`: The frames per period; the scene has `_SAMPLES_PER_FRAME`
        samples per frame.

    **Returns:**

    The `_SyncScene`, with `n_periods × frames_per_period + 1` frames and
    `n_periods × frames_per_period × _SAMPLES_PER_FRAME + 1` samples, the last at
    `t = n_periods × T`.

    **Raises:**

    `ValueError` when a phase is outside [0, 1), when `n_periods < 1`, or when a seeded
    point is outside `system.in_basin`, naming the phase and the radius of each.
    """
    phases = jnp.asarray(phases, dtype=float)
    radii = jnp.asarray(radii, dtype=float)
    seeded = np.asarray(phases)
    outside_unit = seeded[(seeded < 0.0) | (seeded >= 1.0)]
    if outside_unit.size:
        raise ValueError(f"phases {outside_unit.tolist()} are outside [0, 1).")
    if n_periods < 1:
        raise ValueError(f"n_periods={n_periods} is below 1.")
    starts = jax.vmap(lambda phase: system.isochron(phase, radii))(phases)
    inside = np.asarray(jax.vmap(jax.vmap(system.in_basin))(starts))
    if not inside.all():
        seeds = [(float(phases[g]), float(radii[i])) for g, i in np.argwhere(~inside)]
        raise ValueError(
            f"seeded points (phase, radius) {seeds} are outside the basin of the cycle."
        )
    per_period = frames_per_period * _SAMPLES_PER_FRAME
    ts = system.period() * jnp.arange(n_periods * per_period + 1) / per_period
    # type-level only: ys is None only for a solve that saves nothing
    trajectories = jax.vmap(jax.vmap(lambda u: cast(Array, system.flow(ts, u).ys)))(
        starts
    )
    frame_periods = jnp.arange(n_periods * frames_per_period + 1) / frames_per_period
    return _SyncScene(
        ts=ts,
        isochron_phases=jnp.mod(phases[:, None] + frame_periods[None], 1.0),
        trajectories=trajectories,
    )


def animate_isochron_sync(
    system: AbstractNormalForm,
    *,
    phases: Float[ArrayLike, " m"],
    radii: Float[ArrayLike, " n"],
    n_periods: int,
    style: PortraitStyle = PortraitStyle(),
) -> FuncAnimation:
    """Animate points seeded on chosen isochrons, to show that the points of one
    isochron fall into step on the limit cycle.

    The points of a group start on the isochron of one of `phases`, one point per
    radius of `radii`, and follow the flow. The isochron of the group is redrawn at
    every frame as the isochron of phase `(phase + t/T) mod 1`, so that its points stay
    on it while they slide onto the cycle. Each group, its isochron and its time series
    take one of `style.group_colors`. The animation freezes on the start for
    `style.pause_seconds`.

    The phase plane on the left shows the ten isochrons of phases `k/10` in
    `style.isochron_color`, fixed, the cycle, the moving isochrons, each point with a
    fading tail of `style.tail_periods` periods, and the clock. The time panel on the
    right shows the horizontal coordinate of every point as a line that grows with time.
    Tails and lines are drawn from `_SAMPLES_PER_FRAME` samples per frame. During the
    final hold of `style.hold_seconds` each group's phase is labeled, as in `φ = 0.50`,
    at `style.annotation_font_size` near the outer end of its isochron; after a whole
    number of periods that isochron is back at the seeded phase.

    The caller activates the plot style and saves the animation, with the explicit
    `dpi=style.dpi`, as `animate_kick`'s docstring describes.

    **Arguments:**

    - `system`: The normal form; its `flow`, closed form by default, gives every
        trajectory.
    - `phases`: The phases of the seeded isochrons, in turns, each in [0, 1); at most
        as many as `style.group_colors`.
    - `radii`: The radii of the seeded points on each isochron, inside the basin.
    - `n_periods`: The length of the animation, in periods; at least 1.
    - `style`: The drawing constants.

    **Returns:**

    The animation: `n_periods × style.frames_per_period + 1` frames, the start pause of
    `round(style.pause_seconds × style.fps)` frames and the final hold of
    `round(style.hold_seconds × style.fps)` frames. Its figure stays open until the
    caller closes it.

    **Raises:**

    `ValueError` when there are more phases than `style.group_colors`, and as
    `_sync_scene` raises.
    """
    n_groups = len(np.atleast_1d(np.asarray(phases)))
    if n_groups > len(style.group_colors):
        raise ValueError(
            f"{n_groups} phases, but style.group_colors has "
            f"{len(style.group_colors)} colors."
        )
    per_period = style.frames_per_period
    scene = _sync_scene(
        system,
        phases=phases,
        radii=radii,
        n_periods=n_periods,
        frames_per_period=per_period,
    )
    paths = np.asarray(scene.trajectories)
    colors = style.group_colors[:n_groups]
    samples_per_period = per_period * _SAMPLES_PER_FRAME
    times = np.arange(paths.shape[2]) / samples_per_period  # in periods
    tail_samples = round(style.tail_periods * samples_per_period)
    curve_radii = jnp.linspace(
        _INNER_RADIUS, style.box * math.sqrt(2), _ISOCHRON_SAMPLES
    )

    def isochrons(phase: Float[Array, " *k"]) -> Float[Array, " *k r 2"]:
        """The isochrons of `phase`, sampled at `curve_radii`."""
        flat = jax.vmap(lambda p: system.isochron(p, curve_radii))(phase.reshape(-1))
        return flat.reshape(*phase.shape, *flat.shape[1:])

    fig, plane, time = _portrait_figure(style)
    for curve in np.asarray(isochrons(jnp.arange(10) / 10)):
        plane.plot(
            *curve.T, color=style.isochron_color, lw=style.isochron_width, zorder=0.9
        )
    _draw_cycle(plane, system, style)
    moving = np.asarray(isochrons(scene.isochron_phases))  # (m, frames, r, 2)
    highlighted = [
        plane.plot([], [], color=c, lw=style.trace_width, zorder=2)[0] for c in colors
    ]
    tails = [[_fading_tail(plane, style) for _ in group] for group in paths]
    points = [
        plane.plot([], [], "o", color=c, ms=style.point_size, zorder=5)[0]
        for c in colors
    ]
    clock = _clock(plane, style)

    labels = []
    for phase, curve, c in zip(np.asarray(phases).reshape(-1), moving[:, -1], colors):
        shown = np.flatnonzero(
            np.all(np.abs(curve) <= _LABEL_INSET * style.box, axis=1)
        )
        x, y = curve[shown[-1]]  # the outermost sample in the view
        labels.append(
            plane.text(
                x,
                y,
                f"φ = {phase:.2f}",
                color=c,
                fontsize=style.annotation_font_size,
                ha="right" if x > 0 else "left",
                va="top" if y > 0 else "bottom",
                zorder=6,
                visible=False,
            )
        )

    x_all = paths[..., 0]
    _time_axis(
        time, n_periods, (float(x_all.min()) - 0.15, float(x_all.max()) + 0.15), style
    )
    traces = [
        [time.plot([], [], color=c, lw=style.trace_width)[0] for _ in group]
        for group, c in zip(paths, colors)
    ]
    heads = [
        time.plot([], [], "o", color=c, ms=style.point_size, zorder=5)[0]
        for c in colors
    ]
    last_frame = n_periods * per_period

    def draw(frame: tuple[int, bool]) -> tuple:
        """Update every artist to `frame`, a pair (scene frame, repeat), with the φ
        labels in the repeats of the last frame, the final hold."""
        j, repeat = frame
        s = j * _SAMPLES_PER_FRAME  # the sample of the frame
        start = max(s - tail_samples, 0)
        for g, (group, c) in enumerate(zip(paths, colors)):
            highlighted[g].set_data(*moving[g, j].T)
            for i, path in enumerate(group):
                _set_tail(tails[g][i], path[start : s + 1], c)
                traces[g][i].set_data(times[: s + 1], path[: s + 1, 0])
            points[g].set_data(group[:, s, 0], group[:, s, 1])
            heads[g].set_data(np.full(len(group), times[s]), group[:, s, 0])
        for label in labels:
            label.set_visible(repeat and j == last_frame)
        clock.set_text(_clock_label(j, per_period))
        flat_tails = [tail for group in tails for tail in group]
        flat_traces = [trace for group in traces for trace in group]
        return (
            *highlighted,
            *flat_tails,
            *points,
            *flat_traces,
            *heads,
            *labels,
            clock,
        )

    sequence = _frame_sequence(
        last_frame + 1,
        {0: style.pause_seconds, last_frame: style.hold_seconds},
        style.fps,
    )
    draw(sequence[0])
    return FuncAnimation(
        fig,
        draw,
        frames=sequence,
        interval=1000 / style.fps,
        blit=False,
        repeat=False,
    )
