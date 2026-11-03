"""The kick animation: `animate_kick` shows that a kick shifts the phase of a point on
the limit cycle, with a delay or an advance depending on the kick. The phase plane is on
the left and the horizontal coordinate against time on the right.

Every drawn object comes from the normal form itself: its closed-form `flow`, `phase`,
`limit_cycle`, `period` and `in_basin`. `_kick_scene` computes the trajectories and the
phase shifts, and `animate_kick` draws them with the helpers of `drawing.py`.

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
from matplotlib.patches import FancyArrowPatch

from ...systems.normal_forms.base import AbstractNormalForm
from ...utils.numerics import phase_difference
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
    MINUS,
)
from .style import PortraitStyle


class _KickScene(eqx.Module):
    """The trajectories of `animate_kick`: a reference point on the cycle and its
    copies, kicked at `t = T`, sampled `_SAMPLES_PER_FRAME` times per frame.

    **Attributes:**

    - `ts`: The sample times, shape `(samples,)`, from 0 in steps of
        `T / (frames_per_period × _SAMPLES_PER_FRAME)`; every `_SAMPLES_PER_FRAME`-th
        sample is a frame.
    - `phase_shifts`: The asymptotic phase shift Δφ of each copy against the reference,
        in turns in (−1/2, 1/2], positive when the copy is ahead; shape `(k,)`.
    - `trajectories`: The reference first, then the copies, shape
        `(k + 1, samples, 2)`. A copy equals the reference before the kick and is the
        kicked point from the kick's sample on.
    """

    ts: Float[Array, " samples"]
    phase_shifts: Float[Array, " k"]
    trajectories: Float[Array, "k+1 samples 2"]


def _kick_scene(
    system: AbstractNormalForm,
    *,
    kick_phase: float,
    kicks: Float[ArrayLike, "k 2"],
    n_periods: int,
    frames_per_period: int,
) -> _KickScene:
    """Compute the scene of `animate_kick`: a point that starts on the limit cycle and
    copies of it kicked after one period.

    The reference starts at `system.limit_cycle(kick_phase)`. At `t = T` each row of
    `kicks` is added to its state to give one copy, and the reference continues
    unkicked. Every trajectory is `system.flow`. The phase shift of a copy is
    `phase_difference` of the copy's phase and the reference's phase at the kick,
    positive when the copy is ahead (ADR-0020).

    **Arguments:**

    - `system`: The normal form.
    - `kick_phase`: The phase of the starting point, in turns.
    - `kicks`: The jumps added to the state, one copy per row.
    - `n_periods`: The length of the scene, in periods, the one-period lead-in
        included.
    - `frames_per_period`: The frames per period; the scene has `_SAMPLES_PER_FRAME`
        samples per frame.

    **Returns:**

    The `_KickScene`, with `n_periods × frames_per_period × _SAMPLES_PER_FRAME + 1`
    samples, the last at `t = n_periods × T`.

    **Raises:**

    `ValueError` when `kick_phase` is outside [0, 1), when `n_periods < 2`, when
    `kicks` is not of shape `(k, 2)` with `k ≥ 1`, or when a kicked point is outside
    `system.in_basin`, naming the kicks that are.
    """
    if not 0.0 <= kick_phase < 1.0:
        raise ValueError(f"kick_phase={kick_phase} is outside [0, 1).")
    if n_periods < 2:
        raise ValueError(f"n_periods={n_periods} is below 2: one period comes first.")
    kicks = jnp.asarray(kicks, dtype=float)
    if kicks.ndim != 2 or kicks.shape[1] != 2 or kicks.shape[0] < 1:
        raise ValueError(
            f"kicks must be of shape (k, 2) with k >= 1; got {kicks.shape}."
        )
    per_period = frames_per_period * _SAMPLES_PER_FRAME
    ts = system.period() * jnp.arange(n_periods * per_period + 1) / per_period
    # type-level only: ys is None only for a solve that saves nothing
    reference = cast(
        Array, system.flow(ts, system.limit_cycle(jnp.asarray(kick_phase))).ys
    )
    before = reference[per_period]
    kicked = before + kicks
    outside = np.flatnonzero(~np.asarray(jax.vmap(system.in_basin)(kicked)))
    if outside.size:
        raise ValueError(
            f"kicks {outside.tolist()} move the point outside the basin of the cycle."
        )
    after = jax.vmap(lambda u: cast(Array, system.flow(ts[per_period:], u).ys))(kicked)
    lead_in = jnp.broadcast_to(reference[:per_period], (len(kicks), per_period, 2))
    copies = jnp.concatenate((lead_in, after), axis=1)
    phase_shifts = phase_difference(
        jax.vmap(system.phase)(kicked), system.phase(before)
    )
    return _KickScene(
        ts=ts,
        phase_shifts=phase_shifts,
        trajectories=jnp.concatenate((reference[None], copies)),
    )


def _shift_label(phase_shift: float) -> str:
    """The label of a phase shift, signed to two decimals with a Unicode minus, as in
    `Δφ = −0.15` or `Δφ = +0.15`; exactly zero reads as an advance."""
    sign = MINUS if phase_shift < 0 else "+"
    return f"Δφ = {sign}{abs(phase_shift):.2f}"


def animate_kick(
    system: AbstractNormalForm,
    *,
    kick_phase: float,
    kicks: Float[ArrayLike, "k 2"],
    n_periods: int,
    style: PortraitStyle = PortraitStyle(),
) -> FuncAnimation:
    """Animate a point on the limit cycle and its kicked copies, to show that a kick
    shifts the phase.

    The point starts on the cycle at phase `kick_phase` and runs for one period. Then
    each row of `kicks` is added to its state to give one copy, while the reference
    continues unkicked. The animation freezes on the first frame after the kick for
    `style.pause_seconds`, with an arrow from the point before the kick to each
    copy, and the arrows stay for `style.kick_arrow_frames` frames after that. Each
    copy is colored by the sign of its phase shift Δφ: `style.advance_color` when it
    is ahead of the reference or level with it, `style.delay_color` when it is behind.

    The phase plane on the left shows the cycle, each point with a fading tail of
    `style.tail_periods` periods, and the clock. The time panel on the right shows the
    horizontal coordinate of every point as a line that grows with time. Tails and
    lines are drawn from `_SAMPLES_PER_FRAME` samples per frame. During the final hold
    of `style.hold_seconds` the time panel labels each copy's Δφ, at
    `style.annotation_font_size`, with a double arrow between the reference's
    last peak and the copy's matching peak, `Δφ` periods earlier for a copy ahead.
    The peaks are where the phase is 0, since the cycle's point at phase 0 is the one
    of largest horizontal coordinate (`AbstractNormalForm.limit_cycle`).

    The caller activates the plot style, such as `configure_rcparams`, and saves the
    animation, as a GIF with
    `animation.save(path, writer=PillowWriter(fps=style.fps), dpi=style.dpi)` or as an
    MP4 with the recipe of `animate_deformation`'s docstring, `fps=style.fps` and
    `dpi=style.dpi`. The explicit `dpi` keeps the size of `style.figsize` and
    `style.dpi`, since `FuncAnimation.save` otherwise takes `savefig.dpi`, which the
    project style sets to 600.

    **Arguments:**

    - `system`: The normal form; its `flow`, closed form by default, gives every
        trajectory.
    - `kick_phase`: The phase of the starting point, in turns, in [0, 1).
    - `kicks`: The jumps added to the state after one period, one copy per row; they
        alone set the length and the direction of each kick.
    - `n_periods`: The length of the animation, in periods, the one-period lead-in
        included; at least 2.
    - `style`: The drawing constants.

    **Returns:**

    The animation: `n_periods × style.frames_per_period + 1` frames, the kick pause of
    `round(style.pause_seconds × style.fps)` frames and the final hold of
    `round(style.hold_seconds × style.fps)` frames. Its figure stays open until the
    caller closes it.

    **Raises:**

    `ValueError` as `_kick_scene` raises.
    """
    per_period = style.frames_per_period
    scene = _kick_scene(
        system,
        kick_phase=kick_phase,
        kicks=kicks,
        n_periods=n_periods,
        frames_per_period=per_period,
    )
    paths = np.asarray(scene.trajectories)
    shifts = np.asarray(scene.phase_shifts)
    colors = [style.reference_color] + [
        style.delay_color if shift < 0 else style.advance_color for shift in shifts
    ]
    samples_per_period = per_period * _SAMPLES_PER_FRAME
    kick = samples_per_period  # the sample of the kick, at t = T
    times = np.arange(paths.shape[1]) / samples_per_period  # in periods
    tail_samples = round(style.tail_periods * samples_per_period)

    fig, plane, time = _portrait_figure(style)
    _draw_cycle(plane, system, style)
    tails = [_fading_tail(plane, style) for _ in colors]
    points = [
        plane.plot([], [], "o", color=c, ms=style.point_size, zorder=5)[0]
        for c in colors
    ]
    arrows = [
        plane.add_patch(
            FancyArrowPatch(
                paths[0, kick],
                path[kick],
                arrowstyle="-|>",
                mutation_scale=22,
                lw=style.trace_width,
                color=c,
                zorder=4,
                visible=False,
            )
        )
        for path, c in zip(paths[1:], colors[1:])
    ]
    clock = _clock(plane, style)

    rows = 1.35 + 0.55 * np.arange(len(shifts))  # heights of the Δφ arrows
    x = paths[..., 0]
    y_limits = (float(x.min()) - 0.15, max(float(x.max()), float(rows[-1])) + 0.35)
    _time_axis(time, n_periods, y_limits, style)
    traces = [time.plot([], [], color=c, lw=style.trace_width)[0] for c in colors]
    heads = [
        time.plot([], [], "o", color=c, ms=style.point_size, zorder=5)[0]
        for c in colors
    ]
    # the reference's last peak (phase kick_phase + t/T an integer) whose copies' peaks
    # all fall within the run
    latest = n_periods - max(0.0, -float(shifts.min()))
    peak = math.floor(latest + kick_phase) - kick_phase
    labels = []
    for shift, row, c in zip(shifts, rows, colors[1:]):
        ends = (peak, peak - float(shift))
        labels.append(
            [
                time.annotate(
                    "",
                    xy=(ends[1], row),
                    xytext=(ends[0], row),
                    arrowprops=dict(
                        arrowstyle="<->",
                        color=c,
                        lw=1.25 * style.trace_width,
                        mutation_scale=12,
                        shrinkA=0.0,
                        shrinkB=0.0,
                    ),
                ),
                *time.plot(
                    [ends[0], ends[0], np.nan, ends[1], ends[1]],
                    [1.0, row, np.nan, 1.0, row],
                    ":",
                    color=c,
                    lw=1.5,
                ),
                time.text(
                    min(ends) - 0.08,
                    row,
                    _shift_label(float(shift)),
                    color=c,
                    fontsize=style.annotation_font_size,
                    ha="right",
                    va="center",
                ),
            ]
        )
    for artist in (a for group in labels for a in group):
        artist.set_visible(False)
    last_frame = n_periods * per_period

    def draw(frame: tuple[int, bool]) -> tuple:
        """Update every artist to `frame`, a pair (scene frame, repeat), with the Δφ
        labels in the repeats of the last frame, the final hold."""
        j, repeat = frame
        s = j * _SAMPLES_PER_FRAME  # the sample of the frame
        for i, (path, c) in enumerate(zip(paths, colors)):
            shown = i == 0 or j >= per_period
            start = max(s - tail_samples, kick if i > 0 else 0)
            _set_tail(tails[i], path[start : s + 1], c)
            tails[i].set_visible(shown)
            points[i].set_data(path[s : s + 1, 0], path[s : s + 1, 1])
            points[i].set_visible(shown)
            if i == 0:
                trace_t, trace_x = times[: s + 1], path[: s + 1, 0]
            else:  # from the reference at the kick, so that the jump is vertical
                trace_t = np.concatenate((times[kick : kick + 1], times[kick : s + 1]))
                trace_x = np.concatenate(
                    (paths[0, kick : kick + 1, 0], path[kick : s + 1, 0])
                )
            traces[i].set_data(trace_t, trace_x)
            traces[i].set_visible(shown)
            heads[i].set_data(times[s : s + 1], path[s : s + 1, 0])
            heads[i].set_visible(shown)
        for arrow in arrows:
            arrow.set_visible(per_period <= j <= per_period + style.kick_arrow_frames)
        for artist in (a for group in labels for a in group):
            artist.set_visible(repeat and j == last_frame)
        clock.set_text(_clock_label(j, per_period))
        return (*tails, *points, *arrows, *traces, *heads, clock)

    sequence = _frame_sequence(
        last_frame + 1,
        {per_period: style.pause_seconds, last_frame: style.hold_seconds},
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
