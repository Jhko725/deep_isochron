"""Animations of a normal form for a talk: the phase plane on the left, the horizontal
coordinate against time on the right. `animate_kick` shows that a kick shifts the phase
of a point on the limit cycle, with a delay or an advance depending on the kick.

Every drawn object comes from the normal form itself: its closed-form `flow`, `phase`,
`limit_cycle`, `period` and `in_basin`. `PortraitStyle` holds the drawing constants, a
frozen dataclass changed one field at a time with `copy.replace`. Time is labeled in
units of the period, as in `t = 1.25T`, and the phase-plane panel has no ticks, since
the coordinates mean nothing to the audience.

The design and its decisions are in `docs/changes/2026-11-02-ksiam-animations.md`.
"""

import math
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, ArrayLike, Float
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text

from ..systems.normal_forms.base import AbstractNormalForm
from ..utils.numerics import phase_difference


MINUS = "\N{MINUS SIGN}"
"""The minus sign of the phase-shift labels, wider than a hyphen."""


@dataclass(frozen=True)
class PortraitStyle:
    """The drawing constants of the phase-portrait animations; `copy.replace` changes
    one.

    **Attributes:**

    - `figsize`, `dpi`: The figure size in inches and its resolution. The defaults give
        1920 × 800 pixels, even in both dimensions as the `yuv420p` pixel format of an
        MP4 requires.
    - `frames_per_period`: The frames drawn per period of the cycle.
    - `fps`: The playback rate in frames per second.
    - `hold_seconds`: How long the last frame is held, in seconds.
    - `tail_periods`: The length of the fading tail behind each point in the phase
        plane, in periods.
    - `kick_arrow_frames`: The number of frames for which the kick arrow is drawn.
    - `box`: The half-width of the square phase-plane view.
    - `font_size`: The size of all text, in points; the minimum readable on a slide.
    - `cycle_width`, `isochron_width`: The line widths of the limit cycle and of the
        isochrons.
    - `trace_width`: The line width of the tails and of the time series.
    - `point_size`: The marker size of each moving point.
    - `cycle_color`: The limit cycle.
    - `reference_color`: The unkicked reference point.
    - `delay_color`, `advance_color`: A kicked copy whose phase falls behind the
        reference's, and one whose phase moves ahead or stays level.
    - `group_colors`: The points seeded on the first and on the second isochron.
    - `isochron_color`: The isochrons drawn for orientation only.
    """

    figsize: tuple[float, float] = (12.0, 5.0)
    dpi: int = 160
    frames_per_period: int = 20
    fps: int = 20
    hold_seconds: float = 1.0
    tail_periods: float = 0.5
    kick_arrow_frames: int = 3
    box: float = 2.4
    font_size: float = 14.0
    cycle_width: float = 3.0
    isochron_width: float = 1.0
    trace_width: float = 2.0
    point_size: float = 9.0
    cycle_color: str = "black"
    reference_color: str = "#0072B2"
    delay_color: str = "#D55E00"
    advance_color: str = "#009E73"
    group_colors: tuple[str, str] = ("#0072B2", "#D55E00")
    isochron_color: str = "0.85"


# ------------------------------------------------------------ shared helpers ------
def _portrait_figure(style: PortraitStyle) -> tuple[Figure, Axes, Axes]:
    """A figure with a white opaque background, a square phase-plane axis without
    ticks on the left, spanning `±style.box`, and a time axis on the right.

    **Returns:**

    The figure, the phase-plane axis and the time axis.
    """
    fig = plt.figure(
        figsize=style.figsize, dpi=style.dpi, facecolor="white", layout="constrained"
    )
    plane, time = fig.subplots(1, 2, width_ratios=(1.0, 1.6))
    plane.set_xlim(-style.box, style.box)
    plane.set_ylim(-style.box, style.box)
    plane.set_aspect("equal")
    plane.set_xticks([])
    plane.set_yticks([])
    return fig, plane, time


def _draw_cycle(ax: Axes, system: AbstractNormalForm, style: PortraitStyle) -> None:
    """Draw the limit cycle of `system` on `ax` as a thick closed curve."""
    points = np.asarray(system.limit_cycle(jnp.linspace(0.0, 1.0, 256, endpoint=False)))
    closed = np.concatenate((points, points[:1]))
    ax.plot(*closed.T, color=style.cycle_color, lw=style.cycle_width, zorder=1)


def _fading_tail(ax: Axes, style: PortraitStyle) -> LineCollection:
    """An empty line collection on `ax` for a fading tail, filled by `_set_tail`."""
    tail = LineCollection([], linewidths=style.trace_width, capstyle="round", zorder=3)
    ax.add_collection(tail)
    return tail


def _set_tail(
    tail: LineCollection, points: Float[np.ndarray, "n 2"], color: str
) -> None:
    """Draw `points`, oldest first, as the segments of `tail` in `color`, its opacity
    rising linearly from 0 at the oldest segment to 1 at the newest."""
    segments = np.stack((points[:-1], points[1:]), axis=1)
    red, green, blue, _ = to_rgba(color)
    n = len(segments)
    tail.set_segments(list(segments))
    tail.set_color([(red, green, blue, (i + 1) / n) for i in range(n)])


def _clock(ax: Axes, style: PortraitStyle) -> Text:
    """The clock text at the top left of `ax`, in a monospace font so that its digits
    do not jitter; `_clock_label` gives its string."""
    return ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        family="monospace",
        fontsize=style.font_size,
        ha="left",
        va="top",
        zorder=10,
    )


def _clock_label(frame: int, frames_per_period: int) -> str:
    """The clock's string at `frame`, the time in periods to two decimals, as in
    `t = 1.25T`."""
    return f"t = {frame / frames_per_period:.2f}T"


def _time_axis(
    ax: Axes, n_periods: int, y_limits: tuple[float, float], style: PortraitStyle
) -> None:
    """Set up `ax` as the time axis over `n_periods` periods: x ticks at
    `0, T, 2T, …`, no y ticks, and the y-limits `y_limits`, fixed over the run."""
    ax.set_xlim(0.0, n_periods)
    ax.set_ylim(*y_limits)
    ax.set_xticks(range(n_periods + 1))
    ax.set_xticklabels(["0", "T"] + [f"{m}T" for m in range(2, n_periods + 1)])
    ax.set_yticks([])
    ax.tick_params(labelsize=style.font_size)


def _frame_sequence(
    n_scene_frames: int, style: PortraitStyle
) -> list[tuple[int, bool]]:
    """The frames of an animation over a scene of `n_scene_frames` frames: each scene
    frame once, then the last one repeated `round(style.hold_seconds × style.fps)`
    times.

    **Returns:**

    Pairs `(scene frame, holding)`, with `holding` true for the repeats of the hold.
    """
    hold = round(style.hold_seconds * style.fps)
    return [(j, False) for j in range(n_scene_frames)] + [
        (n_scene_frames - 1, True)
    ] * hold


# ------------------------------------------------------------------ the kick -------
class _KickScene(eqx.Module):
    """The trajectories of `animate_kick`: a reference point on the cycle and its
    copies, kicked at `t = T`, sampled at every frame.

    **Attributes:**

    - `ts`: The frame times, shape `(frames,)`, from 0 in steps of
        `T / frames_per_period`.
    - `phase_shifts`: The asymptotic phase shift Δφ of each copy against the reference,
        in turns in (−1/2, 1/2], positive when the copy is ahead; shape `(k,)`.
    - `trajectories`: The reference first, then the copies, shape `(k + 1, frames, 2)`.
        A copy equals the reference before the kick and is the kicked point from the
        kick's frame on.
    """

    ts: Float[Array, " frames"]
    phase_shifts: Float[Array, " k"]
    trajectories: Float[Array, "k+1 frames 2"]


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
    - `frames_per_period`: The frames per period.

    **Returns:**

    The `_KickScene`, with `n_periods × frames_per_period + 1` frames.

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
    n_frames = n_periods * frames_per_period + 1
    ts = system.period() * jnp.arange(n_frames) / frames_per_period
    # type-level only: ys is None only for a solve that saves nothing
    reference = cast(
        Array, system.flow(ts, system.limit_cycle(jnp.asarray(kick_phase))).ys
    )
    before = reference[frames_per_period]
    kicked = before + kicks
    outside = np.flatnonzero(~np.asarray(jax.vmap(system.in_basin)(kicked)))
    if outside.size:
        raise ValueError(
            f"kicks {outside.tolist()} move the point outside the basin of the cycle."
        )
    after = jax.vmap(lambda u: cast(Array, system.flow(ts[frames_per_period:], u).ys))(
        kicked
    )
    lead_in = jnp.broadcast_to(
        reference[:frames_per_period], (len(kicks), frames_per_period, 2)
    )
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
    each row of `kicks` is added to its state to give one copy, drawn with an arrow
    from the point before the kick to the point after it for
    `style.kick_arrow_frames` frames, while the reference continues unkicked. Each
    copy is colored by the sign of its phase shift Δφ: `style.advance_color` when it
    is ahead of the reference or level with it, `style.delay_color` when it is behind.

    The phase plane on the left shows the cycle, each point with a fading tail of
    `style.tail_periods` periods, and the clock. The time panel on the right shows the
    horizontal coordinate of every point as a line that grows with time. During the
    final hold it labels each copy's Δφ with a double arrow between the reference's
    last peak and the copy's matching peak, `Δφ` periods earlier for a copy ahead.
    The peaks are where the phase is 0, since the cycle's point at phase 0 is the one
    of largest horizontal coordinate (`AbstractNormalForm.limit_cycle`).

    The caller saves the animation, as a GIF with `PillowWriter(fps=style.fps)` or as
    an MP4 with the recipe of `animate_deformation`'s docstring and `fps=style.fps`.

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

    The animation, `n_periods × style.frames_per_period + 1` frames and then the hold;
    its figure stays open until the caller closes it.

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
    n_frames = paths.shape[1]
    times = np.arange(n_frames) / per_period  # in periods
    tail_frames = round(style.tail_periods * per_period)

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
                paths[0, per_period],
                path[per_period],
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

    rows = 1.2 + 0.3 * np.arange(len(shifts))  # heights of the Δφ arrows
    x = paths[..., 0]
    y_limits = (float(x.min()) - 0.15, max(float(x.max()), float(rows[-1])) + 0.2)
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
                    arrowprops=dict(arrowstyle="<->", color=c, lw=style.trace_width),
                ),
                *time.plot(
                    [ends[0], ends[0], np.nan, ends[1], ends[1]],
                    [1.0, row, np.nan, 1.0, row],
                    ":",
                    color=c,
                    lw=1.0,
                ),
                time.text(
                    min(ends) - 0.05,
                    row,
                    _shift_label(float(shift)),
                    color=c,
                    fontsize=style.font_size,
                    ha="right",
                    va="center",
                ),
            ]
        )
    for artist in (a for group in labels for a in group):
        artist.set_visible(False)

    def draw(frame: tuple[int, bool]) -> tuple:
        """Update every artist to `frame`, a pair (scene frame, holding), with the Δφ
        labels while holding."""
        j, holding = frame
        kicked = j >= per_period
        for i, (path, c) in enumerate(zip(paths, colors)):
            shown = i == 0 or kicked
            start = max(j - tail_frames, per_period if i > 0 else 0)
            _set_tail(tails[i], path[start : j + 1], c)
            tails[i].set_visible(shown)
            points[i].set_data(path[j : j + 1, 0], path[j : j + 1, 1])
            points[i].set_visible(shown)
            if i == 0:
                trace_t, trace_x = times[: j + 1], path[: j + 1, 0]
            else:  # from the reference at the kick, so that the jump is vertical
                trace_t = np.concatenate(
                    (times[per_period : per_period + 1], times[per_period : j + 1])
                )
                trace_x = np.concatenate(
                    (
                        paths[0, per_period : per_period + 1, 0],
                        path[per_period : j + 1, 0],
                    )
                )
            traces[i].set_data(trace_t, trace_x)
            traces[i].set_visible(shown)
            heads[i].set_data(times[j : j + 1], path[j : j + 1, 0])
            heads[i].set_visible(shown)
        for arrow in arrows:
            arrow.set_visible(per_period <= j < per_period + style.kick_arrow_frames)
        for artist in (a for group in labels for a in group):
            artist.set_visible(holding)
        clock.set_text(_clock_label(j, per_period))
        return (*tails, *points, *arrows, *traces, *heads, clock)

    sequence = _frame_sequence(n_frames, style)
    draw(sequence[0])
    return FuncAnimation(
        fig,
        draw,
        frames=sequence,
        interval=1000 / style.fps,
        blit=False,
        repeat=False,
    )
