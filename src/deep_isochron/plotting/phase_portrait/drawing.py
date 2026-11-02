"""The helpers that the phase-portrait animations share: the figure with its phase-plane
and time axes, the limit cycle, the fading tail, the clock, the time axis and the frame
sequence with its pauses. The time is labeled in units of the period, as in `t = 1.25T`,
and the phase-plane panel has no tick marks or tick labels, since the coordinates mean
nothing to the audience.

The figures draw under the style that the caller activates, such as
`configure_rcparams`, and override neither its grid nor its spines: the phase plane
keeps its tick positions so that a grid has lines to draw.

The design and its decisions are in `docs/changes/2026-11-02-ksiam-animations.md`.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.text import Text

from ...systems.normal_forms.base import AbstractNormalForm
from .style import PortraitStyle


_SAMPLES_PER_FRAME = 10
"""Trajectory samples per frame, so that traces and tails show no corners on a slide."""

_MARGINS = (0.15, 0.35, 0.5, 0.15)
"""The figure margins left, right, bottom and top, in inches: the bottom one holds the
time axis's tick labels, and the right one the half of its last label past the axis."""

_PANEL_GAP = 0.4
"""The gap between the phase plane and the time axis, in inches."""


# ------------------------------------------------------------ shared helpers ------
def _portrait_figure(style: PortraitStyle) -> tuple[Figure, Axes, Axes]:
    """A figure with a white opaque background, a square phase-plane axis on the left,
    spanning `±style.box` without tick marks or tick labels, and a time axis on the
    right with the same top and bottom edges.

    The axes are placed in inches from `style.figsize`, `_MARGINS` and `_PANEL_GAP`:
    the plane's side is the figure's height less the bottom and top margins, and the
    time axis takes the remaining width. Both axes draw their grid and ticks at zorder
    0.5, below the limit cycle at zorder 1 and every other artist.

    **Returns:**

    The figure, the phase-plane axis and the time axis.
    """
    width, height = style.figsize
    left, right, bottom, top = _MARGINS
    side = height - bottom - top
    time_left = left + side + _PANEL_GAP
    fig = plt.figure(figsize=style.figsize, dpi=style.dpi, facecolor="white")
    plane = fig.add_axes((left / width, bottom / height, side / width, side / height))
    time = fig.add_axes(
        (
            time_left / width,
            bottom / height,
            (width - right - time_left) / width,
            side / height,
        )
    )
    plane.set_xlim(-style.box, style.box)
    plane.set_ylim(-style.box, style.box)
    plane.set_aspect("equal")
    plane.tick_params(length=0, labelbottom=False, labelleft=False)
    for ax in (plane, time):  # a grid of the active style below every drawn artist
        ax.set_axisbelow(True)
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
    `0, T, 2T, …`, no y tick marks or tick labels, and the y-limits `y_limits`, fixed
    over the run."""
    ax.set_xlim(0.0, n_periods)
    ax.set_ylim(*y_limits)
    ax.set_xticks(range(n_periods + 1))
    ax.set_xticklabels(["0", "T"] + [f"{m}T" for m in range(2, n_periods + 1)])
    ax.tick_params(axis="x", labelsize=style.font_size)
    ax.tick_params(axis="y", length=0, labelleft=False)


def _frame_sequence(
    n_scene_frames: int, pauses: dict[int, float], fps: int
) -> list[tuple[int, bool]]:
    """The frames of an animation over a scene of `n_scene_frames` frames: each scene
    frame once, and each frame `j` of `pauses` then repeated `round(pauses[j] × fps)`
    times, so that the clock stops while it is shown.

    **Arguments:**

    - `n_scene_frames`: The number of frames of the scene.
    - `pauses`: The seconds for which each paused scene frame is held, the final hold
        included as the entry of the last frame.
    - `fps`: The playback rate in frames per second.

    **Returns:**

    Pairs `(scene frame, repeat)`, with `repeat` true for the repeats of a pause.

    **Raises:**

    `ValueError` when a key of `pauses` is not a frame of the scene.
    """
    outside = [j for j in pauses if not 0 <= j < n_scene_frames]
    if outside:
        raise ValueError(f"pauses at frames {outside} are outside the scene.")
    sequence = []
    for j in range(n_scene_frames):
        sequence.append((j, False))
        sequence.extend([(j, True)] * round(pauses.get(j, 0.0) * fps))
    return sequence
