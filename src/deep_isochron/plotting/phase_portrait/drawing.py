"""The helpers that the phase-portrait animations share: the figure with its phase-plane
and time axes, the one-panel square figure, the limit cycle, the fading tail, the clock,
the time axis and the frame sequence with its pauses. The time is labeled in units of
the period, as in `t = 1.25T` or `t = −2.00T`, and the phase-plane panel has no tick
marks or tick labels, since the coordinates mean nothing to the audience.

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
from matplotlib.lines import Line2D
from matplotlib.text import Text

from ...systems.normal_forms.base import AbstractNormalForm
from .style import PortraitStyle


MINUS = "\N{MINUS SIGN}"
"""The minus sign of the negative clock times and the phase-shift labels, wider than a
hyphen."""

_SAMPLES_PER_FRAME = 10
"""Trajectory samples per frame, so that traces and tails show no corners on a slide."""

_MARGINS = (0.15, 0.35, 0.5, 0.15)
"""The figure margins left, right, bottom and top, in inches: the bottom one holds the
time axis's tick labels, and the right one the half of its last label past the axis."""

_SQUARE_MARGIN = 0.325
"""The margin of `_square_figure` on every side, in inches: half the sum of the bottom
and top margins of `_MARGINS`, so the plane is centered with the vertical extent that
those margins leave."""

_PANEL_GAP = 0.4
"""The gap between the phase plane and the time axis, in inches."""

_TEXT_BOX_PAD = 3.0
"""The margin of the white box around the clock and the φ labels, in points."""

_TEXT_BOX = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=_TEXT_BOX_PAD)
"""The white box behind the clock and the φ labels of `animate_isochron_sync`, so that
they read over the curves that pass behind them."""


# ------------------------------------------------------------ shared helpers ------
def _plane_axis(
    fig: Figure, rect: tuple[float, float, float], style: PortraitStyle
) -> Axes:
    """Add a square phase-plane axis to `fig` at `rect`, its left edge, bottom edge and
    side in inches, spanning `±style.box` without tick marks or tick labels, with its
    grid and ticks below every drawn artist."""
    left, bottom, side = rect
    width, height = fig.get_size_inches()
    plane = fig.add_axes((left / width, bottom / height, side / width, side / height))
    plane.set_xlim(-style.box, style.box)
    plane.set_ylim(-style.box, style.box)
    plane.set_aspect("equal")
    plane.tick_params(length=0, labelbottom=False, labelleft=False)
    plane.set_axisbelow(True)  # a grid of the active style below every drawn artist
    return plane


def _portrait_figure(style: PortraitStyle) -> tuple[Figure, Axes, Axes]:
    """A figure with a white opaque background, a square phase-plane axis on the left
    from `_plane_axis`, and a time axis on the right with the same top and bottom
    edges.

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
    plane = _plane_axis(fig, (left, bottom, side), style)
    time = fig.add_axes(
        (
            time_left / width,
            bottom / height,
            (width - right - time_left) / width,
            side / height,
        )
    )
    time.set_axisbelow(True)  # a grid of the active style below every drawn artist
    return fig, plane, time


def _square_figure(style: PortraitStyle) -> tuple[Figure, Axes]:
    """A figure of `style.figsize_square` with a white opaque background and one
    phase-plane axis from `_plane_axis`, the one-panel sibling of `_portrait_figure`.

    The plane is centered, `_SQUARE_MARGIN` from every edge of the figure, so its side
    is the figure's width less two margins. It does not share `_MARGINS`, whose bottom
    and right margins hold the time axis's tick labels, which this figure does not have.

    **Returns:**

    The figure and the phase-plane axis.
    """
    side = style.figsize_square[0] - 2 * _SQUARE_MARGIN
    fig = plt.figure(figsize=style.figsize_square, dpi=style.dpi, facecolor="white")
    return fig, _plane_axis(fig, (_SQUARE_MARGIN, _SQUARE_MARGIN, side), style)


def _draw_cycle(ax: Axes, system: AbstractNormalForm, style: PortraitStyle) -> Line2D:
    """Draw the limit cycle of `system` on `ax` as a thick closed curve, and return
    the drawn line."""
    points = np.asarray(system.limit_cycle(jnp.linspace(0.0, 1.0, 256, endpoint=False)))
    closed = np.concatenate((points, points[:1]))
    (line,) = ax.plot(
        *closed.T, color=style.cycle_color, lw=style.cycle_width, zorder=1
    )
    return line


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
    do not jitter, in the white box `_TEXT_BOX` above every curve; `_clock_label`
    gives its string."""
    return ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        family="monospace",
        fontsize=style.font_size,
        ha="left",
        va="top",
        bbox=_TEXT_BOX,
        zorder=10,
    )


def _clock_label(frame: int, frames_per_period: int) -> str:
    """The clock's string at `frame`, the time in periods to two decimals, as in
    `t = 1.25T`, and with a Unicode minus before a negative time, as in `t = −2.00T`."""
    sign = MINUS if frame < 0 else ""
    return f"t = {sign}{abs(frame) / frames_per_period:.2f}T"


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
