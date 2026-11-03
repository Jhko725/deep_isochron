"""The phase-field animation: `animate_phase_field` shows that every point of the plane
gets an asymptotic phase and that the equilibrium gets none. A black point lays the
phase colors on the limit cycle, takes a second lap through them, and the colors are
then painted backward from the cycle in whole-period steps until only a hole around the
focus is left.

Every drawn object comes from the normal form itself: its closed-form
`to_phase_amplitude`, `floquet_multiplier` and `limit_cycle`. `_field_scene` computes
the phase and the isostable coordinate on the grid and the band of each step, and
`animate_phase_field` draws them with the helpers of `drawing.py`.

The design and its decisions are in `docs/changes/2026-11-02-ksiam-animations.md`.
"""

import math
from collections.abc import Sequence
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float
from matplotlib.animation import FuncAnimation
from matplotlib.typing import ColorType

from ...systems.normal_forms.base import AbstractNormalForm
from ..isochrons import PHASE_CMAP, PHASE_NORM
from .drawing import (
    _clock,
    _clock_label,
    _draw_cycle,
    _frame_sequence,
    _square_figure,
    _TEXT_BOX,
)
from .labels import _place_labels
from .style import PortraitStyle


_FIELD_RESOLUTION = 500
"""The grid points per axis of the painted phase field, over `±style.box`."""

_FADE_FRAMES = 10
"""The frames over which each backward band fades in, and over which the clock fades
from `t = 2.00T` to `t = 0.00T`."""

_STEP_SECONDS = 1.0
"""How long the end of lap 2, the reset clock and each painted step are held, in
seconds."""


class _FieldScene(eqx.Module):
    """The phase field of `animate_phase_field` on a square grid, and the band that each
    backward step paints.

    **Attributes:**

    - `grid`: The grid points of each axis, shape `(n,)`.
    - `phase`: The asymptotic phase Θ at each grid point, in turns in [0, 1), `nan`
        outside the basin; shape `(n, n)`, indexed by the vertical and then the
        horizontal grid point.
    - `isostable`: The isostable coordinate Ψ at each grid point, `nan` outside the
        basin; shape `(n, n)`, indexed as `phase`.
    - `thresholds`: The largest `|Ψ|` painted after `m` backward steps,
        `tolerance · μ^{−m}` for `m = 0, …, n_periods` and the Floquet multiplier μ;
        shape `(n_periods + 1,)`.
    """

    grid: Float[Array, " n"]
    phase: Float[Array, "n n"]
    isostable: Float[Array, "n n"]
    thresholds: Float[Array, " steps"]

    def step_mask(self, step: int) -> Bool[Array, "n n"]:
        """The grid points painted after `step` backward steps,
        `|Ψ| ≤ thresholds[step]`; step 0 is the band around the cycle."""
        return jnp.abs(self.isostable) <= self.thresholds[step]

    def lap_mask(self, phase: float) -> Bool[Array, "n n"]:
        """The grid points painted in lap 1 when the black point is at `phase`, in
        turns: the band of step 0 where `Θ ≤ phase`."""
        return self.step_mask(0) & (self.phase <= phase)


def _field_scene(
    system: AbstractNormalForm,
    *,
    tolerance: float,
    n_periods: int,
    grid: Float[ArrayLike, " n"],
) -> _FieldScene:
    """Compute the scene of `animate_phase_field`: the phase–amplitude coordinates of
    `system` on the square grid with the points `grid` on each axis, and the threshold
    of each backward step.

    One period of backward time multiplies `|Ψ|` by `μ^{−1}`, since the flow multiplies
    it by the Floquet multiplier μ every period, so step `m` paints `|Ψ| ≤
    tolerance · μ^{−m}`.

    **Arguments:**

    - `system`: The normal form.
    - `tolerance`: The largest `|Ψ|` of the band around the cycle.
    - `n_periods`: The number of backward steps, one period each.
    - `grid`: The grid points of each axis.

    **Returns:**

    The `_FieldScene`.

    **Raises:**

    `ValueError` when `tolerance ≤ 0` or `n_periods < 1`.
    """
    if not tolerance > 0.0:
        raise ValueError(f"tolerance={tolerance} is not positive.")
    if n_periods < 1:
        raise ValueError(f"n_periods={n_periods} is below 1.")
    grid = jnp.asarray(grid, dtype=float)
    xs, ys = jnp.meshgrid(grid, grid)
    points = jnp.stack((xs.reshape(-1), ys.reshape(-1)), axis=1)
    coordinates = jax.vmap(system.to_phase_amplitude)(points)
    steps = jnp.arange(n_periods + 1)
    return _FieldScene(
        grid=grid,
        phase=coordinates[:, 0].reshape(xs.shape),
        isostable=coordinates[:, 1].reshape(xs.shape),
        thresholds=tolerance * system.floquet_multiplier() ** (-steps),
    )


def animate_phase_field(
    system: AbstractNormalForm,
    *,
    tolerance: float,
    n_periods: int,
    style: PortraitStyle = PortraitStyle(),
) -> FuncAnimation:
    """Animate the asymptotic phase painted on the plane, to show that every point gets
    a phase and that the equilibrium gets none.

    The animation has six segments, at `style.frames_per_period` frames per period:

    1. The start, held for `style.pause_seconds`: a white plane, the cycle in
        `style.cycle_color`, the black point at phase 0 and the clock at `t = 0.00T`.
    2. Lap 1, one period: the point moves along the cycle, and the band
        `|Ψ| ≤ tolerance` is painted behind it, where `Θ ≤ t/T`.
    3. Lap 2, one period: the point moves again over the fixed, fully painted band, and
        the clock runs to `t = 2.00T`; the end is held for `_STEP_SECONDS`.
    4. The reset: the clock fades out and back in as `t = 0.00T` over `_FADE_FRAMES`
        frames, and nothing else changes; `t = 0.00T` is held for `_STEP_SECONDS`.
    5. Backward steps `m = 1, …, n_periods`: the clock reads `t = −mT`, and the band
        `|Ψ| ≤ tolerance · μ^{−m}` fades in over `_FADE_FRAMES` frames, the new pixels'
        opacity rising from 0 to 1, then holds for `_STEP_SECONDS`. The point stays at
        phase 0.
    6. The final hold, for `style.hold_seconds`: an open circle at the equilibrium and
        the label `no phase` at `style.annotation_font_size` in the white box of the
        clock, placed by `_place_labels` as far as it can from the cycle, the point and
        the open circle.

    The colors are Θ in `PHASE_CMAP` and `PHASE_NORM` on a grid of `_FIELD_RESOLUTION`
    points per axis over `±style.box`; an unpainted pixel is transparent over the white
    background, and a pixel outside the basin of the cycle, where Θ is `nan`, is never
    painted. The figure is one square panel of `style.figsize_square`.

    The caller activates the plot style and saves the animation, with the explicit
    `dpi=style.dpi`, as `animate_kick`'s docstring describes.

    **Arguments:**

    - `system`: The normal form; its `to_phase_amplitude` gives the colors.
    - `tolerance`: The largest `|Ψ|` of the band painted in lap 1; positive.
    - `n_periods`: The number of backward steps, one period each; at least 1.
    - `style`: The drawing constants.

    **Returns:**

    The animation: `2 × style.frames_per_period + 1` frames of the laps,
    `(n_periods + 1) × _FADE_FRAMES` frames of the reset and the steps and one final
    frame, with the start pause of `round(style.pause_seconds × style.fps)` frames, a
    hold of `round(_STEP_SECONDS × style.fps)` frames after lap 2, after the reset and
    after each step, and the final hold of `round(style.hold_seconds × style.fps)`
    frames. Its figure stays open until the caller closes it.

    **Raises:**

    `ValueError` when `style.figsize_square` is not square, and as `_field_scene`
    raises.
    """
    width, height = style.figsize_square
    if width != height:
        raise ValueError(f"style.figsize_square={style.figsize_square} is not square.")
    per_period = style.frames_per_period
    scene = _field_scene(
        system,
        tolerance=tolerance,
        n_periods=n_periods,
        grid=jnp.linspace(-style.box, style.box, _FIELD_RESOLUTION),
    )
    colors = PHASE_CMAP(PHASE_NORM(np.asarray(scene.phase)))  # (n, n, 4)
    steps = [np.asarray(scene.step_mask(m)) for m in range(n_periods + 1)]
    laps_end = 2 * per_period
    steps_start = laps_end + _FADE_FRAMES
    last_frame = steps_start + n_periods * _FADE_FRAMES + 1
    positions = np.asarray(
        system.limit_cycle(jnp.mod(jnp.arange(laps_end + 1) / per_period, 1.0))
    )

    fig, plane = _square_figure(style)
    grid = np.asarray(scene.grid)
    # colored per pixel by `paint` rather than by a mapped array, so that each pixel
    # takes its own opacity: `set_alpha` with an array converts every pixel's color
    # separately, about 17 s a frame for this grid under matplotlib 3.11
    mesh = plane.pcolormesh(
        grid, grid, np.zeros(colors.shape[:2]), shading="nearest", zorder=0.8
    )
    mesh.set_array(None)
    cycle = _draw_cycle(plane, system, style)
    (point,) = plane.plot(
        [], [], "o", color=style.cycle_color, ms=style.point_size, zorder=5
    )
    (hole,) = plane.plot(
        [0.0],
        [0.0],
        "o",
        ms=style.point_size,
        markerfacecolor="white",
        markeredgecolor="black",
        zorder=5,
        visible=False,
    )
    label = plane.text(
        0.0,
        0.0,
        "no phase",
        fontsize=style.annotation_font_size,
        bbox=_TEXT_BOX,
        zorder=6,
        visible=False,
    )
    clock = _clock(plane, style)

    def paint(
        mask: Bool[np.ndarray, "n n"], opacity: float | Float[np.ndarray, "n n"]
    ) -> None:
        """Color the pixels of `mask` by their phase at `opacity` over the white
        background, and leave the others transparent."""
        faces = colors.copy()
        faces[..., 3] = np.where(mask, opacity, 0.0)
        # type-level only: matplotlib accepts an (n, 4) RGBA array, its stub a sequence
        mesh.set_facecolor(cast(Sequence[ColorType], faces.reshape(-1, 4)))

    def draw(frame: tuple[int, bool]) -> tuple:
        """Update every artist to `frame`, a pair (scene frame, repeat): the laps up to
        `laps_end`, the reset, the backward steps, and the final frame, the only one
        that shows the open circle and its label."""
        j, _ = frame
        clock.set_alpha(1.0)
        if j <= per_period:  # lap 1 paints behind the point
            paint(np.asarray(scene.lap_mask(j / per_period)), 1.0)
            clock.set_text(_clock_label(j, per_period))
        elif j <= laps_end:  # lap 2 over the fixed band
            paint(steps[0], 1.0)
            clock.set_text(_clock_label(j, per_period))
        elif j <= steps_start:  # the clock fades out at 2.00T and back in at 0.00T
            k = j - laps_end
            paint(steps[0], 1.0)
            clock.set_text(
                _clock_label(laps_end if 2 * k < _FADE_FRAMES else 0, per_period)
            )
            clock.set_alpha(abs(1.0 - 2 * k / _FADE_FRAMES))
        elif j < last_frame:
            m = math.ceil((j - steps_start) / _FADE_FRAMES)
            k = j - steps_start - (m - 1) * _FADE_FRAMES
            paint(steps[m], np.where(steps[m - 1], 1.0, k / _FADE_FRAMES))
            clock.set_text(_clock_label(-m * per_period, per_period))
        else:
            paint(steps[n_periods], 1.0)
            clock.set_text(_clock_label(-n_periods * per_period, per_period))
        s = min(j, laps_end)  # the point stays at phase 0 after the laps
        point.set_data(positions[s : s + 1, 0], positions[s : s + 1, 1])
        hole.set_visible(j == last_frame)
        label.set_visible(j == last_frame)
        return (mesh, point, hole, label, clock)

    pauses = {0: style.pause_seconds, laps_end: _STEP_SECONDS}
    pauses |= {
        steps_start + m * _FADE_FRAMES: _STEP_SECONDS for m in range(n_periods + 1)
    }  # m = 0 is the reset clock at t = 0.00T
    pauses[last_frame] = style.hold_seconds
    sequence = _frame_sequence(last_frame + 1, pauses, style.fps)
    draw((last_frame, False))  # the final frame, the only one that shows the label
    _place_labels(
        plane, [label], np.zeros((1, 2)), [cycle, point, hole], clock, float(fig.dpi)
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
