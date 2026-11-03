"""The animations of ``plotting.phase_portrait`` on the normal form of the talk.

The phase shift that `animate_kick` labels is pinned against an independent oracle: the
polar angle of each kicked copy twelve periods after the kick, when the kick has relaxed
onto the cycle. The animation is pinned by the frames of a saved GIF, its layout by
the edges of its two panels, and its arguments by the four errors it raises. The figure
tests run under the project style, as the notebooks do.

The points of `animate_isochron_sync` are pinned by the closed-form phase: it equals the
seeded phase at the start and the phase of the group's drawn isochron at every frame.
Its animation is pinned by the frames of a saved GIF, its φ labels by their distance
from the foreground curves, and its arguments by the four errors it raises.

The bands that `animate_phase_field` paints are pinned by the isostable coordinate and
the panel's known Floquet multiplier e^{-1}, its lap-1 sweep by the closed-form phase of
the moving point, its animation by the frames of a saved GIF, and its arguments by the
three errors it raises.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.plotting import (
    animate_isochron_sync,
    animate_kick,
    animate_phase_field,
    configure_rcparams,
    PortraitStyle,
)
from deep_isochron.plotting.phase_portrait.drawing import (
    _SAMPLES_PER_FRAME,
    _TEXT_BOX_PAD,
)
from deep_isochron.plotting.phase_portrait.field import (
    _FADE_FRAMES,
    _FIELD_RESOLUTION,
    _field_scene,
    _STEP_SECONDS,
)
from deep_isochron.plotting.phase_portrait.kick import _kick_scene
from deep_isochron.plotting.phase_portrait.labels import _MIN_LABEL_CLEARANCE_PX
from deep_isochron.plotting.phase_portrait.sync import _INNER_RADIUS, _sync_scene
from deep_isochron.systems.normal_forms import (
    AbstractNormalForm,
    BautinNormalForm,
    WinfreeNormalForm,
)
from deep_isochron.utils.numerics import phase_difference
from jaxtyping import Float
from matplotlib.animation import PillowWriter
from matplotlib.transforms import Bbox
from PIL import Image, ImageSequence

from .helpers import TOL


# Panel 2 of the contact sheet: T = 1 and μ = e^{-1} = 0.37, so a kick relaxes visibly
PANEL_2 = BautinNormalForm(a=1 / 3, b=0.5, w=2 * np.pi, w0=2 * np.pi - 2 / 3)
# Winfree's model with the paper's hole: the disk r <= 0.25 is outside the basin
WINFREE_HOLE = WinfreeNormalForm(a=0.25, w=-0.5)
KICKS = np.array([[0.75, 0.0], [-0.75, 0.0]])
"""The default scene of the talk: a kick to the right and one to the left."""
PHASES = np.array([0.0, 0.5])
RADII = np.linspace(0.5, 1.6, 5)
"""The default scene of the synchrony slide: five points on each of two isochrons."""


def test_kick_phase_shift_matches_relaxed_angles():
    """Twelve periods after the kick, the signed difference of polar angles between
    each copy and the reference, in turns, is the phase shift of the scene."""
    n_periods = 13
    scene = _kick_scene(
        PANEL_2,
        kick_phase=0.83,
        kicks=KICKS,
        n_periods=n_periods,
        frames_per_period=20,
    )
    final = np.asarray(scene.trajectories[:, -1])
    angles = np.arctan2(final[:, 1], final[:, 0]) / (2 * np.pi)
    relaxed = (angles[1:] - angles[0] + 0.5) % 1.0 - 0.5
    np.testing.assert_allclose(
        np.asarray(scene.phase_shifts), relaxed, rtol=0.0, atol=TOL["relaxed_phase"]
    )


def test_animate_kick_writes_every_frame(tmp_path: Path):
    """The animation saved as a GIF under the project style plays 4 × 40 + 1 frames,
    the 30 frames of the 1.5-second kick pause and the 60 frames of the 3-second final
    hold. Pillow merges identical consecutive frames into one of longer duration, so the
    frames are counted by duration, 50 ms each at 20 fps."""
    path = tmp_path / "kick.gif"
    with matplotlib.rc_context():
        configure_rcparams()
        animation = animate_kick(PANEL_2, kick_phase=0.83, kicks=KICKS, n_periods=4)
        animation.save(path, writer=PillowWriter(fps=20), dpi=PortraitStyle().dpi)
        plt.close("all")
    with Image.open(path) as gif:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    assert sum(durations) == (4 * 40 + 1 + round(1.5 * 20) + round(3.0 * 20)) * 50


def test_panels_share_top_and_bottom():
    """Under the project style, the drawn phase plane is square and its top and bottom
    edges are those of the time axis, to within one pixel."""
    with matplotlib.rc_context():
        configure_rcparams()
        animation = animate_kick(PANEL_2, kick_phase=0.83, kicks=KICKS, n_periods=4)
        fig = plt.gcf()  # the figure that animate_kick has just created
        fig.canvas.draw()  # also renders the first frame of the animation
        plane, time = (ax.get_window_extent() for ax in fig.axes)
        plt.close("all")
    del animation
    assert abs(plane.width - plane.height) <= 1.0
    assert abs(plane.y0 - time.y0) <= 1.0
    assert abs(plane.y1 - time.y1) <= 1.0


@pytest.mark.parametrize(
    ("system", "arguments", "message"),
    [
        (PANEL_2, dict(kick_phase=1.0), "kick_phase"),
        (PANEL_2, dict(n_periods=1), "n_periods"),
        (PANEL_2, dict(kicks=np.zeros((0, 2))), "shape"),
        (
            WINFREE_HOLE,
            dict(kick_phase=0.0, kicks=np.array([[0.1, 0.0], [-0.9, 0.0]])),
            r"kicks \[1\]",
        ),
    ],
    ids=["kick_phase", "n_periods", "kicks_shape", "basin"],
)
def test_animate_kick_raises(
    system: AbstractNormalForm, arguments: dict[str, object], message: str
):
    """A kick phase outside [0, 1), fewer than two periods, no kick, and a kick that
    leaves the basin each raise `ValueError`; the last names the kick."""
    defaults = dict(kick_phase=0.83, kicks=KICKS, n_periods=4)
    with pytest.raises(ValueError, match=message):
        animate_kick(system, **(defaults | arguments))


def test_sync_scene_seeds_the_given_phases():
    """The closed-form phase of every start point is the phase of its isochron."""
    scene = _sync_scene(
        PANEL_2, phases=PHASES, radii=RADII, n_periods=1, frames_per_period=20
    )
    seeded = np.asarray(jax.vmap(jax.vmap(PANEL_2.phase))(scene.trajectories[:, :, 0]))
    np.testing.assert_allclose(
        phase_difference(seeded, PHASES[:, None]),
        0.0,
        rtol=0.0,
        atol=TOL["closed_form"],
    )


def test_sync_points_stay_on_their_moving_isochron():
    """At every frame, the closed-form phase of every point is `(θ₀ + t/T) mod 1`, and
    so is the phase of its group's drawn isochron."""
    frames_per_period = 20
    n_periods = 2
    scene = _sync_scene(
        PANEL_2,
        phases=PHASES,
        radii=RADII,
        n_periods=n_periods,
        frames_per_period=frames_per_period,
    )
    frames = scene.trajectories[:, :, ::_SAMPLES_PER_FRAME]
    phases = np.asarray(jax.vmap(jax.vmap(jax.vmap(PANEL_2.phase)))(frames))
    elapsed = np.arange(n_periods * frames_per_period + 1) / frames_per_period
    expected = (PHASES[:, None] + elapsed[None]) % 1.0
    np.testing.assert_allclose(
        phase_difference(np.asarray(scene.isochron_phases), expected),
        0.0,
        rtol=0.0,
        atol=TOL["closed_form"],
    )
    np.testing.assert_allclose(
        phase_difference(phases, expected[:, None]),
        0.0,
        rtol=0.0,
        atol=TOL["closed_form"],
    )


def test_animate_isochron_sync_writes_every_frame(tmp_path: Path):
    """The animation saved as a GIF under the project style plays 5 × 40 + 1 frames,
    the 30 frames of the 1.5-second start pause and the 60 frames of the 3-second final
    hold, counted by duration as in `test_animate_kick_writes_every_frame`."""
    path = tmp_path / "sync.gif"
    style = PortraitStyle()
    with matplotlib.rc_context():
        configure_rcparams()
        animation = animate_isochron_sync(
            PANEL_2, phases=PHASES, radii=RADII, n_periods=5, style=style
        )
        animation.save(path, writer=PillowWriter(fps=style.fps), dpi=style.dpi)
        plt.close("all")
    with Image.open(path) as gif:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    assert sum(durations) == (5 * 40 + 1 + round(1.5 * 20) + round(3.0 * 20)) * 50


def _box_gap(box: Bbox, points: Float[np.ndarray, "k 2"]) -> Float[np.ndarray, " k"]:
    """The distance in pixels from each display point of `points` to the box `box`,
    zero for a point inside it."""
    dx = np.maximum(np.maximum(box.x0 - points[:, 0], points[:, 0] - box.x1), 0.0)
    dy = np.maximum(np.maximum(box.y0 - points[:, 1], points[:, 1] - box.y1), 0.0)
    return np.hypot(dx, dy)


def test_sync_labels_clear_the_foreground():
    """In the final hold of the default scene under the project style, each φ label's
    white box lies inside the phase plane, touches neither the clock's box nor the
    other label's, and keeps `_MIN_LABEL_CLEARANCE_PX` from the foreground: the cycle
    and the two seeded isochrons sampled densely from the closed form, and the point
    markers, each measured with its drawn width."""
    style = PortraitStyle()
    n_periods = 5
    with matplotlib.rc_context():
        configure_rcparams()
        animation = animate_isochron_sync(
            PANEL_2, phases=PHASES, radii=RADII, n_periods=n_periods, style=style
        )
        fig = plt.gcf()  # the figure that animate_isochron_sync has just created
        # a canvas draw would otherwise start the animation and redraw its first frame
        fig.canvas.mpl_disconnect(animation._first_draw_id)
        animation._func((n_periods * style.frames_per_period, True))
        fig.canvas.draw()
        plane = fig.axes[0]
        pad = _TEXT_BOX_PAD * style.dpi / 72
        px = style.dpi / 72  # pixels per point
        labels = [
            t.get_window_extent().padded(pad)
            for t in plane.texts
            if t.get_zorder() == 6
        ]
        (clock,) = [
            t.get_window_extent().padded(pad)
            for t in plane.texts
            if t.get_zorder() == 10
        ]
        to_display = plane.transData.transform
        cycle = to_display(
            np.asarray(PANEL_2.limit_cycle(jnp.linspace(0.0, 1.0, 20000)))
        )
        radii = jnp.linspace(_INNER_RADIUS, style.box * np.sqrt(2), 200000)
        isochrons = [
            to_display(np.asarray(PANEL_2.isochron(phase, radii))) for phase in PHASES
        ]
        markers = to_display(
            np.concatenate(
                [line.get_xydata() for line in plane.lines if line.get_zorder() == 5]
            )
        )
        plane_box = plane.get_window_extent()
        plt.close("all")
    assert len(labels) == len(PHASES)
    for i, box in enumerate(labels):
        assert plane_box.x0 <= box.x0 and box.x1 <= plane_box.x1
        assert plane_box.y0 <= box.y0 and box.y1 <= plane_box.y1
        assert not box.overlaps(clock)
        assert not any(box.overlaps(other) for j, other in enumerate(labels) if j != i)
        clearance = min(
            float(_box_gap(box, cycle).min()) - style.cycle_width * px / 2,
            *(
                float(_box_gap(box, c).min()) - style.trace_width * px / 2
                for c in isochrons
            ),
            float(_box_gap(box, markers).min()) - style.point_size * px / 2,
        )
        assert clearance >= _MIN_LABEL_CLEARANCE_PX


@pytest.mark.parametrize(
    ("system", "arguments", "message"),
    [
        (PANEL_2, dict(phases=np.array([0.0, 1.0])), r"phases \[1\.0\]"),
        (PANEL_2, dict(n_periods=0), "n_periods"),
        (PANEL_2, dict(phases=np.array([0.0, 0.3, 0.6])), "group_colors"),
        (
            WINFREE_HOLE,
            dict(radii=np.array([0.1, 1.0])),
            r"\(0\.0, 0\.1\), \(0\.5, 0\.1\)",
        ),
    ],
    ids=["phases", "n_periods", "group_colors", "basin"],
)
def test_animate_isochron_sync_raises(
    system: AbstractNormalForm, arguments: dict[str, object], message: str
):
    """A phase outside [0, 1), no period, more phases than group colors, and a seed
    outside the basin each raise `ValueError`; the first names the phase and the last
    the phase and the radius of each seed."""
    defaults = dict(phases=PHASES, radii=RADII, n_periods=5)
    with pytest.raises(ValueError, match=message):
        animate_isochron_sync(system, **(defaults | arguments))


def _field_grid() -> Float[np.ndarray, " n"]:
    """The grid of each axis that `animate_phase_field` paints in the default style."""
    box = PortraitStyle().box
    return np.linspace(-box, box, _FIELD_RESOLUTION)


def test_field_scene_masks_match_the_closed_form():
    """After each backward step n the painted band is `|Ψ| ≤ tolerance · e^n` pixel for
    pixel, since the panel's Floquet multiplier is e^{-1}, and the lap-1 band with the
    point at phase 0.6 is `(|Ψ| ≤ tolerance) & (Θ ≤ 0.6)`; Θ and Ψ are the closed-form
    phase–amplitude coordinates of each grid point."""
    tolerance, n_periods = 0.05, 6
    grid = _field_grid()
    scene = _field_scene(PANEL_2, tolerance=tolerance, n_periods=n_periods, grid=grid)
    xs, ys = np.meshgrid(grid, grid)
    points = jnp.asarray(np.stack((xs.ravel(), ys.ravel()), axis=1))
    theta, psi = np.asarray(jax.vmap(PANEL_2.to_phase_amplitude)(points)).T
    theta, psi = theta.reshape(xs.shape), psi.reshape(xs.shape)
    for n in range(n_periods + 1):
        expected = np.abs(psi) <= tolerance * np.exp(n)
        np.testing.assert_array_equal(np.asarray(scene.step_mask(n)), expected)
    np.testing.assert_array_equal(
        np.asarray(scene.lap_mask(0.6)), (np.abs(psi) <= tolerance) & (theta <= 0.6)
    )


def test_field_lap_colors_follow_the_clock():
    """At every frame of lap 1, the point sits on the cycle at the clock's phase, by the
    closed-form phase, and the painted pixel nearest it is behind it: its phase is at
    most the clock's, and it lies within one grid diagonal of the point."""
    frames_per_period = PortraitStyle().frames_per_period
    grid = _field_grid()
    scene = _field_scene(PANEL_2, tolerance=0.05, n_periods=1, grid=grid)
    xs, ys = np.meshgrid(grid, grid)
    phase = np.asarray(scene.phase)
    diagonal = np.sqrt(2) * (grid[1] - grid[0])
    for j in range(1, frames_per_period + 1):
        clock = j / frames_per_period
        point = PANEL_2.limit_cycle(jnp.asarray(clock))
        np.testing.assert_allclose(
            phase_difference(PANEL_2.phase(point), clock),
            0.0,
            rtol=0.0,
            atol=TOL["closed_form"],
        )
        x, y = np.asarray(point)
        distance = np.where(
            np.asarray(scene.lap_mask(clock)), np.hypot(xs - x, ys - y), np.inf
        )
        nearest = np.unravel_index(np.argmin(distance), distance.shape)
        assert distance[nearest] <= diagonal
        assert phase[nearest] <= clock


def test_animate_phase_field_writes_every_frame(tmp_path: Path):
    """The animation saved as a GIF under the project style plays two laps of 40 frames
    and their first frame, the reset and six steps of `_FADE_FRAMES` frames each, and
    the final frame; with the 1.5-second start pause, the holds of `_STEP_SECONDS`
    after lap 2, after the reset and after each step, and the 3-second final hold,
    counted by duration as in `test_animate_kick_writes_every_frame`."""
    path = tmp_path / "field.gif"
    style = PortraitStyle()
    n_periods = 6
    with matplotlib.rc_context():
        configure_rcparams()
        animation = animate_phase_field(
            PANEL_2, tolerance=0.05, n_periods=n_periods, style=style
        )
        animation.save(path, writer=PillowWriter(fps=style.fps), dpi=style.dpi)
        plt.close("all")
    with Image.open(path) as gif:
        size = gif.size
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    frames = (
        2 * style.frames_per_period
        + 1
        + (n_periods + 1) * _FADE_FRAMES
        + 1
        + round(style.pause_seconds * style.fps)
        + (n_periods + 2) * round(_STEP_SECONDS * style.fps)
        + round(style.hold_seconds * style.fps)
    )
    assert sum(durations) == frames * 1000 // style.fps
    assert size == (960, 960)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (dict(tolerance=0.0), "tolerance"),
        (dict(n_periods=0), "n_periods"),
        (dict(style=PortraitStyle(figsize_square=(6.0, 5.0))), "not square"),
    ],
    ids=["tolerance", "n_periods", "figsize_square"],
)
def test_animate_phase_field_raises(arguments: dict[str, object], message: str):
    """A tolerance that is not positive, no backward step, and a figure that is not
    square each raise `ValueError`."""
    defaults = dict(tolerance=0.05, n_periods=6)
    with pytest.raises(ValueError, match=message):
        animate_phase_field(PANEL_2, **(defaults | arguments))
