"""The animations of ``plotting.phase_portrait`` on the normal form of the talk.

The phase shift that `animate_kick` labels is pinned against an independent oracle: the
polar angle of each kicked copy twelve periods after the kick, when the kick has relaxed
onto the cycle. The animation is pinned by the frames of a saved GIF, its layout by
the edges of its two panels, and its arguments by the four errors it raises. The figure
tests run under the project style, as the notebooks do.

The points of `animate_isochron_sync` are pinned by the closed-form phase: it equals the
seeded phase at the start and the phase of the group's drawn isochron at every frame.
Its animation is pinned by the frames of a saved GIF, and its arguments by the four
errors it raises.
"""

from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.plotting import (
    animate_isochron_sync,
    animate_kick,
    configure_rcparams,
    PortraitStyle,
)
from deep_isochron.plotting.phase_portrait.drawing import _SAMPLES_PER_FRAME
from deep_isochron.plotting.phase_portrait.kick import _kick_scene
from deep_isochron.plotting.phase_portrait.sync import _sync_scene
from deep_isochron.systems.normal_forms import (
    AbstractNormalForm,
    BautinNormalForm,
    WinfreeNormalForm,
)
from deep_isochron.utils.numerics import phase_difference
from matplotlib.animation import PillowWriter
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
