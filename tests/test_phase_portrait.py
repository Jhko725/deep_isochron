"""The animations of ``plotting.phase_portrait`` on the normal form of the talk.

The phase shift that `animate_kick` labels is pinned against an independent oracle: the
polar angle of each kicked copy twelve periods after the kick, when the kick has relaxed
onto the cycle. The animation is pinned by the frames of a saved GIF, and its arguments
by the four errors it raises.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.plotting import animate_kick
from deep_isochron.plotting.phase_portrait import _kick_scene
from deep_isochron.systems.normal_forms import (
    AbstractNormalForm,
    BautinNormalForm,
    WinfreeNormalForm,
)
from matplotlib.animation import PillowWriter
from PIL import Image, ImageSequence

from .helpers import TOL


# Panel 2 of the contact sheet: T = 1 and μ = e^{-1} = 0.37, so a kick relaxes visibly
PANEL_2 = BautinNormalForm(a=1 / 3, b=0.5, w=2 * np.pi, w0=2 * np.pi - 2 / 3)
# Winfree's model with the paper's hole: the disk r <= 0.25 is outside the basin
WINFREE_HOLE = WinfreeNormalForm(a=0.25, w=-0.5)
KICKS = np.array([[0.75, 0.0], [-0.75, 0.0]])
"""The default scene of the talk: a kick to the right and one to the left."""


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
    """The animation saved as a GIF plays 4 × 40 + 1 frames, the 30 frames of the
    1.5-second kick pause and the 60 frames of the 3-second final hold. Pillow merges
    identical consecutive frames into one of longer duration, so the frames are counted
    by duration, 50 ms each at 20 fps."""
    animation = animate_kick(PANEL_2, kick_phase=0.83, kicks=KICKS, n_periods=4)
    path = tmp_path / "kick.gif"
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close("all")
    with Image.open(path) as gif:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    assert sum(durations) == (4 * 40 + 1 + round(1.5 * 20) + round(3.0 * 20)) * 50


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
