"""The figures of ``plotting.isochrons`` build on a hand-made foliation of Hopf's cycle
and a model conjugate to the same normal form by the identity, with the expected
panels and the numbers they report; nothing is compared pixel by pixel, so that an API
change breaks the figure here and not in a notebook.
"""

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.model import ConjugateLatentDynamics
from deep_isochron.model.invertible import BiLipschitzLinear
from deep_isochron.plotting import (
    plot_isochron_hairpins,
    plot_isochron_match,
    plot_phase_comparison,
    plot_phase_field,
)

from .helpers import hand_foliation, HAND_FOLIATION_FORM, TOL


matplotlib.use("Agg")


@pytest.fixture(scope="module")
def foliation():
    """`hand_foliation`, once per module."""
    return hand_foliation()


@pytest.fixture(scope="module")
def model():
    """The conjugacy model of the foliation's normal form through an identity
    bijection: its phase is the normal form's."""
    return ConjugateLatentDynamics(
        HAND_FOLIATION_FORM,
        BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0)),
    )


@pytest.fixture(scope="module")
def field():
    """The normal form's phase on a 12×12 grid over ``[-2, 2]²``, ``nan`` within 0.2
    of the origin: ``(xs, ys, field)``."""
    xs = ys = np.linspace(-2.0, 2.0, 12)
    grid = jnp.asarray(np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2))
    phase = np.asarray(jax.vmap(HAND_FOLIATION_FORM.phase)(grid)).reshape(12, 12)
    masked = np.hypot(*np.meshgrid(xs, ys)) < 0.2
    return xs, ys, np.where(masked, np.nan, phase)


def test_phase_field_panel_draws_the_isochrons_and_the_cycle(field, foliation):
    """One image, one line per drawn isochron, the cycle and its origin."""
    xs, ys, phase = field
    fig, ax = plt.subplots()
    image = plot_phase_field(
        ax, xs, ys, phase, foliation.cycle, foliation.isochrons, every=1
    )
    assert image in ax.collections
    assert len(ax.lines) == len(foliation.isochrons) + 2
    plt.close(fig)


def test_phase_comparison_aligns_the_identity_model_exactly(field, foliation, model):
    """Three panels; the identity model aligns with sense +1, offset 0 and a residual
    at round-off, and every isochron is a level set of its phase."""
    xs, ys, phase = field
    fig, result = plot_phase_comparison(
        xs, ys, phase, foliation.cycle, model, foliation.isochrons, name="hopf"
    )
    assert len([ax for ax in fig.axes if ax.get_title()]) == 3
    assert result.alignment.sense == 1.0
    assert abs(result.alignment.offset) < TOL["limit_cycle"]
    assert result.alignment.rms < TOL["limit_cycle"]
    assert result.isochron_gaps.shape == (2,) and result.isochron_gaps.max() < 1e-9
    plt.close(fig)


def test_isochron_match_with_and_without_a_model(foliation, model):
    """Without a model: one line per isochron plus the cycle; with the identity
    model: a dashed learned isochron per reference one and gaps at round-off."""
    fig, gaps = plot_isochron_match(foliation)
    assert gaps.shape == (0,)
    assert len(fig.axes[0].lines) == len(foliation.isochrons) + 2
    plt.close(fig)
    fig, gaps = plot_isochron_match(foliation, model)
    assert len(fig.axes[0].lines) == 2 * len(foliation.isochrons) + 3
    assert gaps.max() < 1e-9
    plt.close(fig)


def test_isochron_hairpins_finds_the_origin_and_matches_the_identity_model(
    foliation, model
):
    """Five panels; the equilibrium is the origin, the model's too, the radial
    isochrons have no hairpins, and the matched distance is at round-off."""
    fig, match = plot_isochron_hairpins(foliation, model, index=1, name="hopf")
    assert len(fig.axes) >= 5
    assert np.linalg.norm(match.equilibrium) < TOL["limit_cycle"]
    assert np.linalg.norm(match.model_equilibrium) < TOL["identity"]
    assert match.index == 1 and match.hairpins.size == 0
    assert match.matched_distance.max() < 1e-9
    plt.close(fig)


def test_isochron_log_polar_and_hairpins(foliation):
    """A radial isochron has constant angle about the origin and no hairpins; a curve
    that reverses its angle by a quarter turn has one."""
    iso = foliation.isochrons[1]
    log_r, turns = iso.log_polar(np.zeros(2))
    assert np.allclose(np.diff(turns), 0.0)
    assert np.allclose(10**log_r, np.hypot(*iso.points.T))
    assert iso.hairpins(np.zeros(2)).size == 0
    angles = np.concatenate(
        [np.linspace(0, np.pi / 2, 20), np.linspace(np.pi / 2, 0, 20)[1:]]
    )
    bent = iso.__class__(
        phase=iso.phase,
        base_point=iso.base_point,
        points=np.column_stack([np.cos(angles), np.sin(angles)]),
        eta=np.zeros_like(angles),
        arclength=np.zeros_like(angles),
        stop=iso.stop,
    )
    assert list(bent.hairpins(np.zeros(2))) == [19]
