"""The figures of ``plotting.isochrons`` build on a hand-made foliation of Hopf's cycle
and a model conjugate to the same normal form by the identity, with the expected
panels and the numbers they report; nothing is compared pixel by pixel, so that an API
change breaks the figure here and not in a notebook.

The data half of ``plotting.deformation`` is pinned by laws: a bad config raises, a
sheet mapped by the identity is itself, the area and anisotropy metrics of the identity
are zero and those of a linear map are the logarithms of its determinant and condition
number by NumPy's SVD of the matrix, the stages and stage maps of a two-layer model end
at the model's own map in both directions, and the limits are the padded union box.
``plotting.scales.clipped_norm`` is pinned on hand-made values with literal ends.
"""

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.model import ConjugateLatentDynamics
from deep_isochron.model.invertible import (
    BiLipschitzLinear,
    CouplingFlow,
    MonotonicRQSpline,
    SequentialINN,
)
from deep_isochron.plotting import (
    plot_isochron_hairpins,
    plot_isochron_match,
    plot_phase_comparison,
    plot_phase_field,
)
from deep_isochron.plotting.deformation import (
    ANISOTROPY,
    AREA,
    Sheet,
    sheet_limits,
    SheetConfig,
    stage_labels,
    stage_maps,
    stage_sheets,
)
from deep_isochron.plotting.scales import clipped_norm
from hypothesis import given
from jaxtyping import Float

from .helpers import hand_foliation, HAND_FOLIATION_FORM, TOL
from .strategies import perturb, planar_matrices


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


@pytest.fixture(scope="module")
def square_sheet() -> Sheet:
    """A sheet of 8×8 cells on $[-1, 1]^2$ with two synthetic trajectories, a spiral
    arc and a straight segment, and one extra curve of another length, an arc of 30
    samples."""
    t = np.linspace(0.0, 1.0, 50)
    spiral = 0.8 * t[:, None] * np.stack([np.cos(6 * t), np.sin(6 * t)], axis=-1)
    segment = np.stack([t - 0.5, 0.5 * t - 0.9], axis=-1)
    s = np.linspace(0.0, 2.0, 30)
    arc = 0.6 * np.stack([np.cos(s), np.sin(s)], axis=-1)
    return Sheet(
        (-1.0, -1.0, 1.0, 1.0),
        np.stack([spiral, segment]),
        curves=(arc,),
        config=SheetConfig(cells=8),
    )


@pytest.fixture(scope="module")
def layers() -> tuple[BiLipschitzLinear, CouplingFlow]:
    """The two layers of ``configs/model/conjugacy.yaml``, perturbed by `perturb` so
    that neither is the identity it is at initialization."""
    k_linear, k_coupling, k_perturb = jax.random.split(jax.random.key(3), 3)
    linear = BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=k_linear)
    coupling = CouplingFlow(
        dim=2,
        bijection=MonotonicRQSpline(num_bins=9, xy_range=(-4.0, 4.0)),
        key=k_coupling,
    )
    k_a, k_b = jax.random.split(k_perturb)
    return perturb(linear, k_a), perturb(coupling, k_b)


def _assert_sheets_close(actual: Sheet, desired: Sheet, tol: float) -> None:
    """Every array leaf of `actual`, the extra curves included, within `tol` of the
    same leaf of `desired`, and the same config."""
    assert actual.config == desired.config
    actual_leaves, desired_leaves = jax.tree.leaves(actual), jax.tree.leaves(desired)
    assert len(actual_leaves) == len(desired_leaves)
    for index, (a, d) in enumerate(zip(actual_leaves, desired_leaves)):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(d), rtol=tol, atol=tol, err_msg=f"leaf {index}"
        )


def test_sheet_config_rejects_bad_values():
    """One value out of range per validated field raises and names the field."""
    bad = {
        "cells": 0,
        "subdivisions": 0,
        "line_samples": 1,
        "circle_every": 0,
        "circle_radius": 0.0,
        "circle_samples": 2,
        "box_pad": -0.1,
    }
    for name, value in bad.items():
        with pytest.raises(ValueError, match=name):
            SheetConfig(**{name: value})


def test_sheet_rejects_a_bad_box():
    """A box of three entries, or one empty along $x$ or along $y$, raises."""
    trajectories = np.zeros((1, 2, 2))
    for box in [(-1.0, -1.0, 1.0), (1.0, -1.0, 1.0, 1.0), (-1.0, 1.0, 1.0, 0.0)]:
        with pytest.raises(ValueError):
            Sheet(box, trajectories)


def test_sheet_map_identity_is_the_sheet(square_sheet):
    """The identity maps every leaf of a sheet to itself exactly, the ragged extra
    curve included, and carries the config."""
    mapped = square_sheet.map(lambda x: x)
    assert len(mapped.curves) == 1
    _assert_sheets_close(mapped, square_sheet, 0.0)


def test_area_and_anisotropy_of_the_identity_are_zero(square_sheet):
    """Under the identity every cell keeps its area and its shape, exactly."""
    for metric in (AREA, ANISOTROPY):
        values = np.asarray(metric.fn(lambda x: x, square_sheet.centers))
        assert values.shape == (8, 8)
        assert np.all(values == 0.0), metric.label


@given(matrix=planar_matrices(bound=2.0, min_det=0.1))
def test_area_and_anisotropy_of_a_linear_map(
    square_sheet, matrix: Float[np.ndarray, "2 2"]
):
    r"""Under $x \mapsto A x$ every cell has log area $\log \det A$ and log
    anisotropy $\log(\sigma_{\max} / \sigma_{\min})$, against NumPy's determinant and
    SVD of $A$."""
    a = jnp.asarray(matrix)
    log_area = AREA.fn(lambda x: a @ x, square_sheet.centers)
    log_anisotropy = ANISOTROPY.fn(lambda x: a @ x, square_sheet.centers)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tol = TOL["cell_metric_affine"]
    np.testing.assert_allclose(
        log_area, np.log(np.linalg.det(matrix)), rtol=0.0, atol=tol
    )
    np.testing.assert_allclose(
        log_anisotropy, np.log(singular[0] / singular[1]), rtol=0.0, atol=tol
    )


def test_area_and_anisotropy_of_a_reflection_are_zero(square_sheet):
    r"""The reflection $\operatorname{diag}(1, -1)$ reverses orientation and keeps every
    area and shape: log area 0 from $|\det J| = 1$, and log anisotropy 0."""
    reflection = jnp.diag(jnp.array([1.0, -1.0]))
    tol = TOL["cell_metric_affine"]
    for metric in (AREA, ANISOTROPY):
        values = metric.fn(lambda x: reflection @ x, square_sheet.centers)
        np.testing.assert_allclose(
            values, 0.0, rtol=0.0, atol=tol, err_msg=metric.label
        )


def test_stage_sheets_count_and_endpoints(square_sheet, layers):
    """In both directions: one sheet per layer plus the start, the start unchanged, and
    the last equal to the sheet mapped by the composed model, or by its inverse, which
    moves the sheet."""
    model = SequentialINN(layers)
    for inverse, whole in [(False, model), (True, model.inverse)]:
        stages = stage_sheets(layers, square_sheet, inverse=inverse)
        assert len(stages) == len(layers) + 1
        assert stages[0] is square_sheet
        _assert_sheets_close(stages[-1], square_sheet.map(whole), TOL["jit_eager"])
        assert not np.allclose(stages[-1].nodes, square_sheet.nodes)


def test_stage_maps_match_stage_sheets(square_sheet, layers):
    """In both directions, the sheet mapped by stage map `k` is stage sheet `k`, for
    every stage of a two-layer model."""
    for inverse in (False, True):
        maps = stage_maps(layers, inverse=inverse)
        stages = stage_sheets(layers, square_sheet, inverse=inverse)
        assert len(maps) == len(stages) == 3
        for k in range(3):
            _assert_sheets_close(square_sheet.map(maps[k]), stages[k], TOL["jit_eager"])


def test_stage_labels(layers):
    """The identity first, then each layer's class in the order applied, marked
    inverted when the stages run backward."""
    assert stage_labels(layers) == (
        "stage 0 / 2 — identity",
        "stage 1 / 2 — BiLipschitzLinear (layer 1)",
        "stage 2 / 2 — CouplingFlow (layer 2)",
    )
    assert stage_labels(layers, inverse=True) == (
        "stage 0 / 2 — identity",
        "stage 1 / 2 — CouplingFlow⁻¹ (layer 1)",
        "stage 2 / 2 — BiLipschitzLinear⁻¹ (layer 2)",
    )


def test_sheet_limits_pad_the_union(square_sheet):
    r"""A sheet and its translate by $(3, -1)$ span $[-1, 4] \times [-2, 1]$, padded by
    5 % of that extent on each side, as four floats."""
    shifted = square_sheet.map(lambda x: x + jnp.array([3.0, -1.0]))
    limits = sheet_limits([square_sheet, shifted])
    assert len(limits) == 4 and all(isinstance(v, float) for v in limits)
    np.testing.assert_allclose(
        limits,
        [-1.0 - 0.25, -2.0 - 0.15, 4.0 + 0.25, 1.0 + 0.15],
        rtol=0.0,
        atol=TOL["identity"],
    )


def test_clipped_norm_symmetric_and_excludes_zeros():
    """At quantile 1 the range is the extreme of the pooled entries, with an identity
    stage's zeros left out: symmetric about 0 at the largest magnitude, otherwise the
    smallest and largest entry; an empty pool gives the default range."""
    zeros = np.zeros((4, 4))
    spread = jnp.linspace(-2.0, 1.0, 16).reshape(4, 4)
    symmetric = clipped_norm([zeros, spread], 1.0, symmetric=True)
    assert (symmetric.vmin, symmetric.vmax) == (-2.0, 2.0) and symmetric.clip
    stretched = np.linspace(0.1, 1.5, 16).reshape(4, 4)
    sequential = clipped_norm([zeros, stretched], 1.0, symmetric=False)
    assert (sequential.vmin, sequential.vmax) == (0.1, 1.5) and sequential.clip
    empty_symmetric = clipped_norm(zeros, 0.98, symmetric=True)
    assert (empty_symmetric.vmin, empty_symmetric.vmax) == (-1.0, 1.0)
    empty = clipped_norm(zeros, 0.98, symmetric=False)
    assert (empty.vmin, empty.vmax) == (0.0, 1.0)


def test_clipped_norm_rejects_a_bad_quantile():
    """A quantile at or below 0.5 leaves no range between the two ends, and one above
    1 is no quantile."""
    for quantile in (0.4, 1.5):
        with pytest.raises(ValueError, match=str(quantile)):
            clipped_norm(np.ones((2, 2)), quantile, symmetric=False)
