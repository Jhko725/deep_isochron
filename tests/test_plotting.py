"""The figures of ``plotting.isochrons`` build on a hand-made foliation of Hopf's cycle
and a model conjugate to the same normal form by the identity, with the expected
panels and the numbers they report; nothing is compared pixel by pixel, so that an API
change breaks the figure here and not in a notebook.

The geometry of ``plotting.sheet`` is pinned by laws: a sheet mapped by the
identity is itself, the cell values of the identity are zero, the cell values of a
linear map are the logarithms of its determinant and condition number by NumPy's SVD of
the matrix, a folded cell raises with the count, the stages of a two-layer model end at
the model's own map in both directions, and the limits are the padded union box. The
fill scales of ``plotting.deformation`` are pinned on hand-made cell values, and its
pair figure by its panels.
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
    anisotropy_scale,
    area_scale,
    plot_deformation,
)
from deep_isochron.plotting.sheet import (
    build_sheet,
    cell_values,
    CellValues,
    Sheet,
    sheet_limits,
    stage_labels,
    stage_sheets,
)
from hypothesis import given
from jaxtyping import Float
from matplotlib.collections import QuadMesh

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
    """A sheet of 8×8 cells on $[-1, 1]^2$ with two synthetic curves, a spiral arc and
    a straight segment."""
    t = np.linspace(0.0, 1.0, 50)
    spiral = 0.8 * t[:, None] * np.stack([np.cos(6 * t), np.sin(6 * t)], axis=-1)
    segment = np.stack([t - 0.5, 0.5 * t - 0.9], axis=-1)
    return build_sheet(
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        np.stack([spiral, segment]),
        cells=8,
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
    """Every field of `actual` within `tol` of the same field of `desired`."""
    for name in ("nodes", "fill_nodes", "lines", "circles", "curves"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(desired, name),
            rtol=tol,
            atol=tol,
            err_msg=name,
        )


def test_sheet_map_identity_is_the_sheet(square_sheet):
    """The identity maps every field of a sheet to itself, exactly."""
    mapped = square_sheet.map(lambda x: x)
    _assert_sheets_close(mapped, square_sheet, 0.0)


def test_cell_values_of_the_identity_are_zero(square_sheet):
    """Under the identity every cell keeps its area and its shape, exactly."""
    values = cell_values(square_sheet, square_sheet.map(lambda x: x))
    assert np.all(values.log_area == 0.0)
    assert np.all(values.log_anisotropy == 0.0)


@given(matrix=planar_matrices(bound=2.0, min_det=0.1))
def test_cell_values_of_a_linear_map_are_its_determinant_and_condition(
    square_sheet, matrix: Float[np.ndarray, "2 2"]
):
    r"""Under $x \mapsto A x$ every cell has log area $\log \det A$ and log
    anisotropy $\log(\sigma_{\max} / \sigma_{\min})$, against NumPy's determinant and
    SVD of $A$."""
    a = jnp.asarray(matrix)
    values = cell_values(square_sheet, square_sheet.map(lambda x: a @ x))
    singular = np.linalg.svd(matrix, compute_uv=False)
    tol = TOL["cell_values_affine"]
    np.testing.assert_allclose(
        values.log_area, np.log(np.linalg.det(matrix)), rtol=0.0, atol=tol
    )
    np.testing.assert_allclose(
        values.log_anisotropy, np.log(singular[0] / singular[1]), rtol=0.0, atol=tol
    )


def test_cell_values_raise_on_a_folded_cell():
    r"""$(x, y) \mapsto (x^2, y)$ reverses every cell left of $x = 0$, and the error
    names their count: half of the $4 \times 4$ cells."""
    sheet = build_sheet(
        np.array([-1.0, -1.0]), np.array([1.0, 1.0]), np.zeros((1, 2, 2)), cells=4
    )
    folded = sheet.map(lambda x: jnp.stack([x[0] ** 2, x[1]]))
    with pytest.raises(ValueError, match=r"\b8\b"):
        cell_values(sheet, folded)


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


def test_stage_labels(layers):
    """The identity first, then each layer's class in the order applied."""
    assert stage_labels(layers) == (
        "stage 0 / 2 — identity",
        "stage 1 / 2 — BiLipschitzLinear (layer 1)",
        "stage 2 / 2 — CouplingFlow (layer 2)",
    )
    assert stage_labels(layers, inverse=True) == (
        "stage 0 / 2 — identity",
        "stage 1 / 2 — CouplingFlow (layer 1)",
        "stage 2 / 2 — BiLipschitzLinear (layer 2)",
    )


def test_sheet_limits_pad_the_union(square_sheet):
    r"""A sheet and its translate by $(3, -1)$ span $[-1, 4] \times [-2, 1]$, padded by
    5 % of that extent on each side."""
    shifted = square_sheet.map(lambda x: x + jnp.array([3.0, -1.0]))
    lo, hi = sheet_limits([square_sheet, shifted])
    np.testing.assert_allclose(
        lo, [-1.0 - 0.25, -2.0 - 0.15], rtol=0.0, atol=TOL["identity"]
    )
    np.testing.assert_allclose(
        hi, [4.0 + 0.25, 1.0 + 0.15], rtol=0.0, atol=TOL["identity"]
    )


def test_area_scale_is_symmetric_and_ticks_lie_inside():
    r"""On log areas spread over $[-2, 1]$ the norm is symmetric about 0, at the 98th
    percentile of $|\log a|$ (about 1.94), so the factors 0.3, 1 and 3 are ticked and
    0.1 and 10 are not."""
    log_area = np.linspace(-2.0, 1.0, 25).reshape(5, 5)
    scale = area_scale([CellValues(log_area, np.zeros_like(log_area))])
    assert scale.norm.vmin == -scale.norm.vmax
    assert all(scale.norm.vmin <= t <= scale.norm.vmax for t in scale.ticks)
    assert scale.tick_labels == ("×0.3", "×1", "×3")


def test_anisotropy_scale_excludes_zeros():
    """Pooled over an identity stage and a stretched one, the scale spans the 2nd to
    98th percentile of the stretched stage alone: the identity's zeros are left out."""
    stretched = np.linspace(0.1, 1.5, 16).reshape(4, 4)
    identity = CellValues(np.zeros((4, 4)), np.zeros((4, 4)))
    scale = anisotropy_scale([identity, CellValues(np.zeros((4, 4)), stretched)])
    assert scale.norm.vmin == np.percentile(stretched, 2)
    assert scale.norm.vmax == np.percentile(stretched, 98)


def test_scales_reject_a_bad_percentile():
    """A percentile at or below 50 leaves no range between the two ends, in both
    scales."""
    values = CellValues(np.ones((2, 2)), np.ones((2, 2)))
    for builder in (area_scale, anisotropy_scale):
        with pytest.raises(ValueError, match="40"):
            builder([values], percentile=40.0)


def test_plot_deformation_draws_both_panels(square_sheet, layers):
    """Two panels and a colorbar; the domain panel holds one fill and one halo'd line
    per trajectory."""
    deformed = square_sheet.map(SequentialINN(layers))
    values = cell_values(square_sheet, deformed)
    fig = plot_deformation(
        square_sheet, deformed, values.log_area, area_scale([values])
    )
    assert len(fig.axes) == 3
    domain = fig.axes[0]
    assert len([c for c in domain.collections if isinstance(c, QuadMesh)]) == 1
    halos = [line for line in domain.lines if line.get_path_effects()]
    assert len(halos) == len(square_sheet.curves)
    plt.close(fig)
