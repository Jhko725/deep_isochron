"""The figures of ``plotting.isochrons`` build on a hand-made foliation of Hopf's cycle
and a model conjugate to the same normal form by the identity, with the expected
panels and the numbers they report; nothing is compared pixel by pixel, so that an API
change breaks the figure here and not in a notebook.

The data half of ``plotting.deformation`` is pinned by laws: a bad config raises, a
sheet mapped by the identity is itself, the area and anisotropy metrics of the identity
are zero and those of a linear map are the logarithms of its determinant and condition
number by NumPy's SVD of the matrix, the stages and stage maps of a two-layer model end
at the model's own map in both directions, and the limits are the padded union box.
``plotting.scales.clipped_norm`` is pinned on hand-made values with literal ends. The
deformation figures are pinned by their panels, the strip's stage selection and range
check, the animation's frames through a saved GIF, and ``draw_sheet`` by a custom
metric's fill; ``plotting.style.set_limits`` by its length check. The drawers of
``plotting.schematic`` are pinned by their start markers' fill and their line styles,
and its style and labels by a range check and ``copy.replace``. The hand-set layers
of ``plotting.coupling_warp`` are pinned by their round trip and the spline's steep
band, its trajectories by fitting the box, its drawers by the knot markers, the warp
panels' spines and ticks, the fixed grid lines staying vertical and one start triangle
per trajectory, its style and labels as the schematic's, and its figure by its six
panels, their shared limits and the headers. ``draw_trajectories`` is pinned by one
start triangle per trajectory. The
hand-set linear layer of ``plotting.inn_stack`` is pinned by NumPy's SVD of its matrix
and its round trip, its style and labels as the schematic's, and its figure by its
blocks, arrows, labels and inset axes.
"""

import copy
from collections.abc import Callable

import colorcet as cc
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
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
    add_colorbar,
    draw_data,
    draw_grid_sheet,
    draw_identity_warp,
    draw_latent_initial,
    draw_latent_trajectories,
    draw_prediction,
    draw_warp_curve,
    plot_coupling_warps,
    plot_inn_stack,
    plot_isochron_hairpins,
    plot_isochron_match,
    plot_phase_comparison,
    plot_phase_field,
    schematic_affine_coupling,
    schematic_linear,
    schematic_sheet,
    schematic_spline_coupling,
    schematic_trajectories,
    SchematicLabels,
    SchematicStyle,
    set_limits,
    StackLabels,
    StackStyle,
    WarpLabels,
    WarpStyle,
)
from deep_isochron.plotting.deformation import (
    animate_deformation,
    ANISOTROPY,
    AREA,
    CellMetric,
    draw_sheet,
    draw_trajectories,
    plot_deformation,
    plot_deformation_strip,
    Sheet,
    sheet_limits,
    SheetConfig,
    SheetStyle,
    stage_labels,
    stage_maps,
    stage_sheets,
)
from deep_isochron.plotting.deformation.figures import _factor_ticks
from deep_isochron.plotting.deformation.sheet import _stage_sequence
from deep_isochron.plotting.scales import clipped_norm
from deep_isochron.plotting.style import curve_colors
from hypothesis import given
from jaxtyping import Array, Float
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle, FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from PIL import Image

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
def trajectories() -> Float[np.ndarray, "2 50 2"]:
    """Two synthetic trajectories in $[-1, 1]^2$: a spiral arc and a straight
    segment."""
    t = np.linspace(0.0, 1.0, 50)
    spiral = 0.8 * t[:, None] * np.stack([np.cos(6 * t), np.sin(6 * t)], axis=-1)
    segment = np.stack([t - 0.5, 0.5 * t - 0.9], axis=-1)
    return np.stack([spiral, segment])


@pytest.fixture(scope="module")
def square_sheet(trajectories: Float[np.ndarray, "2 50 2"]) -> Sheet:
    """A sheet of 8×8 cells on $[-1, 1]^2$ with the two trajectories and one extra
    curve of another length, an arc of 30 samples."""
    s = np.linspace(0.0, 2.0, 30)
    arc = 0.6 * np.stack([np.cos(s), np.sin(s)], axis=-1)
    return Sheet(
        (-1.0, -1.0, 1.0, 1.0),
        trajectories,
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


def test_plot_deformation_draws_both_panels(layers, trajectories):
    """At the default keywords: the domain and image panels and a colorbar, the domain
    panel holding one fill and one halo'd line per trajectory."""
    fig = plot_deformation(layers, trajectories)
    assert len(fig.axes) == 3
    domain, image = fig.axes[:2]
    assert (domain.get_title(), image.get_title()) == ("domain", "image")
    assert len([c for c in domain.collections if isinstance(c, QuadMesh)]) == 1
    halos = [line for line in domain.lines if line.get_path_effects()]
    assert len(halos) == len(trajectories)
    assert fig.axes[2].get_title(loc="right") == AREA.label
    plt.close(fig)


def test_add_colorbar_label_placement():
    """With `label_above` the label is the colorbar's horizontal title, aligned to its
    right edge; without it, the rotated label at the bar's side, and the title is
    empty."""
    for label_above in (True, False):
        fig, ax = plt.subplots()
        image = ax.imshow(np.zeros((2, 2)))
        add_colorbar(
            fig,
            [ax],
            image,
            label="metric",
            ticks=[0.0],
            tick_labels=["0"],
            shrink=0.7,
            aspect=25.0,
            pad=0.02,
            tick_fontsize=9.0,
            label_above=label_above,
        )
        bar = fig.axes[-1]
        expected = ("metric", "") if label_above else ("", "metric")
        assert (bar.get_title(loc="right"), bar.get_ylabel()) == expected
        plt.close(fig)


def test_factor_ticks_rule():
    """A wide range is ticked by the 1–3–10 series, a range with fewer than three of
    those by the 1–2–5 series, and a range with fewer than two of those by at least two
    located ticks inside it; each tick sits at its factor's logarithm, labeled with a
    `×` only for a symmetric metric."""
    ticks, labels = _factor_ticks(Normalize(-2.5, 2.5), symmetric=True)
    assert labels == ("×0.1", "×0.3", "×1", "×3", "×10")
    np.testing.assert_allclose(
        ticks, np.log([0.1, 0.3, 1.0, 3.0, 10.0]), rtol=0.0, atol=TOL["identity"]
    )
    ticks, labels = _factor_ticks(Normalize(-0.8, 0.8), symmetric=True)
    assert labels == ("×0.5", "×1", "×2")
    _, labels = _factor_ticks(Normalize(-0.8, 0.8), symmetric=False)
    assert labels == ("0.5", "1", "2")
    ticks, labels = _factor_ticks(Normalize(-0.2, 0.2), symmetric=True)
    assert len(ticks) >= 2 and len(labels) == len(ticks)
    assert all(-0.2 <= t <= 0.2 for t in ticks)


def test_inverse_sequence_is_the_forward_reversed(square_sheet, layers):
    """Inverted, the stage sequence is the forward sheets in reverse order, leaf by
    leaf, and the area metric at its last stage, evaluated at the deformed sheet's
    centers, is the negated forward metric of the same cells: the log area factor of an
    inverse is the negative."""
    forward, forward_maps, _ = _stage_sequence(layers, square_sheet, inverse=False)
    sheets, maps, labels = _stage_sequence(layers, square_sheet, inverse=True)
    assert len(sheets) == len(forward) == len(labels) == len(layers) + 1
    for actual, desired in zip(sheets, stage_sheets(layers, square_sheet)[::-1]):
        for a, d in zip(jax.tree.leaves(actual), jax.tree.leaves(desired), strict=True):
            np.testing.assert_array_equal(a, d)
    assert labels == stage_labels(layers, inverse=True)
    n = len(layers)
    np.testing.assert_allclose(
        AREA.fn(maps[n], sheets[0].centers),
        -AREA.fn(forward_maps[n], square_sheet.centers),
        rtol=0.0,
        atol=TOL["cell_metric_affine"],
    )


def test_plot_deformation_strip_stages(layers, trajectories):
    """Two selected stages give two panels titled by their stage labels and a colorbar;
    a stage past the last layer raises with its index."""
    fig = plot_deformation_strip(layers, trajectories, stages=(0, 2))
    assert len(fig.axes) == 3
    titles = [ax.get_title() for ax in fig.axes[:2]]
    labels = stage_labels(layers)
    assert titles == [labels[k].replace(" — ", "\n", 1) for k in (0, 2)]
    plt.close(fig)
    with pytest.raises(ValueError, match="5"):
        plot_deformation_strip(layers, trajectories, stages=(0, 5))
    plt.close("all")


def test_animate_deformation_writes_every_frame(layers, trajectories, tmp_path):
    """The animation saved as a GIF has one frame per stage, the start included."""
    animation = animate_deformation(layers, trajectories)
    path = tmp_path / "flipbook.gif"
    animation.save(path, writer=PillowWriter(fps=2))
    plt.close("all")
    with Image.open(path) as gif:
        assert gif.n_frames == len(layers) + 1


def _center_x(
    f: Callable[[Float[Array, " 2"]], Float[Array, " 2"]],
    centers: Float[Array, "c c 2"],
) -> Float[Array, "c c"]:
    """The $x$-coordinate of each cell center's image under `f`."""
    return jax.vmap(jax.vmap(f))(centers)[..., 0]


def test_draw_sheet_with_a_custom_metric(square_sheet):
    """A metric defined outside the package fills each cell with its value, repeated
    over the cell's sub-quads, in the metric's colormap blended over white; the mesh,
    the circles and the extra curve are one line collection each."""
    metric = CellMetric(
        fn=_center_x,
        cmap=cc.cm.CET_L17,
        symmetric=False,
        label="$x$",
    )
    values = np.asarray(metric.fn(lambda x: x, square_sheet.centers))
    fig, ax = plt.subplots()
    norm = clipped_norm(values, 1.0, symmetric=False)
    alpha = 0.25
    fill = draw_sheet(
        ax, square_sheet, values, norm, metric, SheetStyle(fill_alpha=alpha)
    )
    fine = np.asarray(fill.get_array()).reshape(32, 32)
    np.testing.assert_array_equal(fine[::4, ::4], values)
    np.testing.assert_allclose(
        fill.cmap(1.0)[:3],
        1.0 - alpha * (1.0 - np.asarray(metric.cmap(1.0)[:3])),
        rtol=0.0,
        atol=TOL["identity"],
    )
    assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 3
    plt.close(fig)


def _start_markers(ax) -> list[Line2D]:
    """The lines on `ax` drawn as markers without a connecting line."""
    return [
        line
        for line in ax.lines
        if line.get_linestyle() == "None" and line.get_marker() != "None"
    ]


def test_draw_trajectories_marks_each_start(trajectories):
    """Each trajectory gets one curve and one start triangle, a single black marker at
    its first sample."""
    trajectories = np.asarray(trajectories)
    fig, ax = plt.subplots()
    draw_trajectories(ax, trajectories)
    markers = _start_markers(ax)
    assert len(markers) == len(trajectories)
    assert len(ax.lines) == 2 * len(trajectories)
    for marker, trajectory in zip(markers, trajectories, strict=True):
        assert len(marker.get_xdata()) == 1
        assert to_rgba(marker.get_color()) == to_rgba("black")
        np.testing.assert_array_equal(
            np.ravel([marker.get_xdata(), marker.get_ydata()]), trajectory[0]
        )
    plt.close(fig)


def test_curve_colors_forms():
    """One color is repeated, a sequence and a listed colormap are cycled from their
    first entry, and a value that is none of the three raises `ValueError`."""
    n = 5
    for spec, first in [
        ("k", "k"),
        (("r", "g"), "r"),
        (cc.cm.glasbey_dark, cc.cm.glasbey_dark.colors[0]),
    ]:
        colors = curve_colors(spec, n)
        assert len(colors) == n
        assert colors[0] == first
    with pytest.raises(ValueError):
        curve_colors(object(), n)


def test_set_limits_rejects_three_floats():
    """Three limits raise; four set the box at equal aspect."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="3"):
        set_limits(ax, (0.0, 1.0, 2.0))
    set_limits(ax, (0.0, -1.0, 2.0, 3.0))
    assert ax.get_xlim() == (0.0, 2.0) and ax.get_ylim() == (-1.0, 3.0)
    assert ax.get_aspect() == 1.0
    plt.close(fig)


# --- schematic ---


@pytest.fixture(scope="module")
def schematic_inputs(
    model,
) -> tuple[
    Float[np.ndarray, "2 20 2"],
    Float[np.ndarray, "2 20 2"],
    Float[np.ndarray, "65 2"],
    Float[np.ndarray, " 2"],
]:
    """Two windows of the model from $(1.5, 0)$ and $(0, -1.2)$ over 20 times in
    $[0, 1]$: `(xt, yt, cycle_points, equilibrium)`, the cycle closed by repeating its
    first point."""
    ts = jnp.linspace(0, 1, 20)
    x0s = jnp.array([[1.5, 0.0], [0.0, -1.2]])
    xt, yt = jax.vmap(model, in_axes=(None, 0))(ts, x0s)
    cycle = np.asarray(
        HAND_FOLIATION_FORM.limit_cycle(jnp.linspace(0, 1, 64, endpoint=False))
    )
    cycle_points = np.vstack([cycle, cycle[:1]])
    return (
        np.asarray(xt),
        np.asarray(yt),
        cycle_points,
        np.asarray(model.equilibrium()),
    )


def _start_marker_faces(ax, style: SchematicStyle) -> list[tuple[float, ...]]:
    """The face colors, as RGBA, of the single-point markers on `ax` other than the
    equilibrium marker, which is edged in `style.cycle_color`."""
    return [
        to_rgba(line.get_markerfacecolor())
        for line in ax.lines
        if isinstance(line, Line2D)
        and len(line.get_xdata()) == 1
        and line.get_marker() != "None"
        and to_rgba(line.get_markeredgecolor()) != to_rgba(style.cycle_color)
    ]


def test_schematic_drawers_mark_each_start(schematic_inputs):
    """Each drawer adds one start marker per window: filled with the window's color
    in the data space and white in the latent space."""
    xt, yt, cycle_points, equilibrium = schematic_inputs
    style = SchematicStyle()
    colors = [to_rgba(c) for c in curve_colors(style.curve_colors, len(xt))]
    white = [to_rgba("white")] * len(xt)
    for draw, expected in [
        (lambda ax: draw_data(ax, xt, cycle_points, equilibrium, style), colors),
        (lambda ax: draw_latent_initial(ax, yt[:, 0], cycle_points, style), white),
        (lambda ax: draw_latent_trajectories(ax, yt, cycle_points, style), white),
        (
            lambda ax: draw_prediction(ax, xt, xt, cycle_points, equilibrium, style),
            colors,
        ),
    ]:
        fig, ax = plt.subplots()
        draw(ax)
        assert _start_marker_faces(ax, style) == expected
        plt.close(fig)


def test_schematic_curves_dashed_in_latent_space_solid_in_data(schematic_inputs):
    """The latent trajectories are dashed and the data windows solid."""
    xt, yt, cycle_points, equilibrium = schematic_inputs
    style = SchematicStyle()
    for draw, dashed in [
        (lambda ax: draw_latent_trajectories(ax, yt, cycle_points, style), True),
        (lambda ax: draw_data(ax, xt, cycle_points, equilibrium, style), False),
    ]:
        fig, ax = plt.subplots()
        draw(ax)
        curves = [line for line in ax.lines if len(line.get_xdata()) == xt.shape[1]]
        assert len(curves) == len(xt)
        assert all((line.get_linestyle() != "-") == dashed for line in curves)
        plt.close(fig)


def test_schematic_rejects_bad_values(schematic_inputs):
    """Predictions of another window count fail the shape annotation under the
    jaxtyping hook, and a negative margin raises `ValueError`."""
    xt, _, cycle_points, equilibrium = schematic_inputs
    fig, ax = plt.subplots()
    with pytest.raises(jaxtyping.TypeCheckError):
        draw_prediction(ax, xt, xt[:1], cycle_points, equilibrium)
    plt.close(fig)
    with pytest.raises(ValueError, match="margin"):
        SchematicStyle(margin=-0.1)


def test_schematic_labels_replace_one_field():
    """`copy.replace` changes the loss and keeps every other default."""
    default = SchematicLabels()
    labels = copy.replace(default, loss="L")
    assert labels.loss == "L"
    assert copy.replace(labels, loss=default.loss) == default


# --- coupling_warp ---


@pytest.fixture(scope="module")
def warp_layers() -> tuple[CouplingFlow, CouplingFlow]:
    """The hand-set affine and spline coupling layers at their defaults."""
    return schematic_affine_coupling(), schematic_spline_coupling()


def test_schematic_couplings_are_hand_set(warp_layers):
    """Both hand-set layers invert to round-off on a few points of the box and move
    them, so neither is the identity."""
    points = jnp.array([[0.3, -0.2], [-0.7, 0.05], [0.9, 0.6], [-0.4, -0.8]])
    for layer in warp_layers:
        images = jax.vmap(layer)(points)
        np.testing.assert_allclose(
            jax.vmap(layer.inverse)(images), points, rtol=0, atol=TOL["closed_form"]
        )
        assert float(jnp.max(jnp.abs(images - points))) > 0.01


def test_spline_warp_is_steep_in_one_band(warp_layers):
    """At $x_1 = 0$ the spline warp's largest slope is at least five times its slope
    at either end of the box, where the boundary derivatives are 1."""
    _, spline = warp_layers
    bijection = jax.tree.map(lambda a: a[0], spline.make_bijection(jnp.array([0.0])))
    lo, hi = spline.template.xy_range
    slope = jax.grad(bijection)
    largest = jnp.max(jax.vmap(slope)(jnp.linspace(lo, hi, 2001)))
    assert float(largest) >= 5 * max(float(slope(lo)), float(slope(hi)))


def test_warp_curve_marks_knots_only_for_splines(warp_layers):
    """The spline's warp carries one marker line at its `num_bins + 1` knots; the
    affine warp carries none."""
    affine, spline = warp_layers
    for layer, expected in [(affine, []), (spline, [spline.template.num_bins + 1])]:
        fig, ax = plt.subplots()
        draw_warp_curve(ax, layer, (-1.0, 1.0), 0.0)
        assert [len(line.get_xdata()) for line in _start_markers(ax)] == expected
        plt.close(fig)


def test_warp_panels_keep_left_and_bottom_spines(warp_layers):
    """Under the rcParams of `configure_rcparams` the identity, affine and spline warp
    panels keep the left and bottom spines and the grid, and are ticked only at the
    box's ends."""
    box = (-1.0, 1.0)
    affine, spline = warp_layers
    drawers = [
        lambda ax: draw_identity_warp(ax, box),
        lambda ax: draw_warp_curve(ax, affine, box, 0.0),
        lambda ax: draw_warp_curve(ax, spline, box, 0.0),
    ]
    rc = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    with plt.rc_context(rc):
        for draw in drawers:
            fig, ax = plt.subplots()
            draw(ax)
            visible = {name for name, spine in ax.spines.items() if spine.get_visible()}
            assert visible == {"left", "bottom"}
            assert any(line.get_visible() for line in ax.get_xgridlines())
            assert list(ax.get_xticks()) == list(box)
            assert list(ax.get_yticks()) == list(box)
            plt.close(fig)


@pytest.fixture(scope="module")
def warp_trajectories() -> Float[np.ndarray, "t q 2"]:
    """Three short schematic trajectories in the unit box."""
    return schematic_trajectories((-1.0, 1.0), n=3, samples=20)


@pytest.mark.parametrize("box", [(-1.0, 1.0), (2.0, 5.0)])
def test_schematic_trajectories_fit_the_box(box):
    """The default four trajectories of 200 samples lie strictly inside the box and
    span more than half its width."""
    trajectories = schematic_trajectories(box)
    assert trajectories.shape == (4, 200, 2)
    lo, hi = box
    assert np.all((trajectories > lo) & (trajectories < hi))
    assert np.ptp(trajectories[..., 0]) > 0.5 * (hi - lo)


def test_grid_sheet_fixed_lines_stay_vertical(warp_layers):
    """The grid of the default 16-cell sheet, unmapped or mapped by either layer, is
    two collections: 9 of its 17 lines of constant $x_1$ in the fixed color, each at
    one $x$ to round-off, below 17 moved lines in the moved color; the panel has no
    spines and no ticks."""
    style = WarpStyle()
    box = (-1.0, 1.0)
    sheet = schematic_sheet(box, np.zeros((0, 2, 2)))
    for image in [sheet, *(sheet.map(eqx.filter_jit(layer)) for layer in warp_layers)]:
        fig, ax = plt.subplots()
        draw_grid_sheet(ax, image, box, style)
        fixed, moved = (c for c in ax.collections if isinstance(c, LineCollection))
        assert len(fixed.get_segments()) == 9
        assert len(moved.get_segments()) == 17
        for line in fixed.get_segments():
            assert np.ptp(line[:, 0]) <= TOL["identity"]
        assert to_rgba(fixed.get_colors()[0]) == to_rgba(style.fixed_color)
        assert to_rgba(moved.get_colors()[0]) == to_rgba(style.moved_color)
        assert fixed.get_zorder() < moved.get_zorder()
        assert not any(spine.get_visible() for spine in ax.spines.values())
        assert len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0
        plt.close(fig)


def test_grid_sheet_draws_trajectories(warp_layers, warp_trajectories):
    """The sheet's image under the spline layer carries one start triangle per
    trajectory, at the image of its first sample."""
    box = (-1.0, 1.0)
    _, spline = warp_layers
    image = schematic_sheet(box, warp_trajectories).map(eqx.filter_jit(spline))
    fig, ax = plt.subplots()
    draw_grid_sheet(ax, image, box)
    markers = _start_markers(ax)
    assert len(markers) == len(warp_trajectories)
    starts = np.asarray(jax.vmap(spline)(jnp.asarray(warp_trajectories[:, 0])))
    for marker, start in zip(markers, starts, strict=True):
        np.testing.assert_allclose(
            np.ravel([marker.get_xdata(), marker.get_ydata()]),
            start,
            rtol=0,
            atol=TOL["closed_form"],
        )
    plt.close(fig)


def test_warp_style_rejects_bad_values():
    """A negative margin, a zero width or figure size, a zero line spacing and a
    single sample each raise `ValueError` naming the field."""
    for field, value in [
        ("margin", -0.1),
        ("curve_width", 0.0),
        ("figsize", (0.0, 5.6)),
        ("fixed_every", 0),
        ("line_samples", 1),
    ]:
        with pytest.raises(ValueError, match=field):
            WarpStyle(**{field: value})


def test_warp_labels_replace_one_field():
    """`copy.replace` changes the spline header and keeps every other default."""
    default = WarpLabels()
    labels = copy.replace(default, spline_header="spline")
    assert labels.spline_header == "spline"
    assert copy.replace(labels, spline_header=default.spline_header) == default


def test_plot_coupling_warps_draws_six_panels(warp_layers, warp_trajectories):
    """The figure has six panels with identical limits, and the top row's titles are
    the three default headers."""
    fig = plot_coupling_warps(*warp_layers, warp_trajectories)
    axes = fig.get_axes()
    assert len(axes) == 6
    assert len({ax.get_xlim() for ax in axes}) == 1
    assert len({ax.get_ylim() for ax in axes}) == 1
    assert axes[0].get_xlim() == axes[0].get_ylim()
    labels = WarpLabels()
    titles = [ax.get_title() for ax in axes[:3]]
    assert titles == [
        labels.identity_header,
        labels.affine_header,
        labels.spline_header,
    ]
    plt.close(fig)


# --- inn_stack ---


def test_schematic_linear_rotates_and_stretches():
    """The hand-set linear layer has a positive determinant, singular values (by
    NumPy's SVD) a factor above 1.5 apart, a matrix that is not symmetric, so not a
    pure stretch, and an inverse that round-trips a few points."""
    layer = schematic_linear()
    weight = np.asarray(layer.weight)
    assert np.linalg.det(weight) > 0
    singular_values = np.linalg.svd(weight, compute_uv=False)
    assert singular_values[0] > 1.5 * singular_values[1]
    assert abs(weight[0, 1] - weight[1, 0]) > 0.1
    points = jnp.array([[0.3, -0.2], [-0.7, 0.05], [0.9, 0.6]])
    np.testing.assert_allclose(
        jax.vmap(layer.inverse)(jax.vmap(layer)(points)),
        points,
        rtol=0,
        atol=TOL["closed_form"],
    )


def test_stack_labels_replace_one_field():
    """`copy.replace` changes the forward label and keeps every other default, and
    five block labels raise `ValueError`."""
    default = StackLabels()
    labels = copy.replace(default, forward="F")
    assert labels.forward == "F"
    assert copy.replace(labels, forward=default.forward) == default
    with pytest.raises(ValueError, match="blocks"):
        StackLabels(blocks=default.blocks[:5])


def test_stack_style_rejects_bad_values():
    """A negative inset margin, a zero size, gap or font size, and an ellipsis gap
    below the block gap each raise `ValueError` naming the field."""
    for field, value in [
        ("inset_margin", -0.1),
        ("block_size", (0.0, 0.6)),
        ("block_gap", 0.0),
        ("inset_size", 0.0),
        ("arrow_label_size", 0.0),
        ("figsize", (12.0, 0.0)),
        ("ellipsis_gap", 0.1),
    ]:
        with pytest.raises(ValueError, match=field):
            StackStyle(**{field: value})


def test_plot_inn_stack_blocks_and_arrows():
    """The overlay axes carries six blocks and seven arrows, five between blocks and
    the two long ones; every default label is a text of the figure; and four inset
    axes follow the overlay."""
    fig = plot_inn_stack(schematic_linear(), schematic_spline_coupling())
    overlay, *insets = fig.get_axes()
    blocks = [p for p in overlay.patches if isinstance(p, FancyBboxPatch)]
    arrows = [p for p in overlay.patches if isinstance(p, FancyArrowPatch)]
    assert len(blocks) == 6
    assert len(arrows) == 7
    double_headed = [
        a for a in arrows if isinstance(a.get_arrowstyle(), ArrowStyle.CurveAB)
    ]
    assert len(double_headed) == 5
    labels = StackLabels()
    expected = {
        *labels.blocks,
        labels.ellipsis,
        labels.forward,
        labels.inverse,
        labels.input,
        labels.output,
        labels.linear,
        labels.coupling,
    }
    texts = {t.get_text() for t in fig.findobj(Text)}
    assert expected <= texts
    assert len(insets) == 4
    plt.close(fig)
