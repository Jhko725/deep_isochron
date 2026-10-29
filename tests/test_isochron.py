"""The isochron pipeline's results and driver (``analysis.ode.auto``, roadmap O7).

Section 1 tests the settings, the checks on the requested phases, and `resample`,
saving, loading and plotting on a foliation built by hand; none needs AUTO. Section 2
(marker ``auto``) runs `isochron_foliation` on FitzHugh–Nagumo against the coworker's
stored curves (``tests/golden/fncMultiSeg_Iso_*.csv``, file ``k`` at phase fraction
``1 - k/100`` after their anchor), measured from points to the stored polyline
(decision D3): one way with the quick settings, whose curves are shorter, together with
forward integration of the traced points; both ways with the full settings (also
``slow``). The stages' own oracles are in ``test_auto_stages.py``.
"""

import math
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.analysis.ode import (
    find_limit_cycle,
    Isochron,
    isochron_foliation,
    IsochronFailure,
    IsochronFoliation,
    IsochronSettings,
    LimitCycle,
    reanchor,
)
from deep_isochron.analysis.ode.auto import _env, STOP_REASONS, wrap_phases
from deep_isochron.plotting import plot_isochrons
from deep_isochron.systems.normal_forms import HopfNormalForm
from deep_isochron.utils.numerics import arclength
from jaxtyping import Float

from .helpers import (
    anchored_cycle,
    golden_isochron,
    golden_phase,
    ISOCHRON_SETTINGS,
    TOL,
)


def _distance_to_curve(
    points: Float[np.ndarray, "n 2"], curve: Float[np.ndarray, "m 2"]
) -> Float[np.ndarray, " n"]:
    """The distance from each of `points` to the polyline through `curve`."""
    a, b = curve[:-1], curve[1:]
    ab = b - a
    length2 = np.maximum((ab**2).sum(axis=1), 1e-300)
    out = np.empty(len(points))
    for i, p in enumerate(points):
        t = np.clip(((p - a) * ab).sum(axis=1) / length2, 0.0, 1.0)
        out[i] = np.linalg.norm(a + t[:, None] * ab - p, axis=1).min()
    return out


# ------------------------------------------------------------- 1. settings ----
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_periods": 0},
        {"eta_stop": 0.0},
        {"box_scale": 1.0},
        {"ds": -1e-5},
        {"ntst": 0},
        {"max_workers": 0},
        {"max_steps": 0},
    ],
)
def test_settings_refuse_bad_values(kwargs):
    """`IsochronSettings` raises `ValueError` for a value outside its range."""
    with pytest.raises(ValueError):
        IsochronSettings(**kwargs)


def test_settings_defaults_scale_the_mesh_and_use_the_allowed_cores():
    """The mesh has 60 intervals per period unless `ntst` is set (decision D6), and
    the pool uses the cores the process may use unless `max_workers` is set (H3)."""
    assert IsochronSettings(n_periods=3).mesh_intervals == 180
    assert IsochronSettings(n_periods=3, ntst=50).mesh_intervals == 50
    assert IsochronSettings().workers == os.process_cpu_count()
    assert IsochronSettings(max_workers=3).workers == 3


def test_phases_are_wrapped_and_refused_when_not_finite_or_repeated():
    """The requested phases are wrapped to $(-π, π]$; a non-finite phase and a phase
    that repeats another after wrapping raise `ValueError` naming them (G9)."""
    wrapped = wrap_phases([0.0, 2 * math.pi + 0.5, -math.pi])
    assert wrapped == pytest.approx((0.0, 0.5, math.pi))
    with pytest.raises(ValueError, match="finite"):
        wrap_phases([0.0, float("nan")])
    with pytest.raises(ValueError, match="repeat"):
        wrap_phases([0.5, 0.5 + 2 * math.pi])


# ------------------------------------------------ 1b. saving, loading, plotting ----
@pytest.fixture(scope="module")
def hopf_foliation() -> IsochronFoliation:
    """A foliation of Hopf's cycle built by hand, without AUTO: two isochrons on
    radial lines, one with its halves stopped for different reasons, and a failure."""
    cycle = find_limit_cycle(
        HopfNormalForm(1.3, 1.5, 0.4), jnp.array([1.2, 0.1]), t_settle=20.0, m=2
    )
    isochrons = []
    for phase, n in ((0.0, 5), (2.0, 7)):
        r = np.linspace(0.5, 1.5, n)
        points = np.column_stack([r * np.cos(phase), r * np.sin(phase)])
        isochrons.append(
            Isochron(
                phase=phase,
                base_point=np.array([np.cos(phase), np.sin(phase)]),
                points=points,
                eta=r - 1.0,
                arclength=r - 1.0,
                stop=("eta", "box") if n == 5 else ("max_steps", "box"),
            )
        )
    failure = IsochronFailure(-1.0, "a planted failure", "last line", {"NMX": 10})
    return IsochronFoliation(
        cycle,
        tuple(isochrons),
        (failure,),
        IsochronSettings(n_periods=2),
        {"auto_commit": "abc"},
    )


def test_foliation_round_trips_through_netcdf(hopf_foliation, tmp_path):
    """`save` then `load` with the same cycle returns every isochron, the failures,
    the settings and the provenance unchanged."""
    path = hopf_foliation.save(tmp_path / "foliation.nc")
    back = IsochronFoliation.load(path, hopf_foliation.cycle)
    assert len(back.isochrons) == len(hopf_foliation.isochrons)
    for got, want in zip(back.isochrons, hopf_foliation.isochrons, strict=True):
        assert got.phase == want.phase and got.stop == want.stop
        for name in ("base_point", "points", "eta", "arclength"):
            np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
    assert back.failures == hopf_foliation.failures
    assert back.settings == hopf_foliation.settings
    assert back.provenance == hopf_foliation.provenance
    with pytest.raises(FileExistsError):
        hopf_foliation.save(path)


def test_load_refuses_another_cycle(hopf_foliation, tmp_path):
    """`load` raises `ValueError` naming the mismatch for a cycle of other
    parameters, and for the same cycle with another phase origin."""
    path = hopf_foliation.save(tmp_path / "foliation.nc")
    other = find_limit_cycle(
        HopfNormalForm(1.3, 1.6, 0.4), jnp.array([1.2, 0.1]), t_settle=20.0, m=2
    )
    with pytest.raises(ValueError, match="parameters"):
        IsochronFoliation.load(path, other)
    moved = reanchor(hopf_foliation.cycle, jnp.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="phase origin"):
        IsochronFoliation.load(path, moved)


def test_plot_draws_every_isochron_and_the_cycle(hopf_foliation):
    """`plot_isochrons` draws one line per isochron and one for the cycle."""
    fig = plot_isochrons(hopf_foliation)
    assert len(fig.axes[0].lines) == len(hopf_foliation.isochrons) + 1
    plt.close(fig)


def _arc_isochron(radius: float, angles: Float[np.ndarray, " n"]) -> Isochron:
    """An isochron built by hand on the circle of `radius` about the origin, at the
    increasing `angles`, with its base point at angle 0 (which `angles` must contain)
    and ``eta`` growing geometrically away from it, negative on the first half."""
    points = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    s = np.array(arclength(points))
    base = int(np.argmin(np.abs(angles)))
    s -= s[base]
    eta = np.sign(s) * 1e-12 * np.exp(10 * np.abs(s))
    return Isochron(0.0, points[base], points, eta, s, ("eta", "box"))


def test_resample_spaces_points_evenly_on_the_traced_curve():
    """On a circular arc traced at uneven steps, `resample` returns points at exactly
    the requested arclength spacing, anchored at the base point, which it keeps; each
    lies on the traced polyline, so within ``h²/8R`` of the circle for the traced
    steps ``h``; ``eta`` keeps its sign, is 0 at the base point and grows away from it
    on both halves."""
    radius = 2.0
    angles = np.concatenate(
        [-np.geomspace(0.6, 1e-4, 300), [0.0], np.linspace(1e-3, 0.4, 40) ** 1.5]
    )
    iso = _arc_isochron(radius, angles)
    out = iso.resample(0.01)
    np.testing.assert_allclose(np.diff(out.arclength), 0.01, rtol=1e-12)
    assert out.arclength.min() >= iso.arclength.min()
    assert out.arclength.max() <= iso.arclength.max()
    zero = int(np.flatnonzero(out.arclength == 0.0)[0])
    np.testing.assert_array_equal(out.points[zero], iso.base_point)
    h = np.diff(iso.arclength).max()
    sag = np.abs(np.linalg.norm(out.points, axis=1) - radius)
    assert sag.max() <= h**2 / (8 * radius) * (1 + 1e-6)
    assert out.eta[zero] == 0.0
    assert np.all(np.sign(out.eta) == np.sign(out.arclength))
    assert np.all(np.diff(np.abs(out.eta[: zero + 1])) <= 0)
    assert np.all(np.diff(np.abs(out.eta[zero:])) >= 0)
    assert (out.phase, out.stop) == (iso.phase, iso.stop)


def test_resample_is_exact_on_a_line_and_refuses_a_bad_spacing():
    """On a straight segment the resampled points are the exact points at their
    arclength; a spacing that is not positive and finite raises `ValueError`."""
    s = np.concatenate(
        [-np.geomspace(1.0, 1e-3, 50), [0.0], np.geomspace(1e-3, 0.5, 30)]
    )
    direction = np.array([0.6, 0.8])
    base = np.array([1.0, -1.0])
    iso = Isochron(
        1.0, base, base + s[:, None] * direction, s * 1e-9, s, ("eta", "eta")
    )
    out = iso.resample(0.05)
    np.testing.assert_allclose(
        out.points, base + out.arclength[:, None] * direction, atol=1e-15
    )
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="spacing"):
            iso.resample(bad)


# ------------------------------------------- 2. the foliation against the coworker ----
@pytest.fixture(scope="module")
def fhn_cycle() -> LimitCycle:
    """FitzHugh–Nagumo's cycle with phase 0 at the coworker's anchor."""
    return anchored_cycle("fitzhugh_nagumo")


def _arclength_from_base(
    points: Float[np.ndarray, "n 2"], base: Float[np.ndarray, "2"]
) -> Float[np.ndarray, " n"]:
    """The arclength along `points` from the point nearest `base`."""
    s = np.array(arclength(points))
    return s - s[np.argmin(np.linalg.norm(points - base, axis=1))]


def _check_against_golden(foliation, files, *, both_ways: bool):
    """Every traced isochron starts at the golden base point and lies within
    ``TOL["isochron_curve"]`` of the golden curve. With `both_ways`, the golden curve
    also lies within that distance of the traced one, over the arclength the traced
    one covers: the box that ends a half is the cycle's, not the coworker's
    ``|x| = 10``, so the two curves end at different places."""
    assert not foliation.failures, foliation.failures
    for iso, k in zip(foliation.isochrons, files, strict=True):
        base, curve = golden_isochron(k)
        assert iso.phase == pytest.approx(golden_phase(k), abs=1e-12)
        np.testing.assert_allclose(iso.base_point, base, atol=1e-6)
        assert _distance_to_curve(iso.points, curve).max() < TOL["isochron_curve"]
        if both_ways:
            s = _arclength_from_base(curve, base)
            margin = 1e-3 * (iso.arclength.max() - iso.arclength.min())
            inside = (s > iso.arclength.min() + margin) & (
                s < iso.arclength.max() - margin
            )
            assert inside.sum() > len(curve) // 2  # most of the stored curve is covered
            dist = _distance_to_curve(curve[inside], iso.points)
            assert dist.max() < TOL["isochron_curve"]
        # the record: arclength zero at the base point, a stop reason per half
        i = int(np.argmin(np.abs(iso.arclength)))
        np.testing.assert_allclose(iso.points[i], iso.base_point, atol=1e-6)
        assert set(iso.stop) <= set(STOP_REASONS)
        assert iso.points.shape == (len(iso.eta), 2) == (len(iso.arclength), 2)


@pytest.mark.auto
def test_fitzhugh_nagumo_isochrons_lie_on_the_coworkers(auto, fhn_cycle):
    """With the quick settings, the isochrons at the phases of the coworker's files
    100 and 26 start at their base points and lie on their curves; every traced point,
    flowed five periods by O1's solver, ends at its base point; the foliation records
    its provenance."""
    files = (100, 26)
    foliation = isochron_foliation(
        fhn_cycle, [golden_phase(k) for k in files], ISOCHRON_SETTINGS["quick"]
    )
    _check_against_golden(foliation, files, both_ways=False)
    ode, horizon = fhn_cycle.ode, jnp.array([0.0, 5 * float(fhn_cycle.period)])
    end = jax.vmap(lambda p: ode.flow(horizon, p, config=fhn_cycle.config).ys[-1])
    for iso in foliation.isochrons:
        ends = np.asarray(end(jnp.asarray(iso.points)))
        distance = np.linalg.norm(ends - iso.base_point, axis=1)
        assert distance.max() < TOL["isochron_forward"]
    assert foliation.provenance["auto_commit"] == _env.pinned_commit()
    assert foliation.provenance["compiler"].startswith("GNU Fortran")


@pytest.mark.auto
@pytest.mark.slow
def test_fitzhugh_nagumo_isochrons_reproduce_the_coworkers(auto, fhn_cycle):
    """With the coworker's settings, the isochrons of files 100, 26 and 51 coincide
    with theirs within ``TOL["isochron_curve"]`` in both directions."""
    files = (100, 26, 51)
    foliation = isochron_foliation(
        fhn_cycle, [golden_phase(k) for k in files], ISOCHRON_SETTINGS["full"]
    )
    _check_against_golden(foliation, files, both_ways=True)
