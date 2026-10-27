"""The isochron pipeline (``deep_isochron.analysis.ode.auto.isochron``, roadmap O7).

Section 1 tests the settings and the checks on the requested phases, and section 1b
saving, loading and plotting a foliation built by hand; neither needs AUTO.
Section 2 (marker ``auto``) runs the pipeline. Stage 2 must find the adjoint bundle's
branch point at multiplier 1 for every planar system, which the coworker's
``+Aᵀ`` bundle did not (``docs/design/isochrons.md`` §3). The FitzHugh–Nagumo isochrons
must lie on the coworker's stored curves (``tests/golden/fncMultiSeg_Iso_*.csv``, file
``k`` at phase fraction ``1 - k/100`` after their anchor), measured from points to the
stored polyline (decision D3): one way with the quick settings, whose curves are
shorter, and both ways with the full settings (also ``slow``).
"""

import math
import os
from pathlib import Path

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
from deep_isochron.analysis.ode.auto import _env, isochron
from deep_isochron.plotting import plot_isochrons
from deep_isochron.systems import FitzhughNagumo
from deep_isochron.systems.normal_forms import (
    BautinNormalForm,
    HopfNormalForm,
    WinfreeNormalForm,
)
from jaxtyping import Float

from .helpers import ISOCHRON_SETTINGS, TOL


GOLDEN = Path(__file__).parent / "golden"
ANCHOR = jnp.array([-1.72170711, 0.5])
"""The coworker's phase 0: the base point of their file 100."""


def _golden(k: int) -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "n 2"]]:
    """The base point and the points of the coworker's file `k`."""
    data = np.loadtxt(GOLDEN / f"fncMultiSeg_Iso_{k}.csv")
    return data[0, :2], data[:, 2:]


def _golden_phase(k: int) -> float:
    """The phase of the coworker's file `k`, wrapped to $(-π, π]$."""
    return math.remainder(2 * math.pi * (1 - k / 100), 2 * math.pi)


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
    wrapped = isochron._check_phases([0.0, 2 * math.pi + 0.5, -math.pi])
    assert wrapped == pytest.approx((0.0, 0.5, math.pi))
    with pytest.raises(ValueError, match="finite"):
        isochron._check_phases([0.0, float("nan")])
    with pytest.raises(ValueError, match="repeat"):
        isochron._check_phases([0.5, 0.5 + 2 * math.pi])


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


# ------------------------------------------------------------ 2. AUTO (auto) ----
@pytest.fixture(scope="module")
def auto():
    """The loaded AUTO, or a skip when no tree is built."""
    try:
        return _env.load_auto()
    except _env.AutoNotInstalled as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def fhn_cycle() -> LimitCycle:
    """FitzHugh–Nagumo's cycle with phase 0 at the coworker's anchor."""
    cycle = find_limit_cycle(FitzhughNagumo(), jnp.array([1.0, 0.5]), t_settle=50.0)
    return reanchor(cycle, ANCHOR)


@pytest.mark.auto
@pytest.mark.parametrize(
    "ode, x0",
    [
        (FitzhughNagumo(), [1.0, 0.5]),
        (HopfNormalForm(1.3, 1.5, 0.4), [1.2, 0.1]),
        (BautinNormalForm(1.0, 0.3, 1.5, 0.5), [1.1, 0.1]),
        (WinfreeNormalForm(-0.25, -0.5), [1.1, 0.1]),
    ],
    ids=["fitzhugh_nagumo", "hopf", "bautin", "winfree"],
)
def test_adjoint_bundle_branches_at_multiplier_one(auto, ode, x0):
    """Stages 1 and 2 find the bundle's branch point at multiplier 1 and normalize
    the bundle, for every planar system (decision D1); the coworker's ``+Aᵀ`` bundle
    has no such branch point on the normal forms."""
    cycle = find_limit_cycle(ode, jnp.asarray(x0), t_settle=50.0)
    with isochron._workdir(ode, "isochron") as problem:
        run = isochron._Runner(auto, problem)
        bundle = isochron._adjoint_bundle(run, isochron._periodic_orbit(run, cycle))
    idx = problem.idx
    assert bundle[f"PAR({idx.mult})"] == pytest.approx(1.0, abs=TOL["branch_point"])
    assert bundle[f"PAR({idx.norm})"] == pytest.approx(1.0, abs=TOL["branch_point"])
    assert bundle[f"PAR({idx.period})"] == pytest.approx(float(cycle.period), rel=1e-8)


def _arclength_from_base(
    points: Float[np.ndarray, "n 2"], base: Float[np.ndarray, "2"]
) -> Float[np.ndarray, " n"]:
    """The arclength along `points` from the point nearest `base`."""
    s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    return s - s[np.argmin(np.linalg.norm(points - base, axis=1))]


def _check_against_golden(foliation, files, *, both_ways: bool):
    """Every traced isochron starts at the golden base point and lies within
    ``TOL["isochron_curve"]`` of the golden curve. With `both_ways`, the golden curve
    also lies within that distance of the traced one, over the arclength the traced
    one covers: the box that ends a half is the cycle's, not the coworker's
    ``|x| = 10``, so the two curves end at different places."""
    assert not foliation.failures, foliation.failures
    for iso, k in zip(foliation.isochrons, files, strict=True):
        base, curve = _golden(k)
        assert iso.phase == pytest.approx(_golden_phase(k), abs=1e-12)
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
        assert set(iso.stop) <= set(isochron.STOP_REASONS)
        assert iso.points.shape == (len(iso.eta), 2) == (len(iso.arclength), 2)


@pytest.mark.auto
def test_fitzhugh_nagumo_isochrons_lie_on_the_coworkers(fhn_cycle):
    """With the quick settings, the isochrons at the phases of the coworker's files
    100 and 26 start at their base points and lie on their curves; the foliation
    records its provenance."""
    files = (100, 26)
    foliation = isochron_foliation(
        fhn_cycle, [_golden_phase(k) for k in files], ISOCHRON_SETTINGS["quick"]
    )
    _check_against_golden(foliation, files, both_ways=False)
    assert foliation.provenance["auto_commit"] == _env.pinned_commit()
    assert foliation.provenance["compiler"].startswith("GNU Fortran")


@pytest.mark.auto
@pytest.mark.slow
def test_fitzhugh_nagumo_isochrons_reproduce_the_coworkers(fhn_cycle):
    """With the coworker's settings, the isochrons of files 100, 26 and 51 coincide
    with theirs within ``TOL["isochron_curve"]`` in both directions."""
    files = (100, 26, 51)
    foliation = isochron_foliation(
        fhn_cycle, [_golden_phase(k) for k in files], ISOCHRON_SETTINGS["full"]
    )
    _check_against_golden(foliation, files, both_ways=True)
