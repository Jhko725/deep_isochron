"""The isochron pipeline (``deep_isochron.analysis.ode.auto.isochron``, roadmap O7).

Section 1 tests the settings and the checks on the requested phases, and section 1b
saving, loading and plotting a foliation built by hand; neither needs AUTO.
Section 2 (marker ``auto``) runs the pipeline against the oracles of
``docs/design/isochrons.md`` §8:

- stages 1–3 against O1 and O2, for every planar system: the bundle's branch point at
  multiplier 1, which the coworker's ``+Aᵀ`` bundle did not have (§3); the orbit and
  period against O1's flow; ``Jw`` along O2's Floquet vector; ``w·F`` constant; each
  rotated base point at its phase on O1's cycle; and, for the normal forms, ``w`` along
  the closed-form gradient of the phase;
- the coworker's Hopf route (his runs 0–1) against O1's cycle (decision 12);
- the Bautin and Winfree isochrons against their closed-form phase (decision 14);
- the FitzHugh–Nagumo isochrons against the coworker's stored curves
  (``tests/golden/fncMultiSeg_Iso_*.csv``, file ``k`` at phase fraction ``1 - k/100``
  after their anchor), measured from points to the stored polyline (decision D3): one
  way with the quick settings, whose curves are shorter, together with forward
  integration of the traced points; both ways with the full settings (also ``slow``).
"""

import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from deep_isochron.analysis.ode import (
    find_limit_cycle,
    floquet,
    Isochron,
    isochron_foliation,
    IsochronFailure,
    IsochronFoliation,
    IsochronSettings,
    LimitCycle,
    reanchor,
)
from deep_isochron.analysis.ode.auto import _env, isochron
from deep_isochron.analysis.ode.auto.codegen import AutoProblem
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


def test_a_user_point_on_the_box_is_a_box_stop():
    """A half that ends on an edge of the box stopped there; one that ends inside it
    stopped at ``eta_stop``, even when AUTO located that point with
    ``|eta|`` a little below ``eta_stop``."""
    idx = SimpleNamespace(eta=1, point=(2, 3))
    box = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    settings = IsochronSettings(eta_stop=1e-6)

    def end(eta, x, y):
        return {"TY": "UZ", "PAR(1)": eta, "PAR(2)": x, "PAR(3)": y}

    assert (
        isochron._stop_reason(end(-9.9999999532e-07, 0.8, 0.5), idx, settings, box)
        == "eta"
    )
    assert isochron._stop_reason(end(1.3e-10, 2.0, 0.1), idx, settings, box) == "box"
    assert isochron._stop_reason(end(-5e-8, 0.2, -2.0), idx, settings, box) == "box"


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
    return _cycle("fitzhugh_nagumo")


J = np.array([[0.0, -1.0], [1.0, 0.0]])
"""The quarter turn, which takes the bundle ``w`` to the stable direction."""


def _sine(a: Float[np.ndarray, "2"], b: Float[np.ndarray, "2"]) -> float:
    """The sine of the angle between the lines along `a` and `b`, accurate near 0."""
    return float(abs(a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(a) / np.linalg.norm(b))


def _components(solution: Any, ndim: int) -> Float[np.ndarray, "n d"]:
    """The first `ndim` components of an AUTO solution, one row per stored point."""
    return np.column_stack([solution[f"U({i})"] for i in range(1, ndim + 1)])


SYSTEMS = {
    "fitzhugh_nagumo": (FitzhughNagumo(), [1.0, 0.5], ANCHOR),
    "hopf": (HopfNormalForm(1.3, 1.5, 0.4), [1.2, 0.1], jnp.array([1.0, 0.0])),
    "bautin": (BautinNormalForm(1.0, 0.3, 1.5, 0.5), [1.1, 0.1], jnp.array([1.0, 0.0])),
    "winfree": (WinfreeNormalForm(-0.25, -0.5), [1.1, 0.1], jnp.array([1.0, 0.0])),
}
"""The planar systems of section 2: the system, a point in the basin, phase 0."""


def _cycle(name: str) -> LimitCycle:
    """The cycle of ``SYSTEMS[name]`` with phase 0 at its anchor."""
    ode, x0, anchor = SYSTEMS[name]
    return reanchor(find_limit_cycle(ode, jnp.asarray(x0), t_settle=50.0), anchor)


@pytest.mark.auto
@pytest.mark.parametrize("name", list(SYSTEMS))
def test_stages_one_to_three_agree_with_o1_and_o2(auto, name):
    """Stage 2 branches at multiplier 1 and normalizes the bundle (decision D1); its
    orbit is O1's flow and its period O1's; ``Jw`` lies along O2's Floquet vector and
    ``w·F`` is constant along the cycle; stage 3 puts the base point of phase
    ``2πk/m`` at O1's mesh point ``x0s[k]``, with ``Jw`` along the Floquet vector
    there, phase π included, where the rotation starts. On the normal forms ``w``
    lies along the gradient of the closed-form phase, and that phase at each base point
    is the requested one, signed by the sense of rotation (decision 14)."""
    cycle = _cycle(name)
    ode, m = cycle.ode, cycle.x0s.shape[0]
    vectors = np.asarray(floquet(cycle).vectors)
    assert m % 2 == 0  # so that k = m/2 is phase π, the start of stage 3
    ks = sorted({m // 8, m // 3, m // 2, (5 * m) // 8})
    phases = [math.remainder(2 * math.pi * k / m, 2 * math.pi) for k in ks]
    with isochron._workdir(ode, "isochron") as problem:
        run = isochron._Runner(auto, problem)
        idx = problem.idx
        bundle = isochron._adjoint_bundle(run, isochron._periodic_orbit(run, cycle))
        rotated = isochron._rotate_phase(run, bundle, phases, max(idx.last, 36))
    assert bundle[f"PAR({idx.mult})"] == pytest.approx(1.0, abs=TOL["branch_point"])
    assert bundle[f"PAR({idx.norm})"] == pytest.approx(1.0, abs=TOL["branch_point"])
    period, seg1 = bundle[f"PAR({idx.period})"], bundle[f"PAR({idx.seg1})"]
    assert period == pytest.approx(float(cycle.period), rel=TOL["auto_orbit"])
    # the orbit: segment 1 from phase 0, segment 2 from where segment 1 ends
    t, u = np.asarray(bundle["t"]), _components(bundle, 8)
    first = np.asarray(
        ode.flow(jnp.asarray(t * seg1), cycle.x0s[0], config=cycle.config).ys
    )
    second = ode.flow(
        jnp.asarray(t * (period - seg1)), jnp.asarray(first[-1]), config=cycle.config
    )
    np.testing.assert_allclose(u[:, :2], first, atol=TOL["auto_orbit"])
    np.testing.assert_allclose(u[:, 2:4], np.asarray(second.ys), atol=TOL["auto_orbit"])
    # the bundle's direction and scale
    assert _sine(J @ u[0, 4:6], vectors[0, :, 0]) < TOL["bundle_direction"]
    field = jax.vmap(ode.rhs, (None, 0))
    w_dot_f = np.concatenate(
        [
            (
                u[:, 4 + j : 6 + j]
                * np.asarray(field(0.0, jnp.asarray(u[:, j : j + 2])))
            ).sum(1)
            for j in (0, 2)
        ]
    )
    assert np.ptp(w_dot_f) < TOL["bundle_scale"] * abs(w_dot_f.mean())
    # stage 3
    assert set(rotated) == set(phases)
    for k, theta in zip(ks, phases):
        base, w = rotated[theta].u[0, 2:4], rotated[theta].u[0, 6:8]
        np.testing.assert_allclose(
            base, np.asarray(cycle.x0s[k]), atol=TOL["auto_orbit"]
        )
        assert _sine(J @ w, vectors[k, :, 0]) < TOL["bundle_direction"]
        if name != "fitzhugh_nagumo":
            gradient = np.asarray(jax.grad(ode.phase)(jnp.asarray(base)))
            assert _sine(w, gradient) < TOL["bundle_direction"]
            sense = math.copysign(1.0, float(ode.omega()))
            gap = math.remainder(
                float(ode.phase(jnp.asarray(base))) - sense * theta, 2 * math.pi
            )
            assert abs(gap) < TOL["isochron_phase"]


def _hopf_route(
    run: isochron._Runner, problem: AutoProblem, c: float
) -> tuple[float, Float[np.ndarray, " n"], Float[np.ndarray, "n 2"]]:
    """The coworker's runs 0–1 for FitzHugh–Nagumo: the equilibrium continued in ``c``
    from 3 down to the Hopf point, then the branch of periodic orbits born there
    (``IPS = 2``, period in ``PAR(11)``) continued back up to ``c``.

    **Returns:**

    The period, AUTO's stored mesh of $[0, 1]$, and the orbit's points on it.
    """
    slot, plain = problem.slot("c"), {"NDIM": 2, "NBC": 0, "NINT": 0}
    equilibria = run(
        [0.9066, -0.2582], PAR=problem.pars, ICP=[slot], IPS=1, DS=-0.01, NMX=200,
        SP=["HB1"], **plain,
    )  # fmt: skip
    orbits = run(
        equilibria("HB1"), IPS=2, ISW=-1, ICP=[slot, 11], DS=0.1, NMX=200,
        UZSTOP={slot: c}, **plain,
    )  # fmt: skip
    end = orbits("UZ1")
    return float(end["PAR(11)"]), np.asarray(end["t"]), _components(end, 2)


@pytest.mark.auto
def test_hopf_route_reaches_o1s_cycle(auto):
    """The coworker's route to the FitzHugh–Nagumo cycle, continuation from the Hopf
    point, ends at O1's period, and its orbit is O1's flow (decision 12)."""
    cycle = _cycle("fitzhugh_nagumo")
    ode = cycle.ode
    with isochron._workdir(ode, "isochron") as problem:
        period, t, points = _hopf_route(isochron._Runner(auto, problem), problem, ode.c)
    assert period == pytest.approx(float(cycle.period), rel=TOL["auto_orbit"])
    flowed = ode.flow(
        jnp.asarray(t * period), jnp.asarray(points[0]), config=cycle.config
    )
    np.testing.assert_allclose(points, np.asarray(flowed.ys), atol=TOL["auto_orbit"])


@pytest.mark.auto
@pytest.mark.parametrize("name", ["bautin", "winfree"])
def test_normal_form_isochrons_carry_the_closed_form_phase(auto, name):
    """With phase 0 at ``(1, 0)``, the closed-form phase at every traced point equals
    the requested phase, signed by the sense of rotation (decision 14), on halves
    that reach from inside half the cycle's radius to outside one and a half."""
    cycle = _cycle(name)
    ode = cycle.ode
    phases = [0.5, 2.0]
    foliation = isochron_foliation(cycle, phases, ISOCHRON_SETTINGS["reach"])
    assert not foliation.failures, foliation.failures
    sense = math.copysign(1.0, float(ode.omega()))
    for iso, theta in zip(foliation.isochrons, phases, strict=True):
        got = np.asarray(jax.vmap(ode.phase)(jnp.asarray(iso.points)))
        gap = np.remainder(got - sense * theta + np.pi, 2 * np.pi) - np.pi
        assert np.abs(gap).max() < TOL["isochron_phase"]
        radius = np.linalg.norm(iso.points, axis=1)
        assert radius.min() < 0.5 and radius.max() > 1.5


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
def test_fitzhugh_nagumo_isochrons_lie_on_the_coworkers(auto, fhn_cycle):
    """With the quick settings, the isochrons at the phases of the coworker's files
    100 and 26 start at their base points and lie on their curves; every traced point,
    flowed five periods by O1's solver, ends at its base point; the foliation records
    its provenance."""
    files = (100, 26)
    foliation = isochron_foliation(
        fhn_cycle, [_golden_phase(k) for k in files], ISOCHRON_SETTINGS["quick"]
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
        fhn_cycle, [_golden_phase(k) for k in files], ISOCHRON_SETTINGS["full"]
    )
    _check_against_golden(foliation, files, both_ways=True)
