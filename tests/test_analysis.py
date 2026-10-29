"""The data side of ``deep_isochron.analysis``, against the normal forms as oracles.

Section 1 tests `Cycle`, the closed curve: evaluation, derivative, nearest phase and
distance, least-squares construction from points, and validation. Section 2 tests
`estimate_cycle`, the cycle from trajectory tails: Bautin and Hopf give the unit circle
and ``nf.period()``, an affine image gives the image curve with the phase still
uniform in time, pooled trajectories agree with a single one, measurement noise
averages out, a transient tail is reported as not converged, the winding follows the
sign of ``w``, and FitzHugh–Nagumo matches Langfield et al. 2014 (slow).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.analysis import Cycle
from deep_isochron.analysis.data import (
    density_coefficients,
    estimate_cycle,
    phase_from_protophase,
    protophase,
)
from deep_isochron.data import generate, UniformAnnulus
from deep_isochron.systems.normal_forms import BautinNormalForm, HopfNormalForm
from deep_isochron.utils.numerics import phase_difference

from .helpers import TOL


def trajectories(nf, *, periods=8.0, steps_per_period=100, n=6, seed=0, r=(0.3, 2.5)):
    """``n`` trajectories of ``nf`` from an annulus, ``periods`` periods long, as
    ``(ys (n, T, 2), ts (T,))`` NumPy arrays."""
    T = float(nf.period())
    ts = jnp.linspace(0.0, periods * T, int(periods * steps_per_period) + 1)
    src = generate(nf, UniformAnnulus(*r), ts, n, seed=seed)
    return np.asarray(src.ys), np.asarray(src.ts)


# ------------------------------------------------------------------ 1. Cycle --------
def test_circle_evaluation_derivative_and_queries():
    """A `Cycle.circle` evaluates to its parametrization, its derivative is the
    tangent, and `nearest_phase`, `distance` and `nearest_point` are exact on a ray
    from the center."""
    c = Cycle.circle(2.0, period=3.0, center=(1.0, -1.0))
    phi = c.phases(16)
    assert np.allclose(phi, np.arange(16) / 16, atol=TOL["identity"])
    angle = 2 * np.pi * phi
    pts = c(phi)
    assert np.allclose(
        pts,
        np.stack([1.0 + 2 * np.cos(angle), -1.0 + 2 * np.sin(angle)], -1),
        atol=TOL["identity"],
    )
    # the tangent with respect to the phase in turns: 2π times the angle's tangent
    assert np.allclose(
        c.derivative(phi),
        2 * np.pi * np.stack([-2 * np.sin(angle), 2 * np.cos(angle)], -1),
        atol=TOL["identity"],
    )
    assert c.harmonics == 1 and np.isclose(c.omega, 2 * np.pi / 3.0)
    # nearest phase and distance of points along a ray at polar angle 1.0
    x = np.array([1.0, -1.0]) + np.outer([0.5, 2.0, 3.5], [np.cos(1.0), np.sin(1.0)])
    ray_phase = 1.0 / (2 * np.pi)
    assert np.allclose(
        phase_difference(c.nearest_phase(x), ray_phase), 0.0, atol=TOL["cycle_query"]
    )
    assert np.allclose(c.distance(x), [1.5, 0.0, 1.5], atol=TOL["cycle_query"])
    assert np.allclose(
        c.nearest_point(x), c(np.full(3, ray_phase)), atol=TOL["cycle_query"]
    )


def test_from_points_recovers_a_fourier_curve_and_reports_the_residual():
    """`Cycle.from_points` recovers a three-harmonic curve from 500 unordered samples
    to `TOL["closed_form"]`, reports a residual at the noise level when the samples
    are noisy, and raises for fewer points than coefficients."""
    rng = np.random.default_rng(0)
    truth = Cycle(
        center=np.array([0.3, -0.2]),
        cos=np.array([[1.0, 0.1], [0.2, 0.0], [0.0, 0.05]]),
        sin=np.array([[0.0, 0.8], [0.1, 0.1], [0.02, 0.0]]),
        period=2.0,
    )
    phi = rng.uniform(0, 1, 500)  # any order, any spacing
    fit = Cycle.from_points(phi, truth(phi), truth.period, harmonics=3)
    assert np.allclose(fit.center, truth.center, atol=TOL["closed_form"])
    assert np.allclose(fit.cos, truth.cos, atol=TOL["closed_form"])
    assert np.allclose(fit.sin, truth.sin, atol=TOL["closed_form"])
    assert fit.residual < TOL["closed_form"]
    noisy = Cycle.from_points(phi, truth(phi) + rng.normal(0, 0.05, (500, 2)), 2.0, 3)
    assert 0.03 < noisy.residual < 0.08  # ≈ the noise level (two coordinates)
    with pytest.raises(ValueError, match="cannot determine"):
        Cycle.from_points(phi[:5], truth(phi[:5]), 2.0, harmonics=3)


def test_cycle_validation():
    """`Cycle` raises `ValueError` for a zero period and for coefficient arrays whose
    shape is not ``(K, 2)``."""
    with pytest.raises(ValueError, match="period"):
        Cycle.circle(1.0, period=0.0)
    with pytest.raises(ValueError, match="K, 2"):
        Cycle(np.zeros(2), np.zeros((0, 2)), np.zeros((0, 2)), 1.0)


# ------------------------------------------------------- 2. estimate_cycle (E1) ------
@pytest.mark.parametrize(
    "nf",
    [BautinNormalForm(1.0, 0.5, 2.0, 1.0), HopfNormalForm(1.0, 1.5, 0.7)],
    ids=["bautin", "hopf"],
)
def test_estimate_cycle_recovers_the_unit_circle_and_period(nf):
    """`estimate_cycle` on Bautin and Hopf trajectory tails gives the unit circle, the
    period of the normal form, a winding of 1, and a phase that is the polar angle
    up to one constant."""
    ys, ts = trajectories(nf)
    est = estimate_cycle(ys, ts, tail_periods=3.0)
    assert est.converged and est.winding == 1
    assert np.isclose(est.period, float(nf.period()), rtol=TOL["cycle_period"])
    assert (
        np.abs(est.period_samples - float(nf.period())).max() < TOL["cycle_period"] * 10
    )
    pts = est.cycle.points(256)
    assert np.allclose(np.linalg.norm(pts, axis=-1), 1.0, atol=TOL["cycle_curve"])
    assert np.allclose(est.cycle.center, 0.0, atol=TOL["cycle_curve"])
    # the phase advances uniformly in time: on the unit circle it is the polar angle up
    # to one constant
    phi = est.cycle.phases(256)
    offset = phase_difference(np.arctan2(pts[:, 1], pts[:, 0]) / (2 * np.pi), phi)
    assert np.ptp(offset) < TOL["cycle_phase"]
    assert est.revolutions.min() > 7.0


def test_estimate_cycle_on_an_affine_image_keeps_the_phase_uniform_in_time():
    """Map the Bautin trajectories through a fixed affine map: the cycle is an ellipse
    whose polar angle about its center is *not* uniform in time, so the protophase →
    phase step is what makes the recovered phase agree with the true one (the polar
    angle in the original coordinates) up to a constant."""
    nf = BautinNormalForm(1.0, 0.5, 2.0, 1.0)
    ys, ts = trajectories(nf)
    A, b = np.array([[2.0, 0.6], [-0.3, 0.8]]), np.array([1.0, -0.5])
    est = estimate_cycle(ys @ A.T + b, ts, tail_periods=3.0)
    assert est.converged
    assert np.isclose(est.period, float(nf.period()), rtol=TOL["cycle_period"])
    phi = est.cycle.phases(256)
    # the recovered curve at phase φ is the image of angle φ + const: find the constant
    # from one point, then compare the whole curve
    pts = est.cycle(phi)
    back = (pts - b) @ np.linalg.inv(A).T
    angle = np.arctan2(back[:, 1], back[:, 0])
    offset = phase_difference(angle / (2 * np.pi), phi)
    assert np.ptp(offset) < TOL["cycle_phase"]
    const = np.angle(np.mean(np.exp(2j * np.pi * offset)))  # circular mean, an angle
    shifted = np.stack(
        [np.cos(2 * np.pi * phi + const), np.sin(2 * np.pi * phi + const)], -1
    )
    assert np.allclose(pts, shifted @ A.T + b, atol=TOL["cycle_curve"])


def test_estimate_cycle_pools_trajectories_and_accepts_a_single_one():
    """A single trajectory and four pooled ones give the same period and curve, with
    three period samples per trajectory."""
    nf = BautinNormalForm(1.0, 0.5, 2.0, 1.0)
    ys, ts = trajectories(nf, n=4)
    single = estimate_cycle(ys[0], ts)
    pooled = estimate_cycle(ys, ts)
    assert single.period_samples.shape == (3,) and pooled.period_samples.shape == (12,)
    assert np.isclose(single.period, pooled.period, rtol=TOL["cycle_period"])
    assert pooled.cycle.distance(single.cycle.points(64)).max() < TOL["cycle_curve"]


def test_measurement_noise_is_averaged_out_and_shrinks_with_the_sample():
    """With Gaussian noise of ``sigma = 0.05`` on the Bautin trajectories the residual
    is at the noise level, the curve error is within a loose multiple of the
    coefficient standard error, and four times the sample roughly halves it."""
    nf = BautinNormalForm(1.0, 0.5, 2.0, 1.0)
    rng = np.random.default_rng(1)
    sigma = 0.05
    ys, ts = trajectories(nf, n=16)
    noisy = ys + rng.normal(0.0, sigma, ys.shape)

    def curve_error(est):
        pts = est.cycle.points(256)
        return np.abs(np.linalg.norm(pts, axis=-1) - 1.0).max()

    few, many = estimate_cycle(noisy[:4], ts), estimate_cycle(noisy, ts)
    assert few.converged and many.converged
    for est in (few, many):
        assert np.isclose(est.period, float(nf.period()), rtol=0.01)
        assert 0.5 * sigma * np.sqrt(2) < est.cycle.residual < 2 * sigma * np.sqrt(2)
        # 2K+1 = 23 coefficients, each with standard error ≈ σ·sqrt(2/n) over the
        # n ≈ 100·(tail revolutions) pooled points; the bound is a loose multiple
        n_points = 100 * len(est.period_samples)
        assert curve_error(est) < 10 * sigma * np.sqrt(23 / n_points)
    # four times the sample: the curve error roughly halves (fixed seed; loose bound)
    assert curve_error(many) < 0.8 * curve_error(few)


def test_a_transient_tail_is_reported_as_not_converged():
    """A slowly contracting Bautin cycle (``κ = −0.2``) is not converged after five
    periods from a small annulus and is after forty."""
    nf = BautinNormalForm(0.1, 0.5, 2.0)  # κ = −0.2: the deviation halves per period
    ys, ts = trajectories(nf, periods=5.0, r=(0.2, 0.5), n=4)
    assert not estimate_cycle(ys, ts, tail_periods=3.0).converged
    long_ys, long_ts = trajectories(nf, periods=40.0, r=(0.2, 0.5), n=4)
    assert estimate_cycle(long_ys, long_ts, tail_periods=3.0).converged


def test_winding_follows_the_rotation_direction():
    """A Bautin normal form with negative ``w`` gives winding −1 and a curve that is
    traversed clockwise (negative signed area), with the phase still advancing with
    time."""
    ys, ts = trajectories(BautinNormalForm(1.0, 0.5, -2.0, -1.0))
    est = estimate_cycle(ys, ts)
    assert est.winding == -1 and est.converged
    assert np.isclose(est.period, np.pi, rtol=TOL["cycle_period"])
    # the curve is still parameterized so that φ advances with time: γ runs clockwise
    pts = est.cycle.points(64)
    area = 0.5 * np.sum(
        pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]
    )
    assert area < 0


@pytest.mark.slow
def test_fitzhugh_nagumo_cycle_matches_langfield_et_al():
    """Langfield et al. 2014, Sec. III: the FitzHugh–Nagumo cycle has
    period T_Γ ≈ 11.2279 and is traversed clockwise; the relaxation-type loop needs more
    harmonics than a normal form's."""
    import diffrax as dfx
    from deep_isochron.data import UniformBox
    from deep_isochron.systems import FitzhughNagumo, SolverConfig

    cfg = SolverConfig(solver=dfx.Tsit5(), rtol=1e-9, atol=1e-11, max_steps=1 << 17)
    ts = jnp.linspace(0.0, 120.0, 1201)  # the fhn data config's step, ~10.7 periods
    src = generate(
        FitzhughNagumo(), UniformBox([-3, -3], [3, 3]), ts, 8, seed=0, config=cfg
    )
    est = estimate_cycle(np.asarray(src.ys), np.asarray(src.ts), harmonics=24)
    assert est.converged and est.winding == -1
    assert (
        abs(est.period - 11.2279) < 2e-3
    )  # the tolerance of test_systems' period test
    assert est.cycle.residual < 1e-2
    assert np.all(np.abs(est.period_samples - 11.2279) < 2e-3)


def test_protophase_helpers_and_input_checks():
    """`protophase` is monotone along a trajectory; `phase_from_protophase` is the
    identity on a uniform protophase and undoes a sinusoidal warp; the time-integral
    `density_coefficients` agree with the sample mean over a closed period and ignore
    samples outside the window; too short a tail raises `ValueError`."""
    ys, ts = trajectories(BautinNormalForm(1.0, 0.5, 2.0, 1.0), periods=2.0, n=2)
    psi = protophase(ys, np.zeros(2))
    assert psi.shape == ys.shape[:2] and np.all(np.diff(psi, axis=-1) > 0)
    # a uniform protophase is already the phase: the transformation is the identity
    u = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    phi, F = phase_from_protophase(u, harmonics=5)
    assert np.allclose(phi, u, atol=TOL["closed_form"]) and np.allclose(
        F, 0.0, atol=TOL["closed_form"]
    )
    # a warped protophase ψ = φ + ε sin(φ − α) of a uniform φ: the transformation
    # recovers φ (the warp's density has a geometric Fourier tail, ε^k)
    v = u + 0.05 * np.sin(u - 0.4)
    phi_v, _ = phase_from_protophase(np.mod(v, 2 * np.pi), harmonics=11)
    # up to a constant
    assert np.ptp(np.angle(np.exp(1j * (phi_v - u)))) < TOL["closed_form"]
    # the time-integral estimator over an exact window (one closed period of the warped
    # grid) agrees with the sample mean over the open grid, and ignores samples outside
    # the window
    uc = np.linspace(0, 2 * np.pi, 4097)  # closed: the endpoint repeats the start
    vc = uc + 0.05 * np.sin(uc - 0.4)
    tc = np.linspace(0.0, 1.0, 4097)
    F_mean = density_coefficients(np.mod(vc[:-1], 2 * np.pi), 11)
    F_int = density_coefficients(vc, 11, tc, (0.0, 1.0))
    assert np.allclose(F_mean, F_int, atol=TOL["closed_form"])
    F_extra = density_coefficients(
        np.append(vc, vc[-1] + 0.3), 11, np.append(tc, 1.1), (0.0, 1.0)
    )
    assert np.allclose(F_int, F_extra, atol=TOL["identity"])
    with pytest.raises(ValueError, match="revolutions"):
        estimate_cycle(ys, ts, tail_periods=3.0)
    # (mismatched ``ts`` and non-planar input raise ``ValueError`` too, but under the
    # test suite's jaxtyping hook the shape annotations reject them first)
    with pytest.raises(ValueError, match="at least 1"):
        estimate_cycle(ys, ts, tail_periods=0.5)


def test_jax_arrays_are_accepted():
    """`estimate_cycle` accepts JAX arrays and returns NumPy arrays."""
    ys, ts = trajectories(BautinNormalForm(1.0, 0.5, 2.0, 1.0), n=2)
    est = estimate_cycle(jnp.asarray(ys), jnp.asarray(ts))
    assert isinstance(est.cycle.center, np.ndarray) and est.converged
