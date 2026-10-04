"""The phase-autoencoder baseline of Yawata et al. (Chaos 34, 063111, 2024), roadmap
B12, ``docs/design/normal-forms.md`` §5.3.

The exact phase–amplitude chart of a normal form, ``(cos Θ, sin Θ, Ψ)``, is a perfect
encoder for the latent flow with ``omega = ω₁``, ``kappa = κ`` — that is the
correspondence of §5.3 — so most laws here are checked against it exactly. The one
training test (marked ``slow``) learns the chart of a Hopf form from near-cycle data and
compares the learned phase with ``to_phase_amplitude`` up to a constant; ``Ψ`` (up to
scale) and ``κ`` are reported, not asserted, since the paper itself finds the decay
variable "closely related, though not equivalent" to the isostable coordinate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from deep_isochron.data import generate, OnCycleGaussian, windows
from deep_isochron.model import (
    PhaseAmplitudeAutoencoder,
    PhaseAmplitudeLatentDynamics,
)
from deep_isochron.model.autoencoder import normalise_phase_plane
from deep_isochron.systems import BautinNormalForm, HopfNormalForm
from deep_isochron.training.losses import PhaseAutoencoderLoss
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


def _chart(nf):
    """The exact encoder ``u ↦ (cos Θ, sin Θ, Ψ)`` of a normal form."""

    def encode(u):
        theta, psi = nf.to_phase_amplitude(u)
        return jnp.stack((jnp.cos(theta), jnp.sin(theta), psi))

    return encode


def _exact_dynamics(nf):
    return PhaseAmplitudeLatentDynamics(
        omega=float(nf.omega()), kappa=float(nf.floquet_exponent())
    )


def _exact_autoencoder(nf, dynamics=None):
    """A ``PhaseAmplitudeAutoencoder`` whose encoder/decoder are the exact chart of
    ``nf``
    and its inverse (``eqx.tree_at`` on the MLP fields)."""

    def decode(y):
        return nf.from_phase_amplitude(jnp.stack((jnp.arctan2(y[1], y[0]), y[2])))

    model = PhaseAmplitudeAutoencoder(
        2, dynamics or _exact_dynamics(nf), mlp_width=2, key=jax.random.key(0)
    )
    return eqx.tree_at(lambda m: (m.encoder, m.decoder), model, (_chart(nf), decode))


normal_forms = st.one_of(
    st.builds(
        HopfNormalForm,
        a=st.floats(0.3, 2.0),
        w=st.floats(0.5, 3.0),
        w0=st.floats(-1, 1),
    ),
    st.builds(
        BautinNormalForm,
        a=st.floats(0.3, 2.0),
        b=st.floats(0.0, 1.0),
        w=st.floats(0.5, 3.0),
        w0=st.floats(-1, 1),
    ),
)


# --------------------------------------------------------------- latent dynamics ---
@given(
    omega=st.floats(-3, 3),
    kappa=st.floats(-3, -0.1),
    y0=st.tuples(st.floats(-2, 2), st.floats(-2, 2), st.floats(-1, 1)).filter(
        lambda y: y[0] ** 2 + y[1] ** 2 > 0.01
    ),
)
def test_latent_flow_is_rotation_and_decay(omega, kappa, y0):
    """Eqs. (12)–(14): the radius in the (Y1, Y2) plane is preserved, the angle advances
    by ω t, Y3 decays as e^{κt}; and the flow is a semigroup in t."""
    dyn = PhaseAmplitudeLatentDynamics(omega, kappa)
    assert_close(dyn.kappa, kappa, rtol=TOL["closed_form"])
    y0 = jnp.asarray(y0)
    ts = jnp.array([0.0, 0.3, 1.1])
    ys = dyn(ts, y0)
    assert_close(ys[0], y0, rtol=TOL["closed_form"], atol=TOL["closed_form"])
    radius = jnp.sqrt(ys[:, 0] ** 2 + ys[:, 1] ** 2)
    assert_close(radius, jnp.full_like(radius, radius[0]), rtol=TOL["closed_form"])
    angle = jnp.arctan2(ys[:, 1], ys[:, 0]) - jnp.arctan2(y0[1], y0[0])
    gap = jnp.arctan2(jnp.sin(angle - omega * ts), jnp.cos(angle - omega * ts))
    assert_close(gap, jnp.zeros_like(gap), atol=TOL["scalar_roundtrip"])
    assert_close(ys[:, 2], y0[2] * jnp.exp(kappa * ts), rtol=TOL["closed_form"])
    # semigroup: evolving from the state at t = 0.3 reproduces the state at t = 1.1
    again = dyn(jnp.array([0.3, 1.1]), ys[1])
    assert_close(again[-1], ys[-1], rtol=TOL["scalar_roundtrip"], atol=1e-12)


def test_latent_dynamics_requires_a_stable_cycle():
    with pytest.raises(ValueError):
        PhaseAmplitudeLatentDynamics(1.0, 0.0)
    assert PhaseAmplitudeLatentDynamics(1.0, -2.0).params() == {
        "omega": 1.0,
        "kappa": pytest.approx(-2.0),
    }


@given(nf=normal_forms, r=st.floats(0.5, 1.8), theta=st.floats(-3.1, 3.1))
@settings(max_examples=20)
def test_exact_chart_conjugates_the_normal_form_to_the_latent_flow(nf, r, theta):
    """§5.3: with ``omega = ω₁``, ``kappa = κ`` the latent flow applied to the chart of
    ``u₀`` equals the chart of the flow of ``u₀`` — the latent space *is* ``(Θ, Ψ)``."""
    u0 = jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])
    ts = jnp.linspace(0.0, 1.5, 7)
    ys = nf.flow(ts, u0, integration="closed_form").ys
    chart = _chart(nf)
    assert_close(
        jax.vmap(chart)(ys),
        _exact_dynamics(nf)(ts, chart(u0)),
        rtol=TOL["scalar_roundtrip"],
        atol=TOL["scalar_roundtrip"],
    )


# ------------------------------------------------------------------ autoencoder ---
@given(y=st.tuples(st.floats(-3, 3), st.floats(-3, 3), st.floats(-3, 3)))
def test_encoder_normalisation(y):
    """Eqs. (15)–(16): ``(Y1, Y2)`` on the unit circle, ``Y3`` untouched."""
    y = jnp.asarray(y)
    if y[0] ** 2 + y[1] ** 2 < 1e-6:
        return
    z = normalise_phase_plane(y)
    assert_close(z[0] ** 2 + z[1] ** 2, 1.0, rtol=TOL["closed_form"])
    assert z[2] == y[2]
    assert_close(jnp.arctan2(z[1], z[0]), jnp.arctan2(y[1], y[0]), rtol=1e-12)


def test_autoencoder_shapes_and_derived_quantities():
    model = PhaseAmplitudeAutoencoder(2, mlp_width=8, key=jax.random.key(0))
    x = jnp.array([0.3, -1.2])
    y = model.encode(x)
    assert y.shape == (3,)
    assert_close(y[0] ** 2 + y[1] ** 2, 1.0, rtol=TOL["closed_form"])
    assert bool(jnp.abs(model.phase(x)) <= jnp.pi)
    assert model.amplitude(x) == y[2]
    assert model.cycle_point(jnp.asarray(0.7)).shape == (2,)
    z = model.phase_sensitivity(jnp.asarray(0.7))  # Eq. (20), by autodiff
    assert z.shape == (2,) and bool(jnp.all(jnp.isfinite(z)))
    ts = jnp.linspace(0.0, 1.0, 5)
    xt, yt = model(ts, x)
    assert xt.shape == (5, 2) and yt.shape == (5, 3)
    assert_close(yt[0], y, rtol=TOL["closed_form"])


# ------------------------------------------------------------------------- loss ---
def _window_batch(nf, n=16, length=6, seed=0):
    sampler = OnCycleGaussian.from_normal_form(nf, 200, 0.3)
    u0 = sampler(jax.random.key(seed), n)
    ts = jnp.linspace(0.0, 1.0, length)
    ys = eqx.filter_vmap(lambda u: nf.flow(ts, u, integration="closed_form").ys)(u0)
    return {"t": jnp.broadcast_to(ts, (n, length)), "u": ys}


def test_loss_vanishes_on_the_exact_chart():
    """With the exact chart and the exact (ω₁, κ), reconstruction, phase and deviation
    losses are zero; only the centre-of-mass term is left, and it is reported."""
    nf = HopfNormalForm(1.0, 2.0, 0.5)
    model = _exact_autoencoder(nf)
    batch = _window_batch(nf)
    total, aux = PhaseAutoencoderLoss()(model, batch)
    for name in ("recon", "pha", "dev"):
        assert_close(aux[name], 0.0, atol=TOL["vector_roundtrip"])
    assert 0.0 <= aux["aux"] <= 1.0
    assert_close(total, 2.0 * aux["aux"], rtol=TOL["closed_form"], atol=1e-10)
    assert_close(aux["omega"], nf.omega(), rtol=TOL["identity"])
    assert_close(aux["kappa"], nf.floquet_exponent(), rtol=TOL["closed_form"])


def test_loss_detects_wrong_latent_frequency_and_switches_weights():
    """A wrong ω shows up in ``pha`` only (the decay is still right); with the exact
    chart the weights switch to the paper's second stage once pha and aux are small."""
    nf = HopfNormalForm(1.0, 2.0, 0.5)
    batch = _window_batch(nf)
    wrong = _exact_autoencoder(
        nf,
        PhaseAmplitudeLatentDynamics(
            float(nf.omega()) + 0.5, float(nf.floquet_exponent())
        ),
    )
    _, aux = PhaseAutoencoderLoss()(wrong, batch)
    assert aux["pha"] > 1e-2 and aux["dev"] < TOL["vector_roundtrip"]
    assert not bool(aux["switched"])

    exact = _exact_autoencoder(nf)
    loss = PhaseAutoencoderLoss(switch_at=(0.01, 10.0))  # aux threshold never binds
    total, aux = loss(exact, batch)
    assert bool(aux["switched"])
    assert_close(total, 0.0, atol=TOL["vector_roundtrip"])  # w_aux = 0 after the switch


def test_alpha_schedule_weights_early_steps_when_phase_loss_is_large():
    """Eq. (24): α_k = k^{-min(1, L_pha)}; for a badly wrong ω the k-th step error is
    down-weighted by 1/k, so the loss is smaller than the unweighted sum."""
    nf = HopfNormalForm(1.0, 2.0, 0.5)
    batch = _window_batch(nf)
    wrong = _exact_autoencoder(nf, PhaseAmplitudeLatentDynamics(-2.0, -2.0))
    _, aux = PhaseAutoencoderLoss()(wrong, batch)
    encode = jax.vmap(jax.vmap(wrong.encode))
    y = encode(batch["u"])
    y_pred = jax.vmap(wrong.latent_dynamics)(batch["t"], y[:, 0])
    err = jnp.mean(jnp.sum((y - y_pred)[:, 1:, :2] ** 2, axis=-1), axis=0)
    assert jnp.sum(err) > 1.0  # α_k = 1/k regime
    k = jnp.arange(1, err.shape[0] + 1)
    assert_close(aux["pha"], jnp.sum(err / k), rtol=TOL["closed_form"])


def test_loss_is_differentiable_through_the_mlps():
    nf = HopfNormalForm(1.0, 2.0, 0.5)
    model = PhaseAmplitudeAutoencoder(2, mlp_width=8, key=jax.random.key(1))
    grads = eqx.filter_grad(lambda m, b: PhaseAutoencoderLoss()(m, b)[0])(
        model, _window_batch(nf)
    )
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0)) for g in leaves)
    assert grads.latent_dynamics.omega != 0


# --------------------------------------------------------------------- sampler ---
def test_on_cycle_gaussian_sampler():
    """Eqs. (27)–(28): cycle point + γ₂ σ ⊙ ξ; γ₂ = 0 gives points of the cycle;
    seeded; shapes; metadata."""
    nf = BautinNormalForm(1.0, 0.5, 2.0, 1.0)
    on_cycle = OnCycleGaussian.from_normal_form(nf, 100, gamma2=0.0)
    u = on_cycle(jax.random.key(0), 50)
    assert u.shape == (50, 2)
    assert_close(jnp.linalg.norm(u, axis=-1), jnp.ones(50), rtol=TOL["closed_form"])

    noisy = OnCycleGaussian.from_normal_form(nf, 100, gamma2=0.5)
    assert_close(noisy.sigma, jnp.std(noisy.cycle_points, axis=0), rtol=1e-12)
    u1 = noisy(jax.random.key(3), 2000)
    u2 = noisy(jax.random.key(3), 2000)
    assert bool(jnp.all(u1 == u2))
    radial = jnp.linalg.norm(u1, axis=-1) - 1.0
    # σ ≈ 1/√2 per coordinate on the unit circle, so the radial spread is ≈ γ₂ σ
    assert 0.25 < float(jnp.std(radial)) < 0.45
    assert noisy.params()["gamma2"] == 0.5 and noisy.params()["num_cycle_points"] == 100

    with pytest.raises(ValueError):
        OnCycleGaussian(jnp.zeros((1, 2)))
    with pytest.raises(ValueError):
        OnCycleGaussian(jnp.zeros((4, 2)), gamma2=-1.0)


# -------------------------------------------------------------------- training ---
@pytest.mark.slow
def test_phase_autoencoder_learns_the_hopf_phase():
    """End to end on the paper's recipe (N_s ICs on the cycle + 0.5 σ noise, 3 periods,
    K = 20): the learned Θ matches the exact asymptotic phase up to a constant (and an
    orientation) on a grid in the basin. Ψ (up to scale) and κ are reported only."""
    nf = HopfNormalForm(1.0, 2.0, 1.0)
    period = float(nf.period())
    length = 21
    sampler = OnCycleGaussian.from_normal_form(nf, 1000, 0.5)
    source = generate(
        nf,
        sampler,
        jnp.arange(0.0, 3 * period, period / 40),
        256,
        seed=0,
        integration="closed_form",
    )
    batches = iter(windows(source, length, seed=0).batch(128))

    model = PhaseAmplitudeAutoencoder(
        2, PhaseAmplitudeLatentDynamics(1.0, -0.5), key=jax.random.key(0)
    )
    loss = PhaseAutoencoderLoss()
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, batch):
        (_, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, batch)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), opt_state, aux

    for _ in range(1500):
        b = next(batches)
        model, opt_state, aux = step(
            model, opt_state, {"t": jnp.asarray(b["t"]), "u": jnp.asarray(b["u"])}
        )

    theta = jnp.linspace(-jnp.pi, jnp.pi, 64, endpoint=False)
    grid = jnp.concatenate(
        [r * jnp.stack((jnp.cos(theta), jnp.sin(theta)), -1) for r in (0.8, 1.0, 1.25)]
    )
    exact = jax.vmap(nf.to_phase_amplitude)(grid)
    learned = jax.vmap(model.phase)(grid)
    # concentration of the phase difference on the circle, for either orientation
    concentration = max(
        float(jnp.abs(jnp.mean(jnp.exp(1j * (sign * learned - exact[:, 0])))))
        for sign in (1.0, -1.0)
    )
    circular_std = jnp.sqrt(-2 * jnp.log(concentration))
    assert circular_std < 0.25, f"learned phase off by circular std {circular_std:.3f}"
    assert_close(jnp.abs(aux["omega"]), nf.omega(), rtol=0.05)

    psi_corr = jnp.corrcoef(jax.vmap(model.amplitude)(grid), exact[:, 1])[0, 1]
    print(
        f"\nB12 Hopf: circular std of Θ error {circular_std:.3f}; "
        f"corr(Y3, Ψ) = {psi_corr:.2f}; learned ω = {float(aux['omega']):.3f} "
        f"(exact {float(nf.omega()):.3f}); learned κ = {float(aux['kappa']):.3f} "
        f"(exact {float(nf.floquet_exponent()):.3f})"
    )
