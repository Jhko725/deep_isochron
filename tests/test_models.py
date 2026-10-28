"""``AbstractPhaseAmplitudeModel``: both approaches, the invertible conjugacy
``ConjugateLatentDynamics`` and the phase autoencoder ``PhaseAmplitudeAutoencoder``,
satisfy one contract (``model/base.py``), so training and evaluation see them alike.

Checked on each model: shapes, ``phase`` in ``(-π, π]``, ``phase_sensitivity`` is the
gradient of ``phase`` at the cycle point, ``__call__`` returns ``(xt, yt)``. The
consistency laws ``Θ(cycle_point(θ)) == θ`` and ``amplitude == 0`` on the cycle hold
*by construction* for the conjugacy model and only *after training* for the
autoencoder, so they are asserted for the former only; with the identity bijection the
conjugacy model's contract is the normal form's closed forms exactly (design document
§5.2).
"""

import jax
import jax.numpy as jnp
import pytest
from deep_isochron.model import (
    AbstractPhaseAmplitudeModel,
    ConjugateLatentDynamics,
    PhaseAmplitudeAutoencoder,
)
from deep_isochron.model.invertible import BiLipschitzLinear
from deep_isochron.systems import BautinNormalForm

from tests.helpers import assert_close, TOL


NF = BautinNormalForm(1.0, 0.5, 2.0, 0.7)


def _conjugacy(key):
    return ConjugateLatentDynamics(
        NF, BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=key)
    )


def _autoencoder(key):
    return PhaseAmplitudeAutoencoder(2, mlp_width=8, key=key)


@pytest.mark.parametrize("build", [_conjugacy, _autoencoder], ids=["conjugacy", "ae"])
def test_phase_amplitude_contract(build):
    """Each model is an `AbstractPhaseAmplitudeModel` whose `phase` is a scalar in
    ``[-π, π]``, whose `phase_sensitivity` and `phase_gradient` are the autodiff
    gradient of `phase`, and whose call returns finite ``(xt, yt)`` of the right
    shapes."""
    model = build(jax.random.key(0))
    assert isinstance(model, AbstractPhaseAmplitudeModel)
    x = jnp.array([0.4, -1.1])
    theta = model.phase(x)
    assert theta.shape == ()
    assert bool(jnp.abs(theta) <= jnp.pi)
    assert model.amplitude(x).shape == ()

    phi = jnp.asarray(0.9)
    c = model.cycle_point(phi)
    assert c.shape == (2,)

    z = model.phase_sensitivity(phi)
    assert_close(z, jax.grad(model.phase)(c), rtol=TOL["identity"])
    assert_close(
        model.phase_gradient(x), jax.grad(model.phase)(x), rtol=TOL["identity"]
    )

    ts = jnp.linspace(0.0, 1.0, 4)
    xt, yt = model(ts, x)
    assert xt.shape == (4, 2) and yt.shape[0] == 4
    assert bool(jnp.all(jnp.isfinite(xt)))


@pytest.mark.parametrize("build", [_conjugacy, _autoencoder], ids=["conjugacy", "ae"])
def test_isochron_and_latent_radius_have_the_contract_shapes(build):
    """`isochron(θ, r)` returns one point per radius and `latent_radius` a scalar, for
    both model kinds."""
    model = build(jax.random.key(0))
    r = jnp.array([0.5, 1.0, 1.5])
    points = model.isochron(jnp.asarray(0.9), r)
    assert points.shape == (3, 2) and bool(jnp.all(jnp.isfinite(points)))
    assert model.latent_radius(jnp.array([0.4, -1.1])).shape == ()


def test_conjugacy_isochron_is_the_normal_forms_through_the_bijection():
    """``Θ(isochron(θ, r)) == θ`` and ``latent_radius(isochron(θ, r)) == r`` for the
    conjugacy model, by construction through any bijection; ``isochron(θ, 1)`` is
    `cycle_point`; `equilibrium` is ``H⁻¹(0)``, so its phase is ``nan``."""
    model = _conjugacy(jax.random.key(0))
    theta = jnp.asarray(-2.3)
    r = jnp.array([0.6, 1.0, 1.4])
    points = model.isochron(theta, r)
    phases = jax.vmap(model.phase)(points)
    assert_close(phases, jnp.full(3, theta), rtol=TOL["identity"])
    assert_close(jax.vmap(model.latent_radius)(points), r, rtol=TOL["identity"])
    assert_close(points[1], model.cycle_point(theta), rtol=TOL["identity"])
    q = model.equilibrium()
    assert_close(model.bijection(q), jnp.zeros(2), atol=TOL["identity"])
    assert bool(jnp.isnan(model.phase(q)))


def test_conjugacy_cycle_and_amplitude_are_consistent_by_construction():
    """``Θ(cycle_point(θ)) == θ`` and ``Ψ(cycle_point(θ)) == 0`` through any bijection,
    and ``xt[0]`` is ``x0`` (the bijection round-trips)."""
    model = _conjugacy(jax.random.key(0))
    for phi in (jnp.asarray(0.9), jnp.asarray(-3.0)):
        c = model.cycle_point(phi)
        assert_close(model.phase(c), phi, rtol=TOL["vector_roundtrip"])
        assert_close(model.amplitude(c), 0.0, atol=TOL["vector_roundtrip"])
    x = jnp.array([0.4, -1.1])
    xt, _ = model(jnp.linspace(0.0, 1.0, 4), x)
    assert_close(xt[0], x, rtol=TOL["vector_roundtrip"])


def test_conjugacy_with_identity_bijection_is_the_normal_form():
    """With ``H = id`` (``BiLipschitzLinear`` is the identity at init) the contract is
    the normal form's: ``Θ = Θ_NF``, ``Ψ = Ψ_NF``, the cycle is the unit circle, and the
    prediction is the normal form's (closed-form) flow."""
    model = _conjugacy(jax.random.key(1))
    x = jnp.array([1.3, 0.2])
    assert_close(model.phase(x), NF.phase(x), rtol=TOL["identity"])
    assert_close(model.amplitude(x), NF.amplitude(x), rtol=TOL["identity"])
    phi = jnp.asarray(-2.0)
    assert_close(
        model.cycle_point(phi), jnp.array([jnp.cos(phi), jnp.sin(phi)]), rtol=1e-12
    )
    ts = jnp.linspace(0.0, 2.0, 5)
    xt, yt = model(ts, x)
    assert_close(xt, NF.flow(ts, x).ys, rtol=TOL["identity"])
    assert_close(yt, xt, rtol=TOL["identity"])
