"""Laws every ``AbstractBijection`` must satisfy.

  1. round trip        f⁻¹(f(x)) == x  and  f(f⁻¹(y)) == y
  2. identity at init  from_unconstrained(0) == id  /  fresh vector module == id
  3. orientation       det Df(x) > 0 — scalar case: f'(x) > 0
  4. finiteness        f, f⁻¹ finite on the working domain
  5. jacobian          Df⁻¹(f(x)) · Df(x) == I
  6. pytree hygiene    partition/combine round trip; num_trainable_params
  7. trainability      one aggressive optimizer step keeps every law (raw leaves are
                       unconstrained, so no step can leave the constrained set)

What gets tested is declared in ``tests/registry.py``; how instances are drawn is
in ``tests/strategies.py``.  Two test styles are used deliberately:

* ``@pytest.mark.parametrize(name) + @given(st.data())`` — one pytest ID per
  bijection, draws labeled.  Needed because ``@given`` binds its strategy at
  decoration time while ``name`` only exists at collection time (same shape as
  TFP's ``testBijector``).
* ``@given(f=any_scalar_bijection())`` + ``@example(...)`` — for laws on
  measure-zero sets where pinned regression inputs matter more than per-bijection
  IDs; ``st.data`` tests cannot take ``@example``. Used in ``test_splines.py``.

Spline-specific laws (knots, tails, join regularity) live in ``test_splines.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from deep_isochron.model.invertible import (
    CouplingFlow,
    SequentialINN,
)
from hypothesis import given, strategies as st

from tests.helpers import (
    adam_step,
    assert_close,
    inverse_jacobian_product,
    jacobian_dets,
    roundtrip,
    TOL,
)
from tests.registry import SCALAR_TEMPLATES, VECTOR_BUILDERS
from tests.strategies import (
    point_batches,
    scalar_bijections,
    seeds,
    vector_bijections,
)


SCALAR_IDS = list(SCALAR_TEMPLATES)
VECTOR_IDS = list(VECTOR_BUILDERS)


# ---------------------------------------------------------------- scalar laws -----
@pytest.mark.parametrize("name", SCALAR_IDS)
def test_scalar_identity_at_zero(name):
    f = SCALAR_TEMPLATES[name].identity_like()
    x = jnp.linspace(-4, 4, 41)
    assert_close(
        jax.vmap(f)(x),
        x,
        atol=TOL["identity"],
        msg=f"{name}: from_unconstrained(0) != id",
    )


@pytest.mark.parametrize("name", SCALAR_IDS)
def test_scalar_num_params_matches_leaves(name):
    """A standalone scalar bijection's raw vector is exactly its trainable state."""
    t = SCALAR_TEMPLATES[name]
    assert t.num_trainable_params == t.num_params
    if t.raw is not None:  # a ScalarChain's raw lives in its members
        assert t.raw.shape == (t.num_params,)


@pytest.mark.parametrize("name", SCALAR_IDS)
def test_scalar_template_has_no_trainable_state(name):
    """A template (raw=None) is hashable and contributes nothing to training."""
    t = eqx.partition(SCALAR_TEMPLATES[name], eqx.is_array)[1]
    assert t.num_trainable_params == 0
    hash(t)
    assert t.identity_like().num_params == SCALAR_TEMPLATES[name].num_params


@pytest.mark.parametrize("name", SCALAR_IDS)
@given(data=st.data())
def test_scalar_round_trip(name, data):
    f = data.draw(scalar_bijections(SCALAR_TEMPLATES[name]), label="bijection")
    x = data.draw(point_batches(1), label="x")[:, 0]
    y, x_rt, y_rt = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward not finite"
    assert_close(
        x_rt,
        x,
        rtol=TOL["scalar_roundtrip"],
        atol=TOL["scalar_roundtrip"],
        msg=f"{name}: f⁻¹∘f",
    )
    assert_close(
        y_rt,
        x,
        rtol=TOL["scalar_roundtrip"],
        atol=TOL["scalar_roundtrip"],
        msg=f"{name}: f∘f⁻¹",
    )


@pytest.mark.parametrize("name", SCALAR_IDS)
@given(data=st.data())
def test_scalar_strictly_increasing(name, data):
    f = data.draw(scalar_bijections(SCALAR_TEMPLATES[name]), label="bijection")
    x = data.draw(point_batches(1), label="x")[:, 0]
    assert jnp.all(jax.vmap(jax.grad(f))(x) > 0), f"{name}: non-positive derivative"


@pytest.mark.parametrize("name", SCALAR_IDS)
@given(data=st.data())
def test_scalar_jacobian_consistent_with_inverse(name, data):
    f = data.draw(scalar_bijections(SCALAR_TEMPLATES[name]), label="bijection")
    x = data.draw(point_batches(1), label="x")[:, 0]
    dfdx = jax.vmap(f.jacobian)(x)
    dinv_dy = jax.vmap(jax.grad(f.inverse))(jax.vmap(f)(x))
    assert_close(
        dfdx * dinv_dy,
        jnp.ones_like(x),
        rtol=TOL["jacobian"],
        atol=TOL["jacobian"],
        msg=f"{name}: f'·(f⁻¹)' != 1",
    )


# ---------------------------------------------------------------- vector laws -----
@pytest.mark.parametrize("name", VECTOR_IDS)
def test_vector_identity_at_init(name, key):
    """Universal: every vector bijection is the identity map when freshly built."""
    f = VECTOR_BUILDERS[name](key)
    x = jax.random.normal(jax.random.key(1), (32, f.dim))
    assert_close(
        jax.vmap(f)(x),
        x,
        atol=TOL["identity"],
        msg=f"{name}: init",
    )


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_round_trip_random_weights(name, data):
    f = data.draw(vector_bijections(name), label="bijection")
    x = data.draw(point_batches(f.dim), label="x")
    y, x_rt, y_rt = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward not finite"
    assert_close(
        x_rt,
        x,
        rtol=TOL["vector_roundtrip"],
        atol=TOL["vector_roundtrip"],
        msg=f"{name}: f⁻¹∘f",
    )
    assert_close(
        y_rt,
        x,
        rtol=TOL["vector_roundtrip"],
        atol=TOL["vector_roundtrip"],
        msg=f"{name}: f∘f⁻¹",
    )


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_orientation_preserving(name, data):
    """Every vector bijection is orientation-preserving at every point of its parameter
    space (the former ``InvertibleLinear``, an unconstrained matrix, was the one
    exception and was removed for that reason)."""
    f = data.draw(vector_bijections(name), label="bijection")
    x = data.draw(point_batches(f.dim), label="x")
    dets = jacobian_dets(f, x)
    assert jnp.all(dets > 0), f"{name}: det Df <= 0 (min {float(dets.min()):.3e})"


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_orientation_preserving_at_init(name, data):
    """Regression: a QR-based rotation init once returned a reflection (det = -1)."""
    f = data.draw(vector_bijections(name, perturbed=False), label="bijection")
    x = jax.random.normal(jax.random.key(2), (8, f.dim))
    assert jnp.all(jacobian_dets(f, x) > 0), f"{name}: orientation-reversing at init"


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_jacobian_consistent_with_inverse(name, data):
    f = data.draw(vector_bijections(name), label="bijection")
    x = data.draw(point_batches(f.dim), label="x")
    P = inverse_jacobian_product(f, x)
    assert_close(
        P,
        jnp.broadcast_to(jnp.eye(f.dim), P.shape),
        rtol=TOL["jacobian"],
        atol=TOL["jacobian"],
        msg=f"{name}: Df⁻¹·Df != I",
    )


@given(data=st.data())
def test_sequential_inn_composes_inverse(data):
    """Composition semantics: inverse applies the members' inverses in reverse order.
    Three mildly perturbed dim-2 layers — every layer alone round-trips to ~1e-10, but a
    deep chain of aggressively perturbed layers amplifies that by its Lipschitz constant
    (intermediate values reach ~1e2), which is conditioning, not composition."""
    names = [n for n in VECTOR_IDS if VECTOR_BUILDERS[n](jax.random.key(0)).dim == 2]
    chosen = data.draw(
        st.lists(st.sampled_from(names), min_size=3, max_size=3, unique=True),
        label="layers",
    )
    layers = [data.draw(vector_bijections(n, scale=0.1), label=n) for n in chosen]
    f = SequentialINN(layers)
    x = data.draw(point_batches(2), label="x")
    _, x_rt, _ = roundtrip(f, x)
    assert_close(x_rt, x, rtol=TOL["vector_roundtrip"], atol=TOL["vector_roundtrip"])
    # order: f⁻¹ really is the reversed chain of member inverses
    y = jax.vmap(f)(x)
    z = y
    for l in layers[::-1]:
        z = jax.vmap(l.inverse)(z)
    assert_close(jax.vmap(f.inverse)(y), z, rtol=0, atol=0)


# ------------------------------------------------------------- pytree hygiene -----
@pytest.mark.parametrize("name", SCALAR_IDS + VECTOR_IDS)
def test_partition_combine_round_trip(name, key):
    f = (
        SCALAR_TEMPLATES[name]
        if name in SCALAR_TEMPLATES
        else VECTOR_BUILDERS[name](key)
    )
    arrays, static = eqx.partition(f, eqx.is_array)
    assert eqx.tree_equal(eqx.combine(arrays, static), f)
    n = sum(l.size for l in jax.tree.leaves(eqx.filter(f, eqx.is_inexact_array)))
    assert f.num_trainable_params == n


def test_coupling_does_not_count_generated_params(key):
    """CouplingFlow's trainable state is the conditioner MLP, not the inner template."""
    t = next(iter(SCALAR_TEMPLATES.values()))
    cf = CouplingFlow(dim=2, bijection=t, mlp_width=8, mlp_depth=1, key=key)
    mlp_size = sum(
        l.size
        for l in jax.tree.leaves(eqx.filter(cf.conditioner, eqx.is_inexact_array))
    )
    assert cf.num_trainable_params == mlp_size


# -------------------------------------------------------------- trainability ------
@pytest.mark.parametrize("name", SCALAR_IDS)
@given(seed=seeds, x=point_batches(1))
def test_scalar_standalone_training_keeps_validity(name, seed, x):
    """An aggressive Adam step on a standalone scalar bijection cannot leave the
    constrained set: round trip and monotonicity still hold afterwards."""
    f = adam_step(SCALAR_TEMPLATES[name], jax.random.key(seed))
    x = x[:, 0]
    y, x_rt, _ = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward not finite after a step"
    assert_close(
        x_rt,
        x,
        rtol=TOL["scalar_roundtrip"],
        atol=TOL["scalar_roundtrip"],
        msg=f"{name}: round trip after a step",
    )
    assert jnp.all(jax.vmap(jax.grad(f))(x) > 0), f"{name}: not increasing after a step"


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(seed=seeds, x=point_batches(2))
def test_vector_standalone_training_keeps_validity(name, seed, x):
    f = adam_step(
        VECTOR_BUILDERS[name](jax.random.key(seed)), jax.random.key(seed + 1), lr=0.1
    )
    x = x[:, : f.dim] if f.dim <= 2 else jnp.pad(x, ((0, 0), (0, f.dim - 2)))
    y, x_rt, _ = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward not finite after a step"
    assert_close(
        x_rt,
        x,
        rtol=TOL["vector_roundtrip"],
        atol=TOL["vector_roundtrip"],
        msg=f"{name}: round trip after a step",
    )


@pytest.mark.parametrize("name", SCALAR_IDS)
def test_true_gradient_step_decreases_loss(name):
    """Plumbing check complementing the random-direction law: gradients flow through
    ``constrain`` to ``raw``, and a true gradient step on a fit-to-target loss lowers
    it."""
    import optax

    f = SCALAR_TEMPLATES[name].from_unconstrained(
        0.3 * jax.random.normal(jax.random.key(0), (SCALAR_TEMPLATES[name].num_params,))
    )
    x = jnp.linspace(-1.5, 1.5, 16)
    target = 1.3 * x + 0.2  # a map the templates can approach but not represent exactly

    def loss(m):
        return jnp.mean((jax.vmap(m)(x) - target) ** 2)

    params, static = eqx.partition(f, eqx.is_inexact_array)
    grads = eqx.filter_grad(loss)(f)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)), (
        "non-finite grad"
    )
    opt = optax.sgd(1e-2)
    updates, _ = opt.update(
        eqx.filter(grads, eqx.is_inexact_array), opt.init(params), params
    )
    g = eqx.combine(eqx.apply_updates(params, updates), static)
    assert loss(g) < loss(f), f"{name}: a small gradient step did not decrease the loss"
