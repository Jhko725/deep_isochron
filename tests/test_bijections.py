"""Laws every ``AbstractBijection`` must satisfy.

  1. round trip        f⁻¹(f(x)) == x  and  f(f⁻¹(y)) == y
  2. identity at init  from_unconstrained(0) == id  /  fresh vector module == id
  3. orientation       det Df(x) > 0 — scalar case: f'(x) > 0
  4. finiteness        f, f⁻¹ finite on the working domain
  5. jacobian          Df⁻¹(f(x)) · Df(x) == I
  6. pytree hygiene    partition/combine round trip; num_trainable_params

What gets tested is declared in ``tests/registry.py``; how instances are drawn is
in ``tests/strategies.py``.  Two test styles are used deliberately:

* ``@pytest.mark.parametrize(name) + @given(st.data())`` — one pytest ID per
  bijection, draws labelled.  Needed because ``@given`` binds its strategy at
  decoration time while ``name`` only exists at collection time (same shape as
  TFP's ``testBijector``).
* ``@given(f=any_scalar_bijection())`` + ``@example(...)`` — for laws on
  measure-zero sets (knots, range endpoints) where pinned regression inputs matter
  more than per-bijection IDs; ``st.data`` tests cannot take ``@example``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from deep_isochron.model.invertible import (
    AbstractSpline,
    CouplingFlow,
    SequentialINN,
)
from hypothesis import example, given, strategies as st

from tests.helpers import (
    assert_close,
    inverse_jacobian_product,
    jacobian_dets,
    roundtrip,
)
from tests.registry import (
    IDENTITY_AT_INIT,
    IDENTITY_TOL,
    SCALAR_TEMPLATES,
    VECTOR_BUILDERS,
)
from tests.strategies import (
    any_scalar_bijection,
    any_spline_template,
    point_batches,
    scalar_bijections,
    vector_bijections,
)


SCALAR_IDS = list(SCALAR_TEMPLATES)
VECTOR_IDS = list(VECTOR_BUILDERS)
KNOTTED = [n for n, t in SCALAR_TEMPLATES.items() if isinstance(t, AbstractSpline)]
KNOTTED_C1 = [n for n in KNOTTED if hasattr(SCALAR_TEMPLATES[n], "knot_derivs")]


def _zero_instance(name):
    t = SCALAR_TEMPLATES[name]
    return t.from_unconstrained(jnp.zeros(t.num_params))


# ---------------------------------------------------------------- scalar laws -----
@pytest.mark.parametrize("name", SCALAR_IDS)
def test_scalar_identity_at_zero(name):
    f = _zero_instance(name)
    x = jnp.linspace(-4, 4, 41)
    assert_close(
        jax.vmap(f)(x),
        x,
        atol=IDENTITY_TOL.get(name, 1e-12),
        msg=f"{name}: from_unconstrained(0) != id",
    )


@pytest.mark.parametrize("name", SCALAR_IDS)
def test_scalar_num_params_matches_leaves(name):
    """A standalone scalar bijection's raw vector is exactly its trainable state."""
    t = SCALAR_TEMPLATES[name]
    assert t.num_trainable_params == t.num_params


@pytest.mark.parametrize("name", SCALAR_IDS)
@given(data=st.data())
def test_scalar_round_trip(name, data):
    f = data.draw(scalar_bijections(SCALAR_TEMPLATES[name]), label="bijection")
    x = data.draw(point_batches(1), label="x")[:, 0]
    y, x_rt, y_rt = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward not finite"
    assert_close(x_rt, x, rtol=1e-8, atol=1e-8, msg=f"{name}: f⁻¹∘f")
    assert_close(y_rt, x, rtol=1e-8, atol=1e-8, msg=f"{name}: f∘f⁻¹")


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
        rtol=1e-7,
        atol=1e-7,
        msg=f"{name}: f'·(f⁻¹)' != 1",
    )


# ------------------------------------------------ scalar: knots and boundaries ----
# Measure-zero sets (knots, range endpoints) that property tests hit only by luck.
# Style: any_scalar_bijection() + @example pins, so regressions stay pinned.
# Every AbstractSpline in the registry gets these for free; spline-specific laws
# (regularity of the tail joins, oracles) live in test_splines.py.
@given(f=any_scalar_bijection(KNOTTED))
@example(f=_zero_instance(KNOTTED[0]))
def test_knots_round_trip(f):
    xk = f.xs  # includes both range endpoints
    _, x_rt, _ = roundtrip(f, xk)
    assert_close(x_rt, xk, atol=1e-10, msg="round trip at knots")


@given(f=any_scalar_bijection(KNOTTED))
@example(f=_zero_instance(KNOTTED[0]))
def test_knots_interpolated_and_monotone(f):
    assert_close(jax.vmap(f)(f.xs), f.ys, atol=1e-10, msg="f(x_k) != y_k")
    lo, hi = float(f.xs[0]), float(f.xs[-1])
    x = jnp.sort(jnp.concatenate([jnp.linspace(lo - 1, hi + 1, 2001), f.xs]))
    x = x[
        jnp.concatenate([jnp.array([True]), jnp.diff(x) > 1e-9])
    ]  # drop near-duplicate grid/knot points
    assert jnp.all(jnp.diff(jax.vmap(f)(x)) > 0), "not monotone across knots"


@given(f=any_scalar_bijection(KNOTTED_C1))
@example(f=_zero_instance(KNOTTED_C1[0]))
def test_knot_derivatives(f):
    """f'(x_k) == knot_derivs[k] at every knot, boundaries included (identity tails
    join C¹)."""
    assert_close(
        jax.vmap(jax.grad(f))(f.xs),
        f.knot_derivs,
        rtol=1e-9,
        atol=1e-9,
        msg="f'(x_k) != d_k",
    )


@given(f=any_spline_template().flatmap(scalar_bijections), x=point_batches(1))
def test_spline_any_bin_count_round_trip(f, x):
    """Round trip for drawn spline class, bin count and range (not just the
    registry's configs)."""
    x = x[:, 0]
    _, x_rt, _ = roundtrip(f, x)
    assert_close(x_rt, x, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------- vector laws -----
@pytest.mark.parametrize("name", VECTOR_IDS)
def test_vector_identity_at_init(name, key):
    if name not in IDENTITY_AT_INIT:
        pytest.skip("not identity-at-init by design")
    f = VECTOR_BUILDERS[name](key)
    x = jax.random.normal(jax.random.key(1), (32, f.dim))
    assert_close(
        jax.vmap(f)(x), x, atol=IDENTITY_TOL.get(name, 1e-12), msg=f"{name}: init"
    )


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_round_trip_random_weights(name, data):
    f = data.draw(vector_bijections(name), label="bijection")
    x = data.draw(point_batches(f.dim), label="x")
    y, x_rt, y_rt = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward not finite"
    assert_close(x_rt, x, rtol=1e-7, atol=1e-7, msg=f"{name}: f⁻¹∘f")
    assert_close(y_rt, x, rtol=1e-7, atol=1e-7, msg=f"{name}: f∘f⁻¹")


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_orientation_preserving(name, data):
    f = data.draw(vector_bijections(name), label="bijection")
    x = data.draw(point_batches(f.dim), label="x")
    dets = jacobian_dets(f, x)
    assert jnp.all(dets > 0), f"{name}: det Df <= 0 (min {float(dets.min()):.3e})"


@pytest.mark.parametrize("name", VECTOR_IDS)
@given(data=st.data())
def test_vector_orientation_preserving_at_init(name, data):
    """Regression: QR-based init of InvertibleLinear returned a reflection (det = -1)
    ."""
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
        rtol=1e-6,
        atol=1e-6,
        msg=f"{name}: Df⁻¹·Df != I",
    )


@given(data=st.data())
def test_sequential_inn_composes_inverse(data):
    layers = [data.draw(vector_bijections(n), label=n) for n in VECTOR_IDS]
    f = SequentialINN([l for l in layers if l.dim == 2])
    x = data.draw(point_batches(2), label="x")
    _, x_rt, _ = roundtrip(f, x)
    assert_close(x_rt, x, rtol=1e-7, atol=1e-7)


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
        l.size for l in jax.tree.leaves(eqx.filter(cf.mlp, eqx.is_inexact_array))
    )
    assert cf.num_trainable_params == mlp_size
