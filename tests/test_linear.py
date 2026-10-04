"""Laws specific to ``BiLipschitzLinear`` (``invertible/linear.py``). Shared laws (round
trip, identity at init, orientation, Jacobian) run through the registry.

  1. parameter space   for *any* raw leaves: U, V in SO(dim); s in (1/L, L); hence
                       1/L <= sigma(W) <= L and det W > 0
  2. params            ``weight`` is built from ``params``; its singular values are s
  3. init              ``"identity"`` is I; ``"rotation"`` is orthogonal with s = 1
  4. validation        ``max_lipschitz <= 1`` and an unknown ``init`` are rejected
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from deep_isochron.model.invertible import BiLipschitzLinear
from hypothesis import given, strategies as st

from tests.helpers import assert_close, TOL
from tests.strategies import EXTREME_RAW_BOUND, magnitudes, raw_vectors


@st.composite
def layers(draw) -> BiLipschitzLinear:
    """A layer whose raw leaves are set directly to arbitrary values (singular-value
    leaves in the extreme regime): the laws must hold at every point of parameter
    space, not only near init."""
    d = draw(st.integers(2, 4), label="dim")
    L = draw(st.sampled_from([1.5, 2.0, 10.0]), label="L")
    f = BiLipschitzLinear(dim=d, max_lipschitz=L, key=jax.random.key(0))
    raw_U = draw(raw_vectors(d * d), label="raw_U").reshape(d, d)
    raw_V = draw(raw_vectors(d * d), label="raw_V").reshape(d, d)
    raw_s = draw(raw_vectors(d, magnitudes(1e-6, EXTREME_RAW_BOUND)), label="raw_s")
    bias = draw(raw_vectors(d), label="bias")
    return eqx.tree_at(
        lambda m: (m.raw_U, m.raw_V, m.raw_s, m.bias), f, (raw_U, raw_V, raw_s, bias)
    )


@given(f=layers())
def test_parameter_space_guarantees(f):
    L = f.max_lipschitz
    p = f.params
    eye = jnp.eye(f.dim)
    tol = TOL["closed_form"]
    assert_close(p.U @ p.U.T, eye, rtol=tol, atol=tol, msg="U not orthogonal")
    assert_close(p.V @ p.V.T, eye, rtol=tol, atol=tol, msg="V not orthogonal")
    assert jnp.linalg.det(p.U) > 0 and jnp.linalg.det(p.V) > 0, "not in SO(dim)"
    assert jnp.all((p.s > 1 / L) & (p.s < L)), "singular values outside (1/L, L)"
    sigma = jnp.linalg.svd(f.weight, compute_uv=False)
    assert jnp.all((sigma > 1 / L - tol) & (sigma < L + tol))
    assert jnp.linalg.det(f.weight) > 0


@given(f=layers())
def test_weight_is_built_from_params(f):
    p = f.params
    tol = TOL["identity"]
    assert_close(f.weight, (p.U * p.s) @ p.V.T, rtol=tol, atol=tol)
    sigma = jnp.sort(jnp.linalg.svd(f.weight, compute_uv=False))
    tol = TOL["closed_form"]
    assert_close(sigma, jnp.sort(p.s), rtol=tol, atol=tol)


@pytest.mark.parametrize("dim", [2, 3])
def test_identity_init(dim):
    f = BiLipschitzLinear(dim=dim, max_lipschitz=2.0, key=jax.random.key(3))
    assert_close(f.weight, jnp.eye(dim), atol=TOL["identity"])
    assert_close(f.params.s, jnp.ones(dim), atol=TOL["identity"])


@pytest.mark.parametrize("seed", range(5))
def test_rotation_init_is_orthogonal_with_unit_singular_values(seed):
    f = BiLipschitzLinear(
        dim=3, max_lipschitz=2.0, init="rotation", key=jax.random.key(seed)
    )
    W = f.weight
    tol = TOL["closed_form"]
    assert_close(W @ W.T, jnp.eye(3), rtol=tol, atol=tol)
    assert jnp.linalg.det(W) > 0
    assert_close(f.params.s, jnp.ones(3), atol=TOL["identity"])
    assert not jnp.allclose(W, jnp.eye(3))  # actually rotated


def test_validation():
    with pytest.raises(ValueError):
        BiLipschitzLinear(dim=2, max_lipschitz=1.0, key=jax.random.key(0))
    # Under the beartype import hook the Literal annotation rejects this first.
    with pytest.raises((ValueError, TypeError)):
        BiLipschitzLinear(dim=2, max_lipschitz=2.0, init="haar", key=jax.random.key(0))


@pytest.mark.parametrize("dim", [2, 3])
def test_rotations_have_no_conditionals(dim):
    """ADR-0010: the compiled parameter map has no ``conditional`` — and for ``dim = 2``
    no ``while`` either. ``jax.scipy.linalg.expm`` put 16 ``lax.cond``s per rotation in
    the step, each a device-to-host round trip on GPU (0.65 ms per 2×2 call on a V100).
    For ``dim > 2`` the LU solve keeps a ``dim``-trip pivot loop with no conditional."""
    f = BiLipschitzLinear(
        dim=dim, max_lipschitz=2.0, init="rotation", key=jax.random.key(0)
    )
    hlo = jax.jit(lambda m: m.params).lower(f).compile().as_text()
    assert " conditional(" not in hlo
    if dim == 2:
        assert " while(" not in hlo


def test_cayley_dim2_closed_form_matches_the_solve():
    from deep_isochron.model.invertible.linear import cayley

    for a in (0.0, 0.3, -2.0, 50.0):
        skew = jnp.array([[0.0, -a], [a, 0.0]])
        eye = jnp.eye(2)
        q = cayley(skew)
        ref = jnp.linalg.solve(eye + skew, eye - skew)
        assert_close(q, ref, atol=TOL["closed_form"])
        assert_close(q @ q.T, eye, atol=TOL["closed_form"])
        assert jnp.linalg.det(q) > 0
