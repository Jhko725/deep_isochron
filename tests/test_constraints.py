"""Laws of the constraint primitives (``invertible/constraints.py``).

  1. inverse       c.inverse(c(raw)) == raw         (Widths: in the mean-zero gauge)
  2. at_zero       c(0) == at_zero                  (Widths: equal widths)
  3. image         c(raw) lies in the constrained set
  4. finiteness    c and its gradient are finite for |raw| <= 30

Also pins ``misc.squashed_exp`` / ``inv_squashed_exp`` as mutual inverses for every
``a``
(the ``a`` argument used to be ignored).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.misc import inv_squashed_exp, squashed_exp
from deep_isochron.model.invertible.constraints import (
    Arcsinh,
    Free,
    Interval,
    Positive,
    Widths,
)
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from tests.helpers import assert_close


raw = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
raw_extreme = st.floats(-30.0, 30.0, allow_nan=False, allow_infinity=False)


def raw_vectors(n, elements=raw):
    return hnp.arrays(np.float64, (n,), elements=elements).map(jnp.asarray)


# Elementwise primitives: name -> (constraint, membership predicate)
ELEMENTWISE = {
    "free": (Free(), lambda v: jnp.isfinite(v)),
    "positive": (Positive(), lambda v: v > 0),
    "positive (eps=0.1, at_zero=0.9)": (Positive(0.1, 0.9), lambda v: v > 0.1),
    "interval (-1, 8)": (
        Interval(-1.0, 8.0, at_zero=0.0),
        lambda v: (v > -1) & (v < 8),
    ),
    "interval (0.2, 0.3)": (Interval(0.2, 0.3), lambda v: (v > 0.2) & (v < 0.3)),
    "arcsinh": (Arcsinh(), lambda v: jnp.isfinite(v)),
}
AT_ZERO = {
    "free": 0.0,
    "positive": 1.0,
    "positive (eps=0.1, at_zero=0.9)": 0.9,
    "interval (-1, 8)": 0.0,
    "interval (0.2, 0.3)": 0.25,
    "arcsinh": 0.0,
}


@pytest.mark.parametrize("name", ELEMENTWISE)
@given(r=raw_vectors(8))
def test_elementwise_inverse(name, r):
    c, _ = ELEMENTWISE[name]
    assert_close(
        c.inverse(c(r)), r, rtol=1e-9, atol=1e-9, msg=f"{name}: inverse∘forward"
    )


@pytest.mark.parametrize("name", ELEMENTWISE)
def test_elementwise_at_zero(name):
    c, _ = ELEMENTWISE[name]
    assert_close(
        c(jnp.zeros(4)), jnp.full(4, AT_ZERO[name]), atol=1e-12, msg=f"{name}: c(0)"
    )


@pytest.mark.parametrize("name", ELEMENTWISE)
@given(r=raw_vectors(8, raw_extreme))
def test_elementwise_image_and_finiteness(name, r):
    c, member = ELEMENTWISE[name]
    v = c(r)
    assert jnp.all(jnp.isfinite(v)), f"{name}: non-finite image"
    assert jnp.all(member(v)), f"{name}: image outside the constrained set"
    g = jax.vmap(jax.grad(lambda t: c(t[None])[0]))(r)
    assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite gradient"


# ----------------------------------------------------------------- Widths ---------
@given(
    r=raw_vectors(6),
    total=st.floats(0.1, 10.0, allow_nan=False),
    min_rel=st.sampled_from([0.0, 1e-3, 0.05]),
)
def test_widths_image(r, total, min_rel):
    w = Widths(total, min_rel)(r)
    assert jnp.all(w >= min_rel * total - 1e-12)
    assert_close(jnp.sum(w), total, rtol=1e-12, atol=1e-12)


@given(total=st.floats(0.1, 10.0, allow_nan=False), n=st.integers(1, 12))
def test_widths_at_zero_is_equal(total, n):
    w = Widths(total, 1e-3)(jnp.zeros(n))
    assert_close(w, jnp.full(n, total / n), rtol=1e-12, atol=1e-12)


@given(r=raw_vectors(6), total=st.floats(0.1, 10.0, allow_nan=False))
def test_widths_inverse_in_mean_zero_gauge(r, total):
    c = Widths(total, 1e-3)
    r_centred = r - jnp.mean(r)
    assert_close(c.inverse(c(r_centred)), r_centred, rtol=1e-8, atol=1e-8)
    # gauge consistency: forward is invariant to the constant that inverse drops
    assert_close(c(r), c(r_centred), rtol=1e-12, atol=1e-12)


def test_widths_rejects_impossible_floor():
    with pytest.raises(ValueError):
        Widths(1.0, 0.3)(jnp.zeros(4))  # 4 * 0.3 >= 1


def test_positive_and_interval_validate_at_zero():
    with pytest.raises(ValueError):
        Positive(eps=1.0, at_zero=0.5)
    with pytest.raises(ValueError):
        Interval(0.0, 1.0, at_zero=1.0)


# ------------------------------------------------------------ misc.squashed_exp ---
@pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
@given(x=raw_vectors(8))
def test_squashed_exp_inverse(a, x):
    y = squashed_exp(x, a)
    assert jnp.all((y > jnp.exp(-a)) & (y < jnp.exp(a)))
    assert_close(inv_squashed_exp(y, a), x, rtol=1e-9, atol=1e-9, msg=f"a={a}")
