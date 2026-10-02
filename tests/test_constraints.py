"""Laws of the constraint primitives (``invertible/constraints.py``), generic over every
primitive in ``PRIMITIVES``:

  1. inverse       c.inverse(c(raw)) == raw         (Widths: in the mean-zero gauge)
  2. at_zero       c(0) == at_zero                  (Widths: equal widths)
  3. image         c.is_constrained(c(raw)) for raw in the extreme regime
  4. finiteness    c and its gradient are finite for |raw| <= EXTREME_RAW_BOUND

``is_constrained`` is part of the ``Constraint`` contract, so adding a primitive means
adding one line to ``PRIMITIVES`` and nothing else here.

Also pins ``misc.squashed_exp`` / ``inv_squashed_exp`` as mutual inverses for every
``a`` (the ``a`` argument used to be ignored).
"""

import jax
import jax.numpy as jnp
import pytest
from deep_isochron.misc import inv_squashed_exp, squashed_exp
from deep_isochron.model.invertible.constraints import (
    arcsinh,
    BoundedPositive,
    free,
    Interval,
    Positive,
    Widths,
)
from hypothesis import given, strategies as st

from tests.helpers import assert_close, TOL
from tests.strategies import EXTREME_RAW_BOUND, magnitudes, raw_vectors


# name -> primitive. Elementwise ones also appear in AT_ZERO with their neutral value.
PRIMITIVES = {
    "free": free,
    "arcsinh": arcsinh,
    "positive": Positive(),
    "positive (eps=0.1, at_zero=0.9)": Positive(0.1, 0.9),
    "bounded_positive": BoundedPositive(),
    "bounded_positive (eps=0.01, at_zero=0.3)": BoundedPositive(0.01, 0.3),
    "interval (-1, 8)": Interval(-1.0, 8.0, at_zero=0.0),
    "interval (0.2, 0.3)": Interval(0.2, 0.3),
    "widths (total=2)": Widths(2.0, 1e-3),
    "widths (total=0.5, min_rel=0.05)": Widths(0.5, 0.05),
}
AT_ZERO = {
    "free": 0.0,
    "arcsinh": 0.0,
    "positive": 1.0,
    "positive (eps=0.1, at_zero=0.9)": 0.9,
    "bounded_positive": 1.0,
    "bounded_positive (eps=0.01, at_zero=0.3)": 0.3,
    "interval (-1, 8)": 0.0,
    "interval (0.2, 0.3)": 0.25,
}
WIDTHS = [n for n in PRIMITIVES if n.startswith("widths")]
ELEMENTWISE = [n for n in PRIMITIVES if n not in WIDTHS]

extreme_raw = magnitudes(1e-6, EXTREME_RAW_BOUND)


@pytest.mark.parametrize("name", ELEMENTWISE)
@given(r=raw_vectors(8))
def test_inverse(name, r):
    c, tol = PRIMITIVES[name], TOL["closed_form"]
    assert_close(c.inverse(c(r)), r, rtol=tol, atol=tol, msg=f"{name}: inverse∘forward")


@pytest.mark.parametrize("name", WIDTHS)
@given(r=raw_vectors(6))
def test_widths_inverse_in_mean_zero_gauge(name, r):
    c = PRIMITIVES[name]
    r_centred = r - jnp.mean(r)
    tol = TOL["closed_form"]
    assert_close(c.inverse(c(r_centred)), r_centred, rtol=tol, atol=tol)
    # gauge consistency: forward is invariant to the constant that inverse drops
    tol = TOL["identity"]
    assert_close(c(r), c(r_centred), rtol=tol, atol=tol)


@pytest.mark.parametrize("name", ELEMENTWISE)
def test_at_zero(name):
    c = PRIMITIVES[name]
    assert_close(c(jnp.zeros(4)), jnp.full(4, AT_ZERO[name]), atol=TOL["identity"])
    assert c.is_constrained(c(jnp.zeros(4)))


@pytest.mark.parametrize("name", WIDTHS)
@given(n=st.integers(1, 12))
def test_widths_at_zero_is_equal(name, n):
    c = PRIMITIVES[name]
    w, tol = c(jnp.zeros(n)), TOL["identity"]
    assert_close(w, jnp.full(n, c.total / n), rtol=tol, atol=tol)
    assert c.is_constrained(w)


@pytest.mark.parametrize("name", PRIMITIVES)
@given(r=raw_vectors(6, extreme_raw))
def test_image_and_finiteness(name, r):
    c = PRIMITIVES[name]
    v = c(r)
    assert jnp.all(jnp.isfinite(v)), f"{name}: non-finite image"
    assert c.is_constrained(v), f"{name}: image outside the constrained set"
    g = jax.jacfwd(c)(r)
    assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite Jacobian"


@pytest.mark.parametrize("name", PRIMITIVES)
def test_is_constrained_rejects_outsiders(name):
    """``is_constrained`` is not vacuous: something outside the set is rejected (Free
    and
    Arcsinh have no outside apart from non-finite values)."""
    c = PRIMITIVES[name]
    bad = jnp.full(4, jnp.nan) if name in ("free", "arcsinh") else jnp.full(4, -1e3)
    assert not c.is_constrained(bad)


def test_widths_rejects_impossible_floor():
    with pytest.raises(ValueError):
        Widths(1.0, 0.3)(jnp.zeros(4))  # 4 * 0.3 >= 1


def test_shifted_primitives_validate_at_zero():
    with pytest.raises(ValueError):
        Positive(eps=1.0, at_zero=0.5)
    with pytest.raises(ValueError):
        Interval(0.0, 1.0, at_zero=1.0)
    with pytest.raises(ValueError):
        BoundedPositive(at_zero=100.0)


def test_primitive_construction_is_traceable():
    """The shift is computed under ``ensure_compile_time_eval``, so a primitive may be
    constructed inside a jitted/vmapped function (as ``constrain`` does)."""
    f = lambda r: Interval(-1.0, 8.0, at_zero=0.0)(r) + Positive(0.1)(r)  # noqa: E731
    r, tol = jnp.linspace(-1, 1, 5), TOL["jit_eager"]
    assert_close(jax.jit(jax.vmap(f))(r), jax.vmap(f)(r), rtol=tol, atol=tol)


# ------------------------------------------------------------ misc.squashed_exp ---
@pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
@given(x=raw_vectors(8))
def test_squashed_exp_inverse(a, x):
    y = squashed_exp(x, a)
    assert jnp.all((y > jnp.exp(-a)) & (y < jnp.exp(a)))
    tol = TOL["closed_form"]
    assert_close(inv_squashed_exp(y, a), x, rtol=tol, atol=tol, msg=f"a={a}")
