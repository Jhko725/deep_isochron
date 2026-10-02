"""Laws specific to the analytic bijections (``invertible/analytic.py``), ported from
bijx's ``test_bijections_analytic.py`` where applicable. Shared laws are in
``test_bijections.py``.

  1. inverse symmetry   SinhConjugation⁻¹ is SinhConjugation with (β, μ, ν) -> (-β, -ν,
  -μ) 2. asymptotics        CubicRational -> identity as |x| -> ∞; SinhConjugation is
                        asymptotically linear
  3. extreme regime     f, f⁻¹ and grad..grad³ stay finite for |raw| <= 30, |x| <= 1e3
                        (the overflow-safe branches of the sinh helpers)
  4. from_constrained   round trip through the constraint inverses
"""

import jax
import jax.numpy as jnp
import pytest
from deep_isochron.model.invertible import (
    CubicConjugation,
    CubicRational,
    SinhConjugation,
)
from hypothesis import given, strategies as st

from tests.helpers import assert_close, roundtrip
from tests.strategies import EXTREME_RAW_BOUND, magnitudes, raw_vectors


ANALYTIC = {
    "cubic_rational": CubicRational(),
    "sinh_conjugation": SinhConjugation(),
    "cubic_conjugation": CubicConjugation(),
}
# Magnitudes in [1e-6, 30] / [1e-12, 1e3] or exactly 0: the sinh helpers' second
# derivative underflows to NaN when 0 < |x - loc| < ~1e-20 (pinned below as a strict
# xfail), a float corner rather than the overflow regime these tests are about.
extreme_raw = magnitudes(1e-6, EXTREME_RAW_BOUND)
extreme_x = magnitudes(1e-12, 1e3)


# ---------------------------------------------------------- 1. inverse symmetry ---
@given(raw=raw_vectors(5), x=raw_vectors(8))
def test_sinh_inverse_is_parameter_swap(raw, x):
    f = SinhConjugation().from_unconstrained(raw)
    p = f.params
    g = SinhConjugation.from_constrained(p.loc, p.scale, -p.beta, -p.nu, -p.mu)
    assert_close(jax.vmap(f.inverse)(x), jax.vmap(g)(x), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------- 2. asymptotics ---
@given(raw=raw_vectors(3))
def test_cubic_rational_asymptotic_identity(raw):
    """f(x) - x = alpha x_ / (1 + beta x_^2), so |f(x) - x| <= |alpha| / (beta |x_|)."""
    f = CubicRational().from_unconstrained(raw)
    p = f.params
    x = jnp.array([-1e4, -1e3, 1e3, 1e4])
    bound = jnp.abs(p.alpha) / (p.beta * jnp.abs(x - p.loc))
    assert jnp.all(jnp.abs(jax.vmap(f)(x) - x) <= bound * (1 + 1e-9) + 1e-12)
    assert jnp.all(bound < 0.1)  # and the bound itself is small at |x| >= 1e3


@given(raw=raw_vectors(5))
def test_sinh_asymptotically_linear(raw):
    f = SinhConjugation().from_unconstrained(raw)
    x = jnp.array([50.0, 100.0, 200.0])
    slopes = jnp.diff(jax.vmap(f)(x)) / jnp.diff(x)
    assert_close(slopes, jnp.ones(2), rtol=1e-3, atol=1e-3)


# ----------------------------------------------------------- 3. extreme regime ----
@pytest.mark.parametrize("name", ANALYTIC)
@given(data=st.data())
def test_extreme_regime_is_finite(name, data):
    t = ANALYTIC[name]
    f = t.from_unconstrained(
        data.draw(raw_vectors(t.num_params, extreme_raw), label="raw")
    )
    x = data.draw(raw_vectors(8, extreme_x), label="x")
    y, x_rt, _ = roundtrip(f, x)
    assert jnp.all(jnp.isfinite(y)), f"{name}: forward"
    assert jnp.all(jnp.isfinite(x_rt)), f"{name}: inverse"
    d1, d2, d3 = jax.grad(f), jax.grad(jax.grad(f)), jax.grad(jax.grad(jax.grad(f)))
    for k, d in enumerate((d1, d2, d3), start=1):
        assert jnp.all(jnp.isfinite(jax.vmap(d)(x))), f"{name}: grad^{k} not finite"


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: sinh_conj_nonlinearity's 2nd derivative underflows to NaN "
    "for 0 < |x - loc| < ~1e-20 (squared-argument underflow in the double-where "
    "path); exactly 0 is fine",
)
def test_sinh_second_derivative_near_zero_is_finite():
    f = SinhConjugation()
    d2 = jax.grad(jax.grad(f))
    assert jnp.isfinite(d2(jnp.asarray(1e-158)))


# ---------------------------------------------------------- 4. from_constrained ---
@given(raw=raw_vectors(3))
def test_cubic_rational_from_constrained_round_trip(raw):
    f = CubicRational(eps_beta=0.5).from_unconstrained(raw)
    p = f.params
    g = CubicRational.from_constrained(p.alpha, p.beta, p.loc, eps_beta=0.5)
    assert_close(g.raw, f.raw, rtol=1e-9, atol=1e-9)


@given(raw=raw_vectors(4))
def test_cubic_conjugation_from_constrained_round_trip(raw):
    f = CubicConjugation().from_unconstrained(raw)
    p = f.params
    g = CubicConjugation.from_constrained(p.loc, p.beta, p.a, p.b)
    assert_close(g.raw, f.raw, rtol=1e-8, atol=1e-8)
