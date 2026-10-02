"""Laws specific to ``AbstractSpline`` bijections (piecewise maps on ``xy_range`` with
identity tails). The laws shared with every bijection are in ``test_bijections.py``.

  1. identity tails      f(x) == x outside the range, for every spline
  2. tail gradients      f' == 1 and finite parameter gradients for tail inputs
                         (regression: NaN from the unselected `where` branch)
  3. regularity of join  C^0 / C^1 / C^2 at the range endpoints, by spline class
  4. B-spline oracle     forward map == scipy.interpolate.BSpline
  5. B-spline inverse    Newton solve converges; implicit derivatives match 1/f'
  6. knots               round trip, interpolation and monotonicity at/across knots;
                         f'(x_k) == knot_derivs[k] for C^1+ splines (moved here from
                         test_bijections.py: these laws need `xs`/`ys`)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.model.invertible import (
    AbstractSpline,
    CubicBSpline,
    LinearSpline,
    MonotonicRQSpline,
)
from hypothesis import example, given, strategies as st
from scipy.interpolate import BSpline

from tests.helpers import assert_close, roundtrip, TOL
from tests.registry import SCALAR_TEMPLATES
from tests.strategies import (
    any_scalar_bijection,
    any_spline_template,
    point_batches,
    raw_vectors,
    scalar_bijections,
    spline_templates,
)


# Continuity class of the join between the interpolant and the identity tails, i.e.
# the number of derivatives that match (0 -> only the value). Declared by each class.
SPLINE_CLASSES = (LinearSpline, MonotonicRQSpline, CubicBSpline)
JOIN_ORDER = {
    cls: cls.__dataclass_fields__["smoothness"].default for cls in SPLINE_CLASSES
}
SPLINE_IDS = [c.__name__ for c in JOIN_ORDER]
KNOTTED = [n for n, t in SCALAR_TEMPLATES.items() if isinstance(t, AbstractSpline)]
KNOTTED_C1 = [n for n in KNOTTED if hasattr(SCALAR_TEMPLATES[n], "knot_derivs")]


def _derivatives(f, order):
    """``[f, f', ..., f^(order)]`` as scalar functions."""
    fs = [f]
    for _ in range(order):
        fs.append(jax.grad(fs[-1]))
    return fs


# ---------------------------------------------------------------- 1. tails --------
@pytest.mark.parametrize("cls", list(JOIN_ORDER), ids=SPLINE_IDS)
@given(data=st.data())
def test_identity_tails(cls, data):
    f = data.draw(spline_templates(cls).flatmap(scalar_bijections), label="spline")
    lo, hi = f.xy_range
    d = jnp.abs(data.draw(point_batches(1), label="offsets")[:, 0])
    x = jnp.concatenate([lo - d, hi + d])  # includes the endpoints (d = 0)
    y, x_rt, y_rt = roundtrip(f, x)
    assert_close(y, x, atol=0, rtol=0, msg="f != id on the tails")
    assert_close(x_rt, x, atol=0, rtol=0, msg="f⁻¹ != id on the tails")


# ------------------------------------------------------- 2. tail gradients --------
@pytest.mark.parametrize("cls", list(JOIN_ORDER), ids=SPLINE_IDS)
@given(data=st.data())
def test_tail_gradients_are_finite(cls, data):
    """Gradients w.r.t. x and w.r.t. the parameters stay finite for inputs outside
    the range, in both directions. The interpolant branch is evaluated at the
    clipped input for exactly this reason."""
    # Bin counts are fixed so the un-jitted gradients trace once per class.
    template = data.draw(
        spline_templates(cls, min_bins=6, max_bins=6), label="template"
    )
    raw = data.draw(raw_vectors(template.num_params), label="raw params")
    lo, hi = template.xy_range
    x = jnp.array([lo - 3.0, lo - 1e-3, lo, hi, hi + 1e-3, hi + 3.0])

    for name, op in (("f", lambda g, v: g(v)), ("f⁻¹", lambda g, v: g.inverse(v))):

        def loss(raw_):
            g = template.from_unconstrained(raw_)
            return jax.vmap(lambda v: op(g, v))(x).sum()

        grad_raw = jax.grad(loss)(raw)
        assert jnp.all(jnp.isfinite(grad_raw)), f"{name}: NaN in parameter gradient"

        g = template.from_unconstrained(raw)
        grad_x = jax.vmap(jax.grad(lambda v: op(g, v)))(x)
        assert jnp.all(jnp.isfinite(grad_x)), f"{name}: NaN in d/dx"
        tails = jnp.array([0, 1, 4, 5])
        assert_close(grad_x[tails], 1.0, msg=f"{name}: tail slope != 1")


# ----------------------------------------------------- 3. regularity of join ------
def _check_join(f, order):
    lo, hi = f.xy_range
    # One-sided limits: the mismatch is ~ h * f^(order+1), which is bounded by
    # ~1e4 for the drawn parameter ranges, hence TOL["join"] = 1e-7 at this step h.
    h = 1e-12 * (hi - lo)
    fs = _derivatives(f, order)
    for x0, x_in in ((lo, jnp.asarray(lo + h)), (hi, jnp.asarray(hi - h))):
        assert_close(fs[0](x_in), x0, atol=TOL["join"], msg=f"f discontinuous at {x0}")
        for order, d in enumerate(fs[1:], start=1):
            expected = 1.0 if order == 1 else 0.0
            assert_close(
                d(x_in),
                expected,
                atol=TOL["join"],
                msg=f"f^({order}) at {x0} inside != tail value {expected}",
            )


@pytest.mark.parametrize("cls", list(JOIN_ORDER), ids=SPLINE_IDS)
@given(data=st.data())
def test_join_regularity(cls, data):
    """At both range endpoints, the first ``JOIN_ORDER[cls]`` derivatives of the
    interpolant match those of the identity (1, 0, 0, ...), evaluated one-sided
    from inside the range. This is the property that motivates each spline."""
    f = data.draw(spline_templates(cls).flatmap(scalar_bijections), label="spline")
    _check_join(f, JOIN_ORDER[cls])


def test_bspline_identity_join_regularity():
    """Pinned example: the zero-parameter B-spline on an asymmetric range."""
    _check_join(CubicBSpline(6, xy_range=(-2.0, 3.0)), 2)


@given(data=st.data())
def test_bspline_is_c2_at_interior_knots(data):
    """f, f', f'' are continuous across interior knots; f''' generally jumps."""
    f = data.draw(
        spline_templates(CubicBSpline).flatmap(scalar_bijections), label="spline"
    )
    fs = _derivatives(f, 3)
    knots = f.xs[1:-1]
    h = 1e-8 * f.range_width
    for order, d in enumerate(fs[:3]):
        left, right = jax.vmap(d)(knots - h), jax.vmap(d)(knots + h)
        # A continuous f^(k) still differs across (k-h, k+h) by ~2h * |f^(k+1)|.
        slope = jnp.maximum(
            jnp.abs(jax.vmap(fs[order + 1])(knots - h)),
            jnp.abs(jax.vmap(fs[order + 1])(knots + h)),
        )
        gap, allowed = jnp.abs(right - left), 4 * h * slope + TOL["closed_form"]
        assert jnp.all(gap <= allowed), (
            f"f^({order}) jumps at a knot: {gap} > {allowed}"
        )


# -------------------------------------------------------- 4. B-spline oracle ------
@given(data=st.data())
@example(data=None)
def test_bspline_matches_scipy(data):
    """The forward map equals scipy's evaluation of the same non-uniform cubic
    B-spline (knots, coefficients), which pins the basis and index conventions."""
    if data is None:
        f = CubicBSpline(5, xy_range=(-2.0, 3.0))
    else:
        f = data.draw(
            spline_templates(CubicBSpline).flatmap(scalar_bijections), label="spline"
        )
    lo, hi = f.xy_range
    t, a = np.asarray(f.knots), np.asarray(f.coeffs)
    # scipy wants n + k + 1 knots for n coefficients (k = 3): pad one on each side;
    # the padded knots do not influence the spline on [lo, hi].
    t_pad = np.concatenate([[t[0] - 1.0], t, [t[-1] + 1.0]])
    ref = BSpline(t_pad, a, 3)
    x = np.linspace(lo, hi, 257)
    tol = TOL["identity"]
    assert_close(jax.vmap(f)(jnp.asarray(x)), ref(x), atol=tol, rtol=tol)


@given(data=st.data())
def test_bspline_coefficients_are_increasing(data):
    """Theorem 1 of Hong & Chun: strictly increasing coefficients (with positive knot
    spacing) give a strictly increasing spline. Both premises are enforced by the
    parametrisation."""
    f = data.draw(
        spline_templates(CubicBSpline).flatmap(scalar_bijections), label="spline"
    )
    assert jnp.all(jnp.diff(f.knots) > 0)
    assert jnp.all(jnp.diff(f.coeffs) > 0)


# ------------------------------------------------------- 5. B-spline inverse ------
@given(data=st.data())
def test_bspline_inverse_converges_for_extreme_params(data):
    """The Newton/bisection solve reaches float64 round-trip accuracy even for raw
    parameters far outside the typical regime (very uneven bins)."""
    template = data.draw(spline_templates(CubicBSpline), label="template")
    scale = data.draw(st.sampled_from([5.0, 20.0]), label="raw scale")
    raw = data.draw(raw_vectors(template.num_params), label="raw params")
    f = template.from_unconstrained(scale * raw)
    lo, hi = f.xy_range
    x = jnp.linspace(lo, hi, 129)
    _, x_rt, _ = roundtrip(f, x)
    assert_close(x_rt, x, atol=TOL["bspline_inverse"], rtol=TOL["bspline_inverse"])


@given(data=st.data())
def test_bspline_inverse_second_derivative(data):
    """The implicit-function JVP of the root solve is differentiable to second
    order: (f⁻¹)'' == -f'' / f'^3 (exact for any C^2 diffeomorphism)."""
    f = data.draw(
        spline_templates(CubicBSpline).flatmap(scalar_bijections), label="spline"
    )
    x = data.draw(point_batches(1), label="x")[:, 0]
    x = jnp.clip(x, *f.xy_range)
    y = jax.vmap(f)(x)
    f1, f2 = jax.vmap(jax.grad(f))(x), jax.vmap(jax.grad(jax.grad(f)))(x)
    g2 = jax.vmap(jax.grad(jax.grad(f.inverse)))(y)
    assert_close(g2, -f2 / f1**3, rtol=TOL["jacobian"], atol=TOL["jacobian"])


def test_bspline_template_static_only_is_enough():
    """A coupling layer keeps the template with its array leaves dropped;
    ``from_unconstrained`` must not depend on them."""
    template = CubicBSpline(6)
    static_only = eqx.partition(template, eqx.is_array)[1]
    raw = jnp.linspace(-1, 1, template.num_params)
    a, b = template.from_unconstrained(raw), static_only.from_unconstrained(raw)
    assert eqx.tree_equal(a, b)


# ---------------------------------------------------------------- 6. knots --------
# Measure-zero sets (knots, range endpoints) that property tests hit only by luck.
# Style: any_scalar_bijection() + @example pins, so regressions stay pinned.
def _identity_instance(name):
    return SCALAR_TEMPLATES[name].identity_like()


@given(f=any_scalar_bijection(KNOTTED))
@example(f=MonotonicRQSpline(5, xy_range=(-4.0, 4.0)))
@example(f=CubicBSpline(6))
def test_knots_round_trip(f):
    xk = f.xs  # includes both range endpoints
    _, x_rt, _ = roundtrip(f, xk)
    assert_close(x_rt, xk, atol=TOL["closed_form"], msg="round trip at knots")


@given(f=any_scalar_bijection(KNOTTED))
@example(f=LinearSpline(3))
def test_knots_interpolated_and_monotone(f):
    assert_close(jax.vmap(f)(f.xs), f.ys, atol=TOL["closed_form"], msg="f(x_k) != y_k")
    lo, hi = float(f.xs[0]), float(f.xs[-1])
    x = jnp.sort(jnp.concatenate([jnp.linspace(lo - 1, hi + 1, 2001), f.xs]))
    x = x[jnp.concatenate([jnp.array([True]), jnp.diff(x) > 1e-9])]  # drop near-dups
    assert jnp.all(jnp.diff(jax.vmap(f)(x)) > 0), "not monotone across knots"


@given(f=any_scalar_bijection(KNOTTED_C1))
def test_knot_derivatives(f):
    """f'(x_k) == knot_derivs[k] at every knot, boundaries included (identity tails
    join C¹)."""
    tol = TOL["closed_form"]
    assert_close(
        jax.vmap(jax.grad(f))(f.xs), f.knot_derivs, rtol=tol, atol=tol, msg="f'(x_k)"
    )


@given(f=any_spline_template().flatmap(scalar_bijections), x=point_batches(1))
def test_spline_any_bin_count_round_trip(f, x):
    """Round trip for drawn spline class, bin count and range (not just the
    registry's configs)."""
    x = x[:, 0]
    _, x_rt, _ = roundtrip(f, x)
    assert_close(x_rt, x, rtol=TOL["scalar_roundtrip"], atol=TOL["scalar_roundtrip"])
