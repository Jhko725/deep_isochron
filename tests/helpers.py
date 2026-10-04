import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from deep_isochron.systems import SolverConfig


# Tolerances (float64 throughout). Every tolerance in the suite is an entry here, used
# by name, so that a tolerance is a documented decision rather than a per-test literal.
# Numbers that are *laws* (an asymptotic slope bound, a finite-difference step, a
# magnitude range for a strategy) are not tolerances and stay where the law is stated.
TOL = {
    # the same algebra evaluated under jit and eagerly: equal up to summation order
    "jit_eager": 1e-14,
    # exact algebra, round-off only: identity at zero raw parameters, a product
    # assembled from its factors, an oracle (scipy) evaluating the same spline
    "identity": 1e-12,
    # the CubicBSpline inverse at extreme parameters: Newton reaches float64 round trip
    "bspline_inverse": 1e-11,
    # two closed-form evaluations of one quantity: a primitive's inverse∘forward, a
    # parameter-swap inverse, f'(x_k) vs. the stored knot derivative, SVD vs. the
    # constrained singular values, expm(skew) orthogonality, round trip at knots
    "closed_form": 1e-9,
    # one analytic/spline map and its closed-form or 20-step Newton inverse
    "scalar_roundtrip": 1e-8,
    # a coupling/radial/linear layer: inverse plus a solve or a ratio
    "vector_roundtrip": 1e-7,
    # one-sided limits of f, f', f'' at the range endpoints (mismatch ~ h * f^(k+1))
    "join": 1e-7,
    # products of forward-mode Jacobians, 2x2; second derivatives of inverses
    "jacobian": 1e-6,
    # two numerical integrations of the same ODE at tight tolerance (rtol 1e-10), or an
    # integrated closed-form identity (phase, isostable) against the integrator's error
    "flow": 1e-6,
}


# Solver configurations used by the tests, by name, with the reason — the counterpart of
# ``TOL`` for ODE solves. ``Tsit5`` is diffrax's recommended non-stiff solver ("now
# reckoned on being slightly more efficient overall" than Dopri5; docs, "How to choose a
# solver"); the Kvaerno family is its recommendation for stiff problems.
SOLVERS = {
    # reference solutions for closed-form comparisons: error far below TOL["flow"]
    "tight": SolverConfig(solver=dfx.Tsit5(), rtol=1e-10, atol=1e-12, max_steps=16384),
    # a second integrator at the same tolerance: the answer is solver-independent
    "tight_stiff": SolverConfig(
        solver=dfx.Kvaerno5(), rtol=1e-10, atol=1e-12, max_steps=16384
    ),
    # long FitzHugh–Nagumo runs (period measurement over ~18 cycles): tight, many steps
    "tight_long": SolverConfig(
        solver=dfx.Tsit5(), rtol=1e-10, atol=1e-12, max_steps=1 << 17
    ),
    # Hodgkin–Huxley spiking: stiff gating kinetics, an implicit solver
    "stiff": SolverConfig(
        solver=dfx.Kvaerno5(), rtol=1e-8, atol=1e-10, max_steps=65536
    ),
    # data-generation tests: tight enough that only the metadata is in question
    "data": SolverConfig(solver=dfx.Tsit5(), rtol=1e-9, atol=1e-11),
}


@eqx.filter_jit
def roundtrip(f, x):
    """``(f(x), f⁻¹(f(x)), f(f⁻¹(x)))`` for a batch ``x``."""
    y = jax.vmap(f)(x)
    return y, jax.vmap(f.inverse)(y), jax.vmap(f)(jax.vmap(f.inverse)(x))


@eqx.filter_jit
def jacobian_dets(f, x):
    return jax.vmap(lambda p: jnp.linalg.det(f.jacobian(p)))(x)


@eqx.filter_jit
def inverse_jacobian_product(f, x):
    """``Df⁻¹(f(x)) · Df(x)`` — should be the identity."""
    J = jax.vmap(f.jacobian)(x)
    Jinv = jax.vmap(lambda p: jax.jacfwd(f.inverse)(p))(jax.vmap(f)(x))
    return Jinv @ J


def assert_close(actual, desired, *, rtol=None, atol=None, msg=""):
    """``np.testing.assert_allclose`` with ``TOL["closed_form"]`` as the default."""
    rtol = TOL["closed_form"] if rtol is None else rtol
    atol = TOL["closed_form"] if atol is None else atol
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, err_msg=msg
    )


def adam_step[M: eqx.Module](module: M, key: jax.Array, lr: float = 1.0) -> M:
    """One Adam step along a *random* direction for every trainable leaf.

    A random direction is the stronger test for parameter validity: the constrained set
    must be preserved for any update an optimizer might take, not just the gradient of
    one particular loss. ``test_true_gradient_step_decreases_loss`` covers the plumbing
    (that
    gradients flow through ``constrain``)."""
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    grads = jax.tree.unflatten(
        treedef,
        [
            jax.random.normal(k, l.shape, l.dtype)
            for k, l in zip(jax.random.split(key, len(leaves)), leaves)
        ],
    )
    opt = optax.adam(lr)
    updates, _ = opt.update(grads, opt.init(params), params)
    return eqx.combine(eqx.apply_updates(params, updates), static)
