"""Shared test helpers: the named tolerances ``TOL``, solver configurations
``SOLVERS`` and isochron settings ``ISOCHRON_SETTINGS``, jitted checkers for the
bijection laws, and an optimizer step along a random direction."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from deep_isochron.analysis.ode import IsochronSettings
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
    # analysis (Phase E). ``Cycle`` point queries: a dense search plus one Newton step
    "cycle_query": 1e-6,
    # ``estimate_cycle`` on noise-free normal-form data: the period from interpolated
    # crossing times (relative), the curve and center (absolute), the phase against the
    # true one (peak-to-peak spread of the offset) — limited by the time grid (100 steps
    # per period) and the truncated density expansion, not by round-off
    "cycle_period": 1e-6,
    "cycle_curve": 2e-3,
    "cycle_phase": 2e-2,
    # ``analysis.ode`` (O1), the vector field integrated at rtol 1e-10 / atol 1e-12
    # (``SHOOTING_SOLVER_CONFIG``; always the solver, never a normal form's closed form,
    # which is the oracle): over one period the global error is ~1e-9, so the period and
    # the orbit against the exact values, and the level-set residual of a designed cycle
    # (Guckenheimer & Meloon's own accuracy measure), are asked to a decade above that;
    # the return map's point and time (event located by Newton at the solver's
    # tolerance) and the distance of a return-map derivative's eigenvalue from its exact
    # value likewise
    "limit_cycle": 1e-8,
    "return_map": 1e-8,
    "trivial_multiplier": 1e-8,
    "level_set": 1e-8,
    "floquet": 1e-8,  # O2: the exponent inherits O1's orbit error
    # O7: an AUTO isochron against the coworker's stored curve, as the distance from
    # points to the other curve's polyline; AUTO's own rerun reproduces the stored
    # curves to 3.6e-6, and the corrected bundle moves them by 8.1e-6
    # (docs/design/isochrons.md §7, decision D3)
    "isochron_curve": 2e-5,
    # O7: the adjoint bundle's branch point, where AUTO locates the multiplier 1 to
    # about 1e-11 (0.99999999998 on Hopf and Bautin)
    "branch_point": 1e-8,
    # O7: AUTO's orbit after stage 2 against O1's flow, and the Hopf route's orbit and
    # period against O1's; AUTO corrects to EPSU = 1e-10 and O1's solver runs at
    # rtol 1e-10, and the measured gaps are 3e-9 (FitzHugh–Nagumo) and 6e-11 (Hopf)
    "auto_orbit": 1e-8,
    # O7: the sine of the angle between the bundle's rotation Jw and O2's Floquet
    # vector, or between w and a closed-form gradient of the phase; measured 1e-11
    "bundle_direction": 1e-8,
    # O7: the spread of w·F along the cycle relative to its mean, constant in exact
    # arithmetic since d(w·F)/dt = 0 for the adjoint; measured 7e-10
    "bundle_scale": 1e-8,
    # O7: a normal form's closed-form phase along a traced isochron, against the
    # requested phase; the end direction is the stable direction, so the error is
    # O(eta²) plus the collocation error, measured 4e-10
    "isochron_phase": 1e-8,
    # O7: a traced point flowed five periods, against the base point; a point off its
    # isochron by delta ends |grad Θ| |F| delta away, measured 4e-6 at most on
    # FitzHugh–Nagumo and 1e-9 for most points
    "isochron_forward": 2e-5,
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


# AUTO isochron settings used by the tests, by name (decision G3). The step count of a
# half is set by how far its segment reaches, which grows with the number of periods,
# so the quick settings trace one period; a half that runs out of steps is flagged, not
# raised (D7). One FitzHugh–Nagumo phase then takes about half a minute on two cores.
# Every entry has one worker: pytest already runs one test per core, and a pool of
# its own per test oversubscribed the two cores of a session (8.4 min for the tier).
ISOCHRON_SETTINGS = {
    # every push: one period, a small box, a step cap
    "quick": IsochronSettings(
        n_periods=1, box_scale=2.0, max_steps=1500, max_workers=1
    ),
    # the full oracle suite (auto and slow): the coworker's settings
    "full": IsochronSettings(max_workers=1),
    # the normal forms against their closed forms: two periods, so that the halves
    # reach from near the phaseless set to the box (one period stays within 0.002 of
    # Winfree's cycle, whose multiplier is about 4e-4)
    "reach": IsochronSettings(
        n_periods=2, box_scale=2.0, max_steps=1500, max_workers=1
    ),
}


@eqx.filter_jit
def roundtrip(f, x):
    """``(f(x), f⁻¹(f(x)), f(f⁻¹(x)))`` for a batch ``x``."""
    y = jax.vmap(f)(x)
    return y, jax.vmap(f.inverse)(y), jax.vmap(f)(jax.vmap(f.inverse)(x))


@eqx.filter_jit
def jacobian_dets(f, x):
    """``det Df(x)`` for a batch ``x``."""
    return jax.vmap(lambda p: jnp.linalg.det(f.jacobian(p)))(x)


@eqx.filter_jit
def inverse_jacobian_product(f, x):
    """``Df⁻¹(f(x)) · Df(x)`` for a batch ``x``, the identity for a bijection."""
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
    """One Adam step on `module` along a random direction for every trainable leaf.

    A random direction is the stronger test for parameter validity: the constrained
    set must be preserved for any update an optimizer might take, not only for the
    gradient of one loss. ``test_true_gradient_step_decreases_loss`` covers the
    plumbing, that gradients flow through ``constrain``.

    **Arguments:**

    - `module`: The module to step.
    - `key`: The PRNG key for the direction.
    - `lr`: The Adam learning rate.

    **Returns:**

    The stepped module.
    """
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
