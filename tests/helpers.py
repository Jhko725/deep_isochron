import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


# Tolerances (float64 throughout). Each law uses the entry named after it so that a
# tolerance is a documented decision rather than a per-test literal.
TOL = {
    # exact algebra at zero raw parameters: round-off only
    "identity": 1e-12,
    # one analytic/spline map and its closed-form or 20-step Newton inverse
    "scalar_roundtrip": 1e-8,
    # a coupling/radial/linear layer: inverse plus a solve or a ratio
    "vector_roundtrip": 1e-7,
    # products of forward-mode Jacobians, 2x2
    "jacobian": 1e-6,
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


def assert_close(actual, desired, *, rtol=1e-9, atol=1e-9, msg=""):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, err_msg=msg
    )


def adam_step[M: eqx.Module](module: M, key: jax.Array, lr: float = 1.0) -> M:
    """One Adam step along a *random* direction for every trainable leaf.

    A random direction is the stronger test for parameter validity: the constrained set
    must be preserved for any update an optimiser might take, not just the gradient of
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
