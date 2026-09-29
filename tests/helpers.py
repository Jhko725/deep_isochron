import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


@eqx.filter_jit
def roundtrip(f, x):
    """``(f(x), f⁻¹(f(x)), f(f⁻¹(x)))`` for a batch ``x`` — jitted once per module
    structure."""
    y = jax.vmap(f)(x)
    return y, jax.vmap(f.inverse)(y), jax.vmap(f)(jax.vmap(f.inverse)(x))


@eqx.filter_jit
def jacobian_dets(f, x):
    return jax.vmap(lambda p: jnp.linalg.det(jax.jacfwd(f)(p)))(x)


def perturb(module, key, scale: float = 0.5):
    """Add Gaussian noise to every floating-point leaf, so identity-at-init modules
    become non-trivial maps and the round-trip law is tested for *arbitrary* weights."""
    arrays, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(arrays)
    keys = jax.random.split(key, len(leaves))
    leaves = [
        l + scale * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, keys)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def assert_close(actual, desired, *, rtol=1e-9, atol=1e-9, msg=""):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, err_msg=msg
    )
