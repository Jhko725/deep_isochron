import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


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
