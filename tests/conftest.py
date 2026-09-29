"""Shared fixtures, Hypothesis profiles and strategies for deep_isochron tests.

Conventions
-----------
* float64 + CPU, always.  Tests never touch the GPU.
* Hypothesis profiles: ``dev`` (pre-commit, fast), ``default`` (local ``pytest``),
  ``ci`` (thorough).  Select with ``--hypothesis-profile=<name>``.
* Every property test that runs a JAX function should go through a *jitted* helper
  with a *fixed batch shape* (see ``roundtrip`` below).  Otherwise each Hypothesis
  example re-dispatches op-by-op and a 50-example test takes ~40 s instead of ~9 s.
"""

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.extra import numpy as hnp


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# Run runtime shape checks during testing
jaxtyping.install_import_hook("deep_isochron", "beartype.beartype")
# ----------------------------------------------------------------- profiles --------
# deadline=None and too_slow suppressed: the first example of every test pays the
# XLA compile, which Hypothesis would otherwise flag as a flaky/slow example.
settings.register_profile(
    "default",
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev", parent=settings.get_profile("default"), max_examples=10
)
settings.register_profile(
    "ci", parent=settings.get_profile("default"), max_examples=300, derandomize=True
)
settings.load_profile("default")

BATCH = 16  # fixed so that a jitted helper compiles once per test function


# --------------------------------------------------------------- strategies --------
# "Typical regime" bounds. Extreme parameters (|raw| >> 1, |x| >> domain) are a
# separate, explicitly named test, not something property tests stumble into.
raw_params = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
domain_points = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(0, 2**31 - 1)


def raw_param_arrays(n: int):
    """``n`` unconstrained scalar parameters, as the conditioner MLP would emit them."""
    return st.lists(raw_params, min_size=n, max_size=n).map(jnp.asarray)


def point_batches(dim: int, n: int = BATCH):
    """Fixed-shape ``(n, dim)`` float64 batch of points in the working domain.

    Hypothesis biases float generation toward 'interesting' values (0, bounds, powers
    of two...), which is how it finds knot/boundary bugs without being told about them.
    """
    return hnp.arrays(np.float64, (n, dim), elements=domain_points).map(jnp.asarray)


@pytest.fixture(scope="session")
def key():
    return jax.random.key(0)
