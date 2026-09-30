"""Test configuration: float64/CPU, Hypothesis profiles, jitted checkers.

Strategies live in ``tests/strategies.py``; the tables of what gets tested live in
``tests/registry.py``. Helper functions are in ``tests/helpers.py``. This file only
holds test configurations.

Hypothesis profiles (select with ``--hypothesis-profile=<name>``):
  dev      10 examples   pre-commit / quick local runs
  default  50 examples   plain ``pytest``
  ci       300 examples  derandomized, for CI and nightly

Rationale for the settings (mirrors tensorflow_probability's ``tfp_hp_settings``):
  * deadline=None + too_slow suppressed: the first example of every test pays the
    XLA compile, which Hypothesis would otherwise report as a flaky slow example.
  * print_blob=True: always print the ``@reproduce_failure`` blob.
  * ci is derandomized so CI failures are reproducible; local runs stay random so
    they keep exploring.
"""

import jax
import jaxtyping
import pytest
from hypothesis import HealthCheck, settings


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# Run runtime shape checks during testing
jaxtyping.install_import_hook("deep_isochron", "beartype.beartype")
# ----------------------------------------------------------------- profiles --------
# deadline=None and too_slow suppressed: the first example of every test pays the
# XLA compile, which Hypothesis would otherwise flag as a flaky/slow example.

_common = dict(
    deadline=None, suppress_health_check=[HealthCheck.too_slow], print_blob=True
)
settings.register_profile("default", max_examples=50, **_common)
settings.register_profile("dev", max_examples=10, **_common)
settings.register_profile("ci", max_examples=300, derandomize=True, **_common)
settings.load_profile("default")


@pytest.fixture(scope="session")
def key():
    return jax.random.key(0)
