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
# ci: 100 examples, derandomized. 300 took the `checks` job to its 45-minute limit on
# 2026-10-29 (38, 43 and two canceled runs) once the ODE-integrating laws grew.
settings.register_profile("ci", max_examples=100, derandomize=True, **_common)
settings.load_profile("default")


@pytest.fixture(autouse=True)
def _discard_compilations():
    """Clear JAX's compilation caches after every test.

    JAX keeps every compiled executable for the life of the process, and
    ``tests/test_analysis_ode.py`` alone compiles about 100 MB of fresh diffrax solves
    per test, so on 2026-10-30 both workers of the full tier reached 3.4 and 4.1 GB and
    the 7 GB memory cgroup killed them. The persistent compilation cache keeps the
    recompile of a later test cheap.
    """
    yield
    jax.clear_caches()


@pytest.fixture(scope="session")
def auto():
    """AUTO-07p's Python layer on the built tree, or a skip when no tree is found; the
    fixture of every test marked ``auto``."""
    from deep_isochron.analysis.ode.auto import _env

    try:
        return _env.load_auto()
    except _env.AutoNotInstalled as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def key():
    """The session's PRNG key, ``jax.random.key(0)``."""
    return jax.random.key(0)
