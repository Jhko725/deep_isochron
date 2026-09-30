"""Hypothesis strategies for bijections.

Conventions (from the Hypothesis docs and TFP's ``hypothesis_testlib``):

* ``.map`` for a pure transform of a single draw (raw vector -> bijection).
* ``@st.composite`` only when a later draw depends on an earlier one (a template,
  then a raw vector *of its length*; a knot count, then arrays *of that size*).
  Every ``draw`` inside a composite carries a ``label`` so failing examples read
  as "Draw 2 (raw params): ..." rather than an opaque pytree repr.
* ``st.builds`` for "call a constructor with drawn arguments".
* ``.filter`` on strategies, never ``assume``/early-return in test bodies.
* Array *shapes* are fixed inside a strategy (BATCH points, or a bounded knot
  count) so jitted checkers compile a bounded number of times.

Strategies are plain functions, as the ``@st.composite`` docstring recommends;
``register_type_strategy`` associations are made at the bottom so
``st.from_type(SomeBijection)`` works too.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from deep_isochron.model.invertible import (
    CubicConjugation,
    CubicRational,
    MonotonicRQSpline,
    SinhConjugation,
)
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tests import registry


BATCH = 16  # fixed batch -> one compile per test function

# ------------------------------------------------------------- scalar elements ----
# "Typical regime" bounds; extreme regimes get their own explicitly named tests.
raw_params = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
domain_points = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(0, 2**31 - 1)


# --------------------------------------------------------------- pure transforms --
def raw_vectors(n: int) -> st.SearchStrategy[jax.Array]:
    """Unconstrained parameter vector of length ``n``, as a conditioner would emit it
    ."""
    return hnp.arrays(np.float64, (n,), elements=raw_params).map(jnp.asarray)


def point_batches(dim: int, n: int = BATCH) -> st.SearchStrategy[jax.Array]:
    """Fixed-shape ``(n, dim)`` batch in the working domain.  Hypothesis biases float
    generation toward 'interesting' values (0, the bounds, powers of two), which is
    how it finds knot/boundary bugs without being told about them."""
    return hnp.arrays(np.float64, (n, dim), elements=domain_points).map(jnp.asarray)


# ------------------------------------------------------------- dependent draws ----
@st.composite
def scalar_bijections(draw, template):
    """A concrete scalar bijection: ``template`` with drawn raw parameters."""
    raw = draw(raw_vectors(template.num_params), label="raw params")
    return template.with_unconstrained(raw)


@st.composite
def any_scalar_bijection(draw, names=None):
    """Draw a template by name from the registry, then its parameters.  Use this
    (instead of ``st.data`` + parametrize) in tests that need ``@example`` pins."""
    names = list(registry.SCALAR_TEMPLATES if names is None else names)
    name = draw(st.sampled_from(names), label="template")
    return draw(scalar_bijections(registry.SCALAR_TEMPLATES[name]), label="bijection")


@st.composite
def rq_spline_templates(draw, min_knots=3, max_knots=12):
    """RQ-spline templates with a *drawn* knot count and range — sizes of the
    parameter arrays depend on ``K``, which is exactly what @composite is for.
    ``max_knots`` is bounded so jitted checkers compile at most ~10 times."""
    K = draw(st.integers(min_knots, max_knots), label="num_knots")
    lo = draw(st.floats(-5.0, -0.5, allow_nan=False), label="range lo")
    hi = draw(st.floats(0.5, 5.0, allow_nan=False), label="range hi")
    return MonotonicRQSpline(
        jnp.zeros(K - 1), jnp.zeros(K - 1), jnp.zeros(K - 2), xy_range=(lo, hi)
    )


def perturb(module, key, scale: float = 0.5):
    """Gaussian noise on every floating leaf: identity-at-init modules become
    non-trivial maps so the round-trip / orientation laws test the MLP path."""
    arrays, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(arrays)
    keys = jax.random.split(key, len(leaves))
    leaves = [
        l + scale * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, keys)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


@st.composite
def vector_bijections(draw, name, perturbed=True):
    """A vector bijection from the registry, built from a drawn seed and (optionally)
    with drawn weight perturbations."""
    seed = draw(seeds, label="init seed")
    f = registry.VECTOR_BUILDERS[name](jax.random.key(seed))
    if perturbed:
        pseed = draw(seeds, label="perturbation seed")
        f = perturb(f, jax.random.key(pseed))
    return f


# ---------------------------------------------------- constructors via builds ----
_zero = st.just(jnp.zeros(()))

analytic_templates = st.one_of(
    st.builds(CubicRational, _zero, _zero, _zero, eps_beta=st.sampled_from([0.1, 0.5])),
    st.builds(
        SinhConjugation,
        _zero,
        _zero,
        _zero,
        _zero,
        _zero,
        eps_scale=st.sampled_from([0.1, 0.3]),
    ),
    st.builds(
        CubicConjugation,
        _zero,
        _zero,
        _zero,
        _zero,
        eps_a=st.sampled_from([1e-2, 1e-1]),
    ),
)
"""Analytic templates with drawn static config, for tests of config-independence."""


# ----------------------------------------------------------- type registration ----
for _cls, _n in ((CubicRational, 3), (SinhConjugation, 5), (CubicConjugation, 4)):
    st.register_type_strategy(
        _cls,
        analytic_templates.filter(lambda t, c=_cls: isinstance(t, c)).flatmap(
            scalar_bijections
        ),
    )
st.register_type_strategy(
    MonotonicRQSpline, rq_spline_templates().flatmap(scalar_bijections)
)
