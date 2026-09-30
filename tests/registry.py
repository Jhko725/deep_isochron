"""What gets tested.  Two tables + one exclusion list; ``test_registry.py`` enforces
that every concrete AbstractBijection subclass appears in exactly one of them.

SCALAR_TEMPLATES: name -> configured AbstractScalarBijection instance.  The instance
    is a *template*: tests draw raw vectors of length ``t.num_params`` and call
    ``t.from_unconstrained(raw)``.  Add config variants as extra lines.
VECTOR_BUILDERS:  name -> key -> AbstractBijection.  Identity-at-init classes are
    listed in IDENTITY_AT_INIT; the tests perturb weights for the other laws.
UNTESTED:         class -> reason.  Wrappers/abstract helpers exercised indirectly.
"""

import jax.numpy as jnp
from deep_isochron.model.invertible import (
    AffineCoupling,
    BiLipschitzLinear,
    CouplingFlow,
    CubicBSpline,
    CubicConjugation,
    CubicRational,
    InvertibleLinear,
    LinearSpline,
    MonotonicRQSpline,
    OffsetedBijection,
    ResidualCoupling,
    SequentialINN,
    SinhConjugation,
)
from deep_isochron.model.invertible.polar import (
    CircularMonotonicRQCoupling,
    PolarConditionalBijection,
    RadialBijection,
)


def _zeros(cls, n, **static):
    """A template: only its static config and ``num_params`` matter."""
    return cls(*[jnp.zeros(())] * n, **static)


def _identity(cls, n, **static):
    """A usable identity instance (``scale`` etc. constrained), for wrappers that
    evaluate the bijection directly rather than through ``from_unconstrained``."""
    return _zeros(cls, n, **static).from_unconstrained(jnp.zeros(n))


SCALAR_TEMPLATES = {
    "cubic_rational": _zeros(CubicRational, 3),
    "cubic_rational (eps_beta=0.5)": _zeros(CubicRational, 3),
    "sinh_conjugation": _zeros(SinhConjugation, 5),
    "cubic_conjugation": _zeros(CubicConjugation, 4),
    "linear_spline (K=5)": LinearSpline.identity(5),
    "rq_spline (K=4)": MonotonicRQSpline.identity(4),
    "rq_spline (K=9, range 4)": MonotonicRQSpline.identity(9, xy_range=(-4.0, 4.0)),
    "bspline (K=4)": CubicBSpline.identity(4),
    "bspline (K=10, range 4)": CubicBSpline.identity(10, xy_range=(-4.0, 4.0)),
}
"""``K`` is the number of bins (``num_bins``)."""

IDENTITY_TOL: dict[str, float] = {}


def _coupling(template, **kw):
    return lambda key: CouplingFlow(
        dim=2, bijection=template, mlp_width=16, key=key, **kw
    )


VECTOR_BUILDERS = {
    "coupling (cubic_conjugation x3)": _coupling(
        SequentialINN([_zeros(CubicConjugation, 4)] * 3)
    ),
    "coupling (rq_spline)": _coupling(SCALAR_TEMPLATES["rq_spline (K=9, range 4)"]),
    "coupling (bspline)": _coupling(SCALAR_TEMPLATES["bspline (K=10, range 4)"]),
    "coupling (sinh, flip)": _coupling(_zeros(SinhConjugation, 5), flip=True),
    "affine_coupling": lambda k: AffineCoupling(dim=2, width_hidden=16, key=k),
    "residual_coupling": lambda k: ResidualCoupling(dim=2, width_hidden=16, key=k),
    "invertible_linear": lambda k: InvertibleLinear(dim=2, key=k),
    "bilipschitz_linear": lambda k: BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=k),
    "radial (sinh)": lambda k: RadialBijection(
        _identity(SinhConjugation, 5), jnp.zeros(2), jnp.zeros(2)
    ),
    "circular_rq_coupling": lambda k: CircularMonotonicRQCoupling(num_knots=8, key=k),
    "polar_conditional": lambda k: PolarConditionalBijection(n_radial_blocks=2, key=k),
}
IDENTITY_AT_INIT = {
    "coupling (cubic_conjugation x3)",
    "coupling (rq_spline)",
    "coupling (bspline)",
    "coupling (sinh, flip)",
    "affine_coupling",
    "radial (sinh)",
}

UNTESTED = {
    OffsetedBijection: "wrapper; exercised through RadialBijection",
    SequentialINN: "container; exercised by test_sequential_inn_composes_inverse",
}
