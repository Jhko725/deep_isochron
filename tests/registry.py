"""What gets tested.  Two tables + one exclusion list; ``test_registry.py`` enforces
that every concrete AbstractBijection subclass appears in exactly one of them.

SCALAR_TEMPLATES: name -> AbstractScalarBijection instance built from static config
    only (``cls(config)``, i.e. the identity).  Tests draw raw vectors of length
    ``t.num_params`` and call ``t.from_unconstrained(raw)``.  Add config variants as
    extra lines; they must differ in static configuration, not just in name.
VECTOR_BUILDERS:  name -> key -> AbstractBijection.  Every entry is the identity map
    at init (a universal law since A2 of ``docs/roadmap.md``); the other laws run on
    perturbed weights.
UNTESTED:         class -> reason.  Wrappers/containers exercised indirectly, or classes
    parked for a later step of the current branch (the reason names the step; the
    registry test fails as soon as the class is covered again, so the entry cannot be
    forgotten).
"""

import jax.numpy as jnp
from deep_isochron.model.invertible import (
    Affine,
    AffineCoupling,
    BiLipschitzLinear,
    CouplingFlow,
    CubicBSpline,
    CubicConjugation,
    CubicRational,
    LinearSpline,
    MonotonicRQSpline,
    OffsetedBijection,
    ResidualCoupling,
    ScalarChain,
    SequentialINN,
    Shift,
    SinhConjugation,
)
from deep_isochron.model.invertible.polar import (
    CircularMonotonicRQCoupling,
    PolarCouplingFlow,
    RadialBijection,
)


SCALAR_TEMPLATES = {
    "shift": Shift(),
    "affine": Affine(),
    "affine (clamp=0.5)": Affine(clamp=0.5),
    "chain (affine, shift)": ScalarChain([Affine(), Shift()]),
    "cubic_rational": CubicRational(),
    "cubic_rational (eps_beta=0.5)": CubicRational(eps_beta=0.5),
    "sinh_conjugation": SinhConjugation(),
    "sinh_conjugation (eps_scale=0.3)": SinhConjugation(eps_scale=0.3),
    "cubic_conjugation": CubicConjugation(),
    "cubic_conjugation (eps_a=0.1)": CubicConjugation(eps_a=0.1),
    "chain (rational, sinh, cubic)": ScalarChain(
        [CubicRational(), SinhConjugation(), CubicConjugation()]
    ),
    "linear_spline (K=5)": LinearSpline(5),
    "rq_spline (K=4)": MonotonicRQSpline(4),
    "rq_spline (K=9, range 4)": MonotonicRQSpline(9, xy_range=(-4.0, 4.0)),
    "bspline (K=4)": CubicBSpline(4),
    "bspline (K=10, range 4)": CubicBSpline(10, xy_range=(-4.0, 4.0)),
    "chain (rq_spline, cubic)": ScalarChain([MonotonicRQSpline(4), CubicConjugation()]),
    "offset (sinh)": OffsetedBijection(SinhConjugation()),
    "offset (chain sinh x2)": OffsetedBijection(ScalarChain([SinhConjugation()] * 2)),
}
"""``K`` is the number of bins (``num_bins``)."""


def _coupling(template, **kw):
    return lambda key: CouplingFlow(
        dim=2, bijection=template, mlp_width=16, key=key, **kw
    )


VECTOR_BUILDERS = {
    "coupling (cubic_conjugation x3)": _coupling(ScalarChain([CubicConjugation()] * 3)),
    "coupling (sinh, flip)": _coupling(SinhConjugation(), flip=True),
    "coupling (rq_spline)": _coupling(SCALAR_TEMPLATES["rq_spline (K=9, range 4)"]),
    "coupling (bspline)": _coupling(SCALAR_TEMPLATES["bspline (K=10, range 4)"]),
    "coupling (rational, split_idx=1 of 3)": lambda key: CouplingFlow(
        dim=3, bijection=CubicRational(), split_idx=1, mlp_width=16, key=key
    ),
    "affine_coupling": lambda k: AffineCoupling(dim=2, mlp_width=16, key=k),
    "affine_coupling (flip, 1 of 3)": lambda k: AffineCoupling(
        dim=3, split_idx=1, flip=True, mlp_width=16, key=k
    ),
    "residual_coupling": lambda k: ResidualCoupling(dim=2, mlp_width=16, key=k),
    "bilipschitz_linear": lambda k: BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=k),
    "bilipschitz_linear (dim 3, L=5)": lambda k: BiLipschitzLinear(
        dim=3, max_lipschitz=5.0, key=k
    ),
    "radial (sinh)": lambda k: RadialBijection(
        SinhConjugation(), jnp.zeros(2), jnp.zeros(2)
    ),
    "circular_rq (K=8)": lambda k: CircularMonotonicRQCoupling(num_bins=8),
    "polar_coupling (sinh x2, order 3)": lambda k: PolarCouplingFlow(
        ScalarChain([SinhConjugation()] * 2), fourier_order=3
    ),
    "polar_coupling (rq_spline, order 2)": lambda k: PolarCouplingFlow(
        MonotonicRQSpline(5, xy_range=(-3.0, 3.0)), fourier_order=2
    ),
}

UNTESTED = {
    SequentialINN: "container; exercised by test_sequential_inn_composes_inverse",
}
