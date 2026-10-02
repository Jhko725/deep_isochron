r"""Constraint primitives: maps from unconstrained reals onto a constrained set.

A scalar bijection stores a single unconstrained vector ``raw`` as its trainable leaf
and computes its constrained parameters on read through ``constrain(raw)``, built from
the primitives in this module (``docs/decisions/0001-unconstrained-leaves.md``,
``0005-constraint-primitives.md``). Each primitive is a bijection ``R^n -> S`` in its
own right, with ``__call__`` (raw -> constrained), ``inverse`` (constrained -> raw) and
``is_constrained`` (membership in ``S``), so that bijections can also be built from
constrained values (``from_constrained``) and tests can check the image generically.

Identity-at-zero convention (``docs/decisions/0002-identity-at-zero.md``): every
primitive with an ``at_zero`` argument maps ``raw = 0`` to ``at_zero``. This puts the
shift into the *map* rather than into the initialisation, which is what makes a
zero-initialised conditioner (``CouplingFlow``) the identity. Shifted primitives
implement the unshifted map pair ``_forward``/``_inverse`` once; the shift is
``_inverse(at_zero)``, evaluated under ``jax.ensure_compile_time_eval`` so that it is a
concrete float even when the primitive is constructed inside a ``jit``/``vmap`` trace.

The stateless primitives are also available as module-level singletons ``free`` and
``arcsinh`` (``arcsinh(mu)`` reads better than ``Arcsinh()(mu)``).
"""

import abc

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ...misc import inv_softplus, inv_squashed_exp, squashed_exp


class Constraint(abc.ABC):
    """Map from unconstrained reals onto a constrained set ``S``, with inverse.

    Elementwise primitives act on arrays of any shape (``" *n"``: a 0-d parameter such
    as ``loc`` or a block of a raw vector alike); ``Widths`` is intrinsically a vector
    map (``" n"``) and narrows the annotation. Instances are plain, hashable Python
    objects (not modules); they hold only static configuration and are created inside
    ``constrain`` (ADR-0005).
    """

    @abc.abstractmethod
    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]: ...

    @abc.abstractmethod
    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]: ...

    @abc.abstractmethod
    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether every element of ``value`` lies in ``S``."""


class _Shifted(Constraint):
    """Elementwise primitive ``value = _forward(raw + shift)`` with ``shift`` chosen so
    that ``raw = 0`` maps to ``at_zero``. Subclasses implement the *unshifted* pair
    ``_forward``/``_inverse`` and ``is_constrained``;
    ``__call__``/``inverse``/``_shift``
    derive from them."""

    at_zero: float

    def _init_shift(self, at_zero: float) -> None:
        self.at_zero = at_zero
        # Concrete even inside a trace: constructing a primitive under jit/vmap must not
        # stage this computation (a traced float() would raise ConcretizationTypeError).
        # Requires _inverse to be plain jnp (no jit-decorated helpers; see Interval).
        with jax.ensure_compile_time_eval():
            self._shift = float(self._inverse(jnp.asarray(at_zero)))

    @abc.abstractmethod
    def _forward(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]: ...

    @abc.abstractmethod
    def _inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]: ...

    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        return self._forward(raw + self._shift)

    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        return self._inverse(value) - self._shift


class Free(Constraint):
    """The identity: no constraint. ``raw = 0`` maps to ``0``."""

    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        return raw

    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        return value

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        return jnp.all(jnp.isfinite(value))


class Arcsinh(Constraint):
    """``R -> R`` via ``arcsinh``: a soft, symmetric compression of large raw values
    (used for the log-scale parameters of ``SinhConjugation``). ``raw = 0`` maps to
    ``0``."""

    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        return jnp.arcsinh(raw)

    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        return jnp.sinh(value)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        return jnp.all(jnp.isfinite(value))


class Positive(_Shifted):
    r"""``(eps, inf)`` via a shifted softplus: ``eps + softplus(raw + c)`` with ``c``
    chosen so that ``raw = 0`` maps to ``at_zero``.

    **Arguments:**

    - ``eps``: lower bound (exclusive). Default ``0``.
    - ``at_zero``: image of ``raw = 0``; must exceed ``eps``. Default ``1``.
    """

    def __init__(self, eps: float = 0.0, at_zero: float = 1.0):
        if not at_zero > eps:
            raise ValueError("at_zero must be greater than eps.")
        self.eps = eps
        self._init_shift(at_zero)

    def _forward(self, raw):
        return self.eps + jax.nn.softplus(raw)

    def _inverse(self, value):
        return inv_softplus(value - self.eps)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        return jnp.all(value > self.eps)

    def __repr__(self):
        return f"Positive(eps={self.eps}, at_zero={self.at_zero})"


class BoundedPositive(_Shifted):
    r"""``(eps + exp(-a), eps + exp(a))`` via a shifted ``squashed_exp``:
    ``eps + exp(a * tanh((raw + c) / a))`` with ``c`` chosen so that ``raw = 0`` maps to
    ``at_zero``. Unlike ``Positive`` the image is bounded above, so a large conditioner
    output cannot drive the value to infinity (used for the cubic generator's ``a, b``).

    **Arguments:**

    - ``eps``: lower offset. Default ``0``.
    - ``at_zero``: image of ``raw = 0``; ``at_zero - eps`` must lie in
      ``(exp(-a), exp(a))``. Default ``1``.
    - ``a``: squashing scale. Default ``2``.
    """

    def __init__(self, eps: float = 0.0, at_zero: float = 1.0, a: float = 2.0):
        import math

        if not math.exp(-a) < at_zero - eps < math.exp(a):
            raise ValueError("at_zero - eps must lie in (exp(-a), exp(a)).")
        self.eps = eps
        self.a = a
        self._init_shift(at_zero)

    def _forward(self, raw):
        return self.eps + squashed_exp(raw, self.a)

    def _inverse(self, value):
        return inv_squashed_exp(value - self.eps, self.a)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        lo, hi = self.eps + jnp.exp(-self.a), self.eps + jnp.exp(self.a)
        return jnp.all((value > lo) & (value < hi))

    def __repr__(self):
        return f"BoundedPositive(eps={self.eps}, at_zero={self.at_zero}, a={self.a})"


class Interval(_Shifted):
    r"""``(lo, hi)`` via a shifted sigmoid: ``lo + (hi - lo) * sigmoid(raw + c)`` with
    ``c`` chosen so that ``raw = 0`` maps to ``at_zero``.

    **Arguments:**

    - ``lo``, ``hi``: open interval bounds, ``lo < hi``.
    - ``at_zero``: image of ``raw = 0``; must lie strictly inside the interval. Default
      is the midpoint.
    """

    def __init__(self, lo: float, hi: float, at_zero: float | None = None):
        if not lo < hi:
            raise ValueError("lo must be less than hi.")
        at_zero = 0.5 * (lo + hi) if at_zero is None else at_zero
        if not lo < at_zero < hi:
            raise ValueError("at_zero must lie strictly inside (lo, hi).")
        self.lo = lo
        self.hi = hi
        self._init_shift(at_zero)

    def _forward(self, raw):
        return self.lo + (self.hi - self.lo) * jax.nn.sigmoid(raw)

    def _inverse(self, value):
        # Plain jnp rather than jax.scipy.special.logit: that function is jit-decorated,
        # and a jitted call inside a vmap trace returns a tracer even for a concrete
        # argument, which would defeat ensure_compile_time_eval in _init_shift.
        q = (value - self.lo) / (self.hi - self.lo)
        return jnp.log(q) - jnp.log1p(-q)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        return jnp.all((value > self.lo) & (value < self.hi))

    def __repr__(self):
        return f"Interval(lo={self.lo}, hi={self.hi}, at_zero={self.at_zero})"


class Widths(Constraint):
    r"""``n`` positive widths summing to ``total``, each at least ``min_rel * total``:
    ``(softmax(raw) * (1 - n * min_rel) + min_rel) * total`` (the floored softmax of
    Durkan et al. 2019). ``raw = 0`` maps to equal widths.

    The softmax is shift-invariant, so the inverse is defined up to a constant; the
    gauge chosen here is **mean-zero raw** (``raw = log(rel - min_rel)`` centred), which
    is also the gauge in which equal widths map back to ``raw = 0``.

    **Arguments:**

    - ``total``: sum of the widths (a float or a 0-d array; may be traced).
    - ``min_rel``: minimum relative width. Requires ``n * min_rel < 1``; checked per
    call
      because ``n`` is the length of ``raw``.
    """

    def __init__(self, total, min_rel: float = 1e-3):
        if min_rel < 0:
            raise ValueError("min_rel must be non-negative.")
        self.total = total
        self.min_rel = min_rel

    def _check_n(self, n: int) -> None:
        if n * self.min_rel >= 1:
            raise ValueError(
                f"n * min_rel = {n * self.min_rel} must be < 1 for n = {n} widths."
            )

    def __call__(self, raw: Float[Array, " n"]) -> Float[Array, " n"]:
        n = raw.shape[-1]
        self._check_n(n)
        rel = jax.nn.softmax(raw, axis=-1) * (1 - n * self.min_rel) + self.min_rel
        return rel * self.total

    def inverse(self, value: Float[Array, " n"]) -> Float[Array, " n"]:
        n = value.shape[-1]
        self._check_n(n)
        rel = value / self.total
        raw = jnp.log((rel - self.min_rel) / (1 - n * self.min_rel))
        return raw - jnp.mean(raw, axis=-1, keepdims=True)

    def is_constrained(self, value: Float[Array, " n"]) -> Bool[Array, ""]:
        floor = self.min_rel * self.total
        return jnp.all(value >= floor * (1 - 1e-12)) & jnp.isclose(
            jnp.sum(value), self.total, rtol=1e-12, atol=1e-12
        )

    def __repr__(self):
        return f"Widths(total={self.total}, min_rel={self.min_rel})"


free = Free()
arcsinh = Arcsinh()
