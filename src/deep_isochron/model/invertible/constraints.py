r"""Constraint primitives: maps from unconstrained reals onto a constrained set.

A scalar bijection stores one unconstrained vector ``raw`` as its trainable leaf and
computes its constrained parameters on read through ``constrain(raw)``, built from the
primitives here (ADR-0001, ADR-0005). Each primitive is a bijection ``R^n -> S`` in
its own right, with ``__call__`` from raw to constrained, ``inverse`` back, and
``is_constrained`` for membership in ``S``, so that bijections can also be built from
constrained values and tests can check the image generically.

Every primitive has two independent pieces of configuration: the constrained set, and
the point of that set that ``raw = 0`` maps to, ``at_zero``. The second exists because
a scalar bijection must be the identity at ``raw = 0`` (ADR-0002), and the identity's
parameter value, a scale of 1 or a slope of 1, is usually not what the unshifted map
gives at 0. Each primitive has a default for ``at_zero`` and accepts an explicit value
when the identity needs another one. The shift is in the map, not in the
initialization: a shifted primitive implements the unshifted pair ``_forward`` and
``_inverse`` once, and the shift is evaluated under ``jax.ensure_compile_time_eval``,
so that it is a concrete float even inside a ``jit`` or ``vmap`` trace. The stateless
primitives are also module-level singletons ``free`` and ``arcsinh``.
"""

import abc
import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ...utils.numerics import inv_softplus, inv_squashed_exp, squashed_exp


class Constraint(abc.ABC):
    """A map from unconstrained reals onto a constrained set ``S``, with its inverse.

    An elementwise primitive acts on arrays of any shape, a 0-d parameter or a block
    of a raw vector alike; `Widths` is intrinsically a vector map and narrows the
    annotation. Instances are plain, hashable Python objects, not modules; they hold
    only static configuration and are created inside ``constrain`` (ADR-0005).
    """

    @abc.abstractmethod
    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """The constrained value of `raw`."""

    @abc.abstractmethod
    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """The raw value of the constrained `value`."""

    @abc.abstractmethod
    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether every element of `value` lies in ``S``."""


class _Shifted(Constraint):
    """An elementwise primitive ``value = _forward(raw + shift)`` with the shift chosen
    so that ``raw = 0`` maps to `at_zero`. A subclass implements the unshifted pair
    `_forward` and `_inverse` and `is_constrained`; `__call__`, `inverse` and `_shift`
    derive from them."""

    at_zero: float

    def _init_shift(self, at_zero: float) -> None:
        """Store `at_zero`; the shift itself is computed on first use."""
        self.at_zero = at_zero

    @functools.cached_property
    def _shift(self) -> float:
        """The shift ``_inverse(at_zero)``, computed on first use rather than at
        construction.

        A module-level primitive must not initialize a JAX backend at import time,
        since a grain worker process importing the package would otherwise grab GPU
        memory (ADR-0005). The value is concrete even inside a trace, because
        constructing a primitive under ``jit`` or ``vmap`` must not stage this
        computation; that requires `_inverse` to be plain ``jnp`` with no
        jit-decorated helper (see `Interval`).
        """
        with jax.ensure_compile_time_eval():
            return float(self._inverse(jnp.asarray(self.at_zero)))

    @abc.abstractmethod
    def _forward(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """The unshifted map."""

    @abc.abstractmethod
    def _inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """The inverse of the unshifted map."""

    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """``_forward(raw + shift)``."""
        return self._forward(raw + self._shift)

    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """``_inverse(value) - shift``."""
        return self._inverse(value) - self._shift


class Free(Constraint):
    """The identity: no constraint, and ``raw = 0`` maps to 0."""

    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """`raw` itself."""
        return raw

    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """`value` itself."""
        return value

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether `value` is finite."""
        return jnp.all(jnp.isfinite(value))


class Arcsinh(Constraint):
    """``R -> R`` through ``arcsinh``, a soft, symmetric compression of large raw
    values, used for the log-scale parameters of ``SinhConjugation``; ``raw = 0`` maps
    to 0."""

    def __call__(self, raw: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """``arcsinh(raw)``."""
        return jnp.arcsinh(raw)

    def inverse(self, value: Float[Array, " *n"]) -> Float[Array, " *n"]:
        """``sinh(value)``."""
        return jnp.sinh(value)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether `value` is finite."""
        return jnp.all(jnp.isfinite(value))


class GreaterThan(_Shifted):
    r"""``(lower, inf)`` through a shifted softplus, ``lower + softplus(raw + c)``,
    with ``c`` chosen so that ``raw = 0`` maps to `at_zero`.

    **Arguments:**

    - `lower`: The exclusive lower bound.
    - `at_zero`: The image of ``raw = 0``, above `lower`; ``lower + 1`` by default, so
        that ``GreaterThan(0.0)`` maps 0 to 1, a unit scale, and ``GreaterThan(-1.0)``
        maps 0 to 0, a vanishing coefficient.

    **Raises:** `ValueError` unless ``at_zero > lower``.
    """

    def __init__(self, lower: float, at_zero: float | None = None):
        at_zero = lower + 1.0 if at_zero is None else at_zero
        if not at_zero > lower:
            raise ValueError("at_zero must be greater than lower.")
        self.lower = lower
        self._init_shift(at_zero)

    def _forward(self, raw):
        """``lower + softplus(raw)``."""
        return self.lower + jax.nn.softplus(raw)

    def _inverse(self, value):
        """The inverse softplus of ``value - lower``."""
        return inv_softplus(value - self.lower)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether every element exceeds `lower`."""
        return jnp.all(value > self.lower)

    def __repr__(self):
        return f"{type(self).__name__}(lower={self.lower}, at_zero={self.at_zero})"


class Positive(GreaterThan):
    """``GreaterThan(0.0, at_zero)``: strictly positive values, with ``raw = 0``
    mapped to `at_zero`, 1 by default."""

    def __init__(self, at_zero: float = 1.0):
        super().__init__(0.0, at_zero)


class BoundedPositive(_Shifted):
    r"""``(lower + exp(-a), lower + exp(a))`` through a shifted ``squashed_exp``,
    ``lower + exp(a * tanh((raw + c) / a))``, with ``c`` chosen so that ``raw = 0``
    maps to `at_zero`.

    Unlike `GreaterThan` the image is bounded above, so a large conditioner output
    cannot drive the value to infinity; used for the cubic generator's ``a`` and ``b``
    and for ``Affine``'s scale.

    **Arguments:**

    - `lower`: The lower offset; 0 by default.
    - `at_zero`: The image of ``raw = 0``; ``at_zero - lower`` must lie in
        ``(exp(-a), exp(a))``. ``lower + 1`` by default.
    - `a`: The squashing scale; 2 by default.

    **Raises:** `ValueError` for an `at_zero` outside the image.
    """

    def __init__(
        self, lower: float = 0.0, at_zero: float | None = None, a: float = 2.0
    ):
        import math

        at_zero = lower + 1.0 if at_zero is None else at_zero
        if not math.exp(-a) < at_zero - lower < math.exp(a):
            raise ValueError("at_zero - lower must lie in (exp(-a), exp(a)).")
        self.lower = lower
        self.a = a
        self._init_shift(at_zero)

    def _forward(self, raw):
        """``lower + squashed_exp(raw, a)``."""
        return self.lower + squashed_exp(raw, self.a)

    def _inverse(self, value):
        """The inverse squashed exponential of ``value - lower``."""
        return inv_squashed_exp(value - self.lower, self.a)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether every element lies in the open image."""
        lo, hi = self.lower + jnp.exp(-self.a), self.lower + jnp.exp(self.a)
        return jnp.all((value > lo) & (value < hi))

    def __repr__(self):
        return (
            f"BoundedPositive(lower={self.lower}, at_zero={self.at_zero}, a={self.a})"
        )


class Interval(_Shifted):
    r"""``(lo, hi)`` through a shifted sigmoid, ``lo + (hi - lo) * sigmoid(raw + c)``,
    with ``c`` chosen so that ``raw = 0`` maps to `at_zero`.

    **Arguments:**

    - `lo`, `hi`: The open interval's bounds, ``lo < hi``.
    - `at_zero`: The image of ``raw = 0``, strictly inside the interval; the midpoint
        by default.

    **Raises:** `ValueError` otherwise.
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
        """``lo + (hi - lo) * sigmoid(raw)``."""
        return self.lo + (self.hi - self.lo) * jax.nn.sigmoid(raw)

    def _inverse(self, value):
        """The logit of the position of `value` in the interval, in plain ``jnp``.

        ``jax.scipy.special.logit`` is jit-decorated, and a jitted call inside a
        ``vmap`` trace returns a tracer even for a concrete argument, which would
        defeat ``ensure_compile_time_eval`` in `_shift`.
        """
        q = (value - self.lo) / (self.hi - self.lo)
        return jnp.log(q) - jnp.log1p(-q)

    def is_constrained(self, value: Float[Array, " *n"]) -> Bool[Array, ""]:
        """Whether every element lies strictly inside the interval."""
        return jnp.all((value > self.lo) & (value < self.hi))

    def __repr__(self):
        return f"Interval(lo={self.lo}, hi={self.hi}, at_zero={self.at_zero})"


class Widths(Constraint):
    r"""``n`` positive widths summing to `total`, each at least ``min_rel * total``:
    ``(softmax(raw) * (1 - n * min_rel) + min_rel) * total``, the floored softmax of
    Durkan et al. 2019. ``raw = 0`` maps to equal widths.

    The softmax is shift-invariant, so the inverse is defined up to a constant; the
    gauge chosen here is mean-zero raw, which is also the gauge in which equal widths
    map back to ``raw = 0``.

    **Arguments:**

    - `total`: The sum of the widths, a float or a 0-d array that may be traced.
    - `min_rel`: The minimum relative width, non-negative. ``n * min_rel < 1`` is
        checked per call, because ``n`` is the length of the raw vector.

    **Raises:** `ValueError` for a negative `min_rel` at construction, and for
    ``n * min_rel >= 1`` at a call.
    """

    def __init__(self, total, min_rel: float = 1e-3):
        if min_rel < 0:
            raise ValueError("min_rel must be non-negative.")
        self.total = total
        self.min_rel = min_rel

    def _check_n(self, n: int) -> None:
        """Raise `ValueError` unless ``n * min_rel < 1``."""
        if n * self.min_rel >= 1:
            raise ValueError(
                f"n * min_rel = {n * self.min_rel} must be < 1 for n = {n} widths."
            )

    def __call__(self, raw: Float[Array, " n"]) -> Float[Array, " n"]:
        """The widths, shape ``(n,)``."""
        n = raw.shape[-1]
        self._check_n(n)
        rel = jax.nn.softmax(raw, axis=-1) * (1 - n * self.min_rel) + self.min_rel
        return rel * self.total

    def inverse(self, value: Float[Array, " n"]) -> Float[Array, " n"]:
        """The mean-zero raw vector of the widths `value`."""
        n = value.shape[-1]
        self._check_n(n)
        rel = value / self.total
        raw = jnp.log((rel - self.min_rel) / (1 - n * self.min_rel))
        return raw - jnp.mean(raw, axis=-1, keepdims=True)

    def is_constrained(self, value: Float[Array, " n"]) -> Bool[Array, ""]:
        """Whether `value` is at or above the floor and sums to `total`, to
        round-off."""
        floor = self.min_rel * self.total
        return jnp.all(value >= floor * (1 - 1e-12)) & jnp.isclose(
            jnp.sum(value), self.total, rtol=1e-12, atol=1e-12
        )

    def __repr__(self):
        return f"Widths(total={self.total}, min_rel={self.min_rel})"


free = Free()
"""The `Free` singleton."""
arcsinh = Arcsinh()
"""The `Arcsinh` singleton; ``arcsinh(mu)`` reads better than ``Arcsinh()(mu)``."""
