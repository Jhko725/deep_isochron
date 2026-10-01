r"""Constraint primitives: maps from unconstrained reals onto a constrained set.

A scalar bijection stores a single unconstrained vector ``raw`` as its trainable leaf
and
computes its constrained parameters on read through ``constrain(raw)``, built from the
primitives in this module (see ``docs/decisions/0001-unconstrained-leaves.md``). Each
primitive is a bijection in its own right, with ``__call__`` (raw -> constrained) and
``inverse`` (constrained -> raw) so that bijections can also be built from constrained
values (``from_constrained``).

Identity-at-zero convention (``docs/decisions/0002-identity-at-zero.md``): every
primitive
with an ``at_zero`` argument maps ``raw = 0`` to ``at_zero``. This puts the shift into
the *map* rather than into the initialisation, which is what makes a zero-initialised
conditioner (``CouplingFlow``) the identity.
"""

import abc
import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...misc import inv_softplus, inv_squashed_exp, squashed_exp


class Constraint(abc.ABC):
    """Map ``R^n -> S`` for a constrained set ``S``, with inverse ``S -> R^n``.

    Instances are plain, hashable Python objects (not modules); they hold only static
    configuration and are created inside ``constrain``.
    """

    @abc.abstractmethod
    def __call__(self, raw: Float[Array, " n"]) -> Float[Array, " n"]: ...

    @abc.abstractmethod
    def inverse(self, value: Float[Array, " n"]) -> Float[Array, " n"]: ...


class Free(Constraint):
    """The identity: no constraint."""

    def __call__(self, raw):
        return raw

    def inverse(self, value):
        return value


class Positive(Constraint):
    r"""``(eps, inf)`` via a shifted softplus: ``eps + softplus(raw + c)`` with ``c``
    chosen
    so that ``raw = 0`` maps to ``at_zero``.

    **Arguments:**

    - ``eps``: lower bound (exclusive). Default ``0``.
    - ``at_zero``: image of ``raw = 0``; must exceed ``eps``. Default ``1``.
    """

    def __init__(self, eps: float = 0.0, at_zero: float = 1.0):
        if not at_zero > eps:
            raise ValueError("at_zero must be greater than eps.")
        self.eps = eps
        self.at_zero = at_zero
        self._shift = math.log(math.expm1(at_zero - eps))

    def __call__(self, raw: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.eps + jax.nn.softplus(raw + self._shift)

    def inverse(self, value: Float[Array, " n"]) -> Float[Array, " n"]:
        return inv_softplus(value - self.eps) - self._shift

    def __repr__(self):
        return f"Positive(eps={self.eps}, at_zero={self.at_zero})"


class BoundedPositive(Constraint):
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
        if not math.exp(-a) < at_zero - eps < math.exp(a):
            raise ValueError("at_zero - eps must lie in (exp(-a), exp(a)).")
        self.eps = eps
        self.at_zero = at_zero
        self.a = a
        self._shift = a * math.atanh(math.log(at_zero - eps) / a)

    def __call__(self, raw: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.eps + squashed_exp(raw + self._shift, self.a)

    def inverse(self, value: Float[Array, " n"]) -> Float[Array, " n"]:
        return inv_squashed_exp(value - self.eps, self.a) - self._shift

    def __repr__(self):
        return f"BoundedPositive(eps={self.eps}, at_zero={self.at_zero}, a={self.a})"


class Interval(Constraint):
    r"""``(lo, hi)`` via a shifted sigmoid: ``lo + (hi - lo) * sigmoid(raw + c)`` with
    ``c``
    chosen so that ``raw = 0`` maps to ``at_zero``.

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
        self.at_zero = at_zero
        q = (at_zero - lo) / (hi - lo)
        self._shift = math.log(q / (1 - q))

    def __call__(self, raw: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.lo + (self.hi - self.lo) * jax.nn.sigmoid(raw + self._shift)

    def inverse(self, value: Float[Array, " n"]) -> Float[Array, " n"]:
        return (
            jax.scipy.special.logit((value - self.lo) / (self.hi - self.lo))
            - self._shift
        )

    def __repr__(self):
        return f"Interval(lo={self.lo}, hi={self.hi}, at_zero={self.at_zero})"


class Arcsinh(Constraint):
    """``R -> R`` via ``arcsinh``: a soft, symmetric compression of large raw values
    (used for the log-scale parameters of ``SinhConjugation``). ``raw = 0`` maps to
    ``0``."""

    def __call__(self, raw):
        return jnp.arcsinh(raw)

    def inverse(self, value):
        return jnp.sinh(value)


class Widths(Constraint):
    r"""``n`` positive widths summing to ``total``, each at least ``min_rel * total``:
    ``(softmax(raw) * (1 - n * min_rel) + min_rel) * total`` (the floored softmax of
    Durkan et al. 2019). ``raw = 0`` maps to equal widths.

    The softmax is shift-invariant, so the inverse is defined up to a constant; the
    gauge
    chosen here is **mean-zero raw** (``raw = log(rel - min_rel)`` centred), which is
    also
    the gauge in which equal widths map back to ``raw = 0``.

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

    def __repr__(self):
        return f"Widths(total={self.total}, min_rel={self.min_rel})"
