r"""The analytic bijections of Gerdes & Cheng 2026, ported from their ``bijx``
reference implementation
(https://github.com/mathisgerdes/bijx/blob/master/src/bijx/bijections/analytic.py)
with the neural-network library changed from ``flax.nnx`` to equinox.

``CubicRational`` is a rational perturbation of the identity, ``SinhConjugation`` a
conjugation of a shifted-and-scaled ``sinh`` by ``arcsinh``, and ``CubicConjugation`` a
conjugation of a shift by a monotone cubic. All three are in the ``raw``/``constrain``
shape with ``raw = 0`` the identity. The ``sinh`` helpers are written so that every
order of autodiff stays finite where the direct formulas overflow.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base import AbstractScalarBijection
from .constraints import arcsinh, BoundedPositive, free, GreaterThan, Interval


def solve_cubic(
    a: Float[ArrayLike, ""],
    b: Float[ArrayLike, ""],
    c: Float[ArrayLike, ""],
    d: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """The real root of ``a x³ + b x² + c x + d = 0`` by Cardano's formula, with the
    cube root taken of the larger of the two candidates for stability."""
    d0 = b**2 - 3 * a * c
    d1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

    sqrt = jnp.sqrt(d1**2 - 4 * d0**3)

    minus = d1 - sqrt
    plus = d1 + sqrt
    c_arg = jnp.where(
        jnp.abs(minus) < jnp.abs(plus),
        plus,
        minus,
    )
    c = jnp.cbrt(c_arg / 2)
    return -(b + c + d0 / c) / (3 * a)


class CubicRationalParams(NamedTuple):
    """The constrained parameters of `CubicRational`."""

    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    loc: Float[Array, ""]


class CubicRational(AbstractScalarBijection[CubicRationalParams]):
    """The rational perturbation of the identity ``x + α x' / (1 + β x'²)`` with
    ``x' = x - loc``, a bijection of the line for ``α ∈ (-1, 8)`` and ``β > 0``.

    The raw parameters are ``(alpha, beta, loc)``, with ``alpha`` an `Interval`
    shrunk by `eps_alpha` at each end and ``beta`` a `GreaterThan` with floor
    `eps_beta`; ``raw = 0`` is the identity, ``α = 0``, ``β = 1``.

    **Arguments:**

    - `raw`: The raw parameters; the identity when omitted.
    - `eps_alpha`, `eps_beta`: The margins of the constraints.
    """

    raw: Float[Array, " 3"] | None
    eps_alpha: float = eqx.field(static=True)
    eps_beta: float = eqx.field(static=True)
    num_params: int = eqx.field(static=True, default=3, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(
        self,
        *,
        raw: Float[Array, " 3"] | None = None,
        eps_alpha: float = 1e-3,
        eps_beta: float = 1e-1,
    ):
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta
        self.raw = self._init_raw(jnp.zeros(3) if raw is None else raw)

    def constrain(self, raw) -> CubicRationalParams:
        """``alpha`` through the interval, ``beta`` through the floor, ``loc`` free."""
        alpha, beta, loc = raw
        return CubicRationalParams(
            alpha=Interval(-1 + self.eps_alpha, 8 - self.eps_alpha, at_zero=0.0)(alpha),
            beta=GreaterThan(self.eps_beta, at_zero=1.0)(beta),
            loc=free(loc),
        )

    @classmethod
    def from_constrained(
        cls, alpha, beta, loc, *, eps_alpha: float = 1e-3, eps_beta: float = 1e-1
    ) -> "CubicRational":
        """The map with the constrained `alpha`, `beta` and `loc`."""
        raw = jnp.stack(
            [
                Interval(-1 + eps_alpha, 8 - eps_alpha, at_zero=0.0).inverse(alpha),
                GreaterThan(eps_beta, at_zero=1.0).inverse(beta),
                jnp.asarray(loc),
            ]
        )
        return cls(raw=raw, eps_alpha=eps_alpha, eps_beta=eps_beta)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """``x + α x' / (1 + β x'²)``."""
        p = self.params
        x_ = x - p.loc
        return x + p.alpha * x_ / (1 + p.beta * x_**2)

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """The root of the cubic that the forward map reduces to."""
        p = self.params
        y = y - p.loc
        x = solve_cubic(p.beta, -p.beta * y, p.alpha + 1, -y)
        return x + p.loc


def pow_overflow_log_arg(dtype, power):
    """The ``log|arg|`` above which ``|arg|**power`` overflows `dtype`, with a 0.9
    safety factor.

    The clamp-free ``sinh`` helpers evaluate the singularity at ``arg = 0`` on a
    direct, gradient-clean path and switch to the log-space asymptote only once
    ``|arg|`` is too large for that path to stay finite. The factor is dtype-aware,
    so float32 inputs switch well before float32 overflow.
    """
    return 0.9 * jnp.log(jnp.finfo(dtype).max) / power


def log_abs_sinh_stable(abs_x):
    """``log|sinh(x)|`` as ``|x| - log 2 + log1p(-exp(-2|x|))`` for `abs_x` $= |x|$;
    ``-inf`` at ``x = 0``."""
    return abs_x - jnp.log(2.0) + jnp.log1p(-jnp.exp(-2.0 * abs_x))


def _sinh_overflow_logabs_z(x, beta, mu, nu, power):
    """The overflow mask and ``log|z|`` shared by the all-order-safe ``sinh``
    helpers, for ``z = exp(mu) (exp(nu) sinh x + beta)``.

    **Returns:** ``(big, log_abs_z, sign_z)``. ``big`` marks where ``|z|**power`` is
    too large for the direct primitive path, so that the asymptote is used there;
    ``log_abs_z`` and ``sign_z`` are computed from double-``where`` safe inputs, so
    they are non-singular to all orders wherever ``big`` is false, and the unused
    overflow branch never poisons the reverse-mode gradient at the ``s = 0``
    singularity.

    In the overflow regime ``log|s|`` must not be formed from the raw
    ``s = exp(nu) sinh x + beta``: its value is fine, but its second and third
    derivatives in ``x`` form ``cosh(x)²`` and ``cosh(x)³``, which overflow to
    ``inf`` and then ``inf/inf = nan`` for the ``|x|`` that put the point in that
    regime. The asymptote ``log|s| ≈ nu + log|sinh x|`` is linear in ``x`` with
    clean derivatives, and the dropped term ``log|1 + beta/(exp(nu) sinh x)|`` is
    near 0 with derivatives near ``1/sinh x``. The exact ``log|s|`` is used only
    where ``big`` is false, the asymptote only where it is true.
    """
    abs_x = jnp.abs(x)

    # Overflow mask from magnitude estimate:
    # log|z| ~= mu + nu + (|x| - log 2) once |x| is large (sinh ~ e^|x|/2).
    log_abs_z_mask = mu + nu + (abs_x - jnp.log(2.0))
    big = log_abs_z_mask > pow_overflow_log_arg(x.dtype, power)

    # log|s| where ``big`` is FALSE: exact log(|s|).  Form ``s`` from a safe x
    # (sized down where big) so ``sinh(x)`` cannot overflow to +-inf on the unused
    # branch. An inf there poisons the reverse-mode cotangent even though the
    # ``where`` selects the other branch (0*inf = NaN through log/abs/sign).
    x_for_s = jnp.where(big, 0.0, x)
    s_exact = jnp.exp(nu) * jnp.sinh(x_for_s) + beta
    s_finite_nz = jnp.isfinite(s_exact) & (jnp.abs(s_exact) > 0.0)
    use_exact = (~big) & s_finite_nz
    s_safe = jnp.where(use_exact, s_exact, 1.0)
    log_abs_s_exact = jnp.log(jnp.abs(s_safe))
    sign_s = jnp.sign(s_safe)

    # log|s| where ``big`` is true: asymptote nu + log|sinh x| (clean to all orders
    # for large |x|); feed _log_abs_sinh a safe |x| where not big.
    abs_x_safe = jnp.where(big, abs_x, 1.0)
    log_abs_s_asymp = nu + log_abs_sinh_stable(abs_x_safe)

    log_abs_s = jnp.where(big, log_abs_s_asymp, log_abs_s_exact)
    log_abs_z = mu + log_abs_s

    # sign(z) = sign(s); in the overflow regime s is dominated by exp(nu) sinh x so
    # sign(s) = sign(x).  Guard sign(0) (zero subgradient) at the s=0 point.
    sign_z = jnp.where(big, jnp.sign(x), jnp.where(s_finite_nz, sign_s, 1.0))
    return big, log_abs_z, sign_z


def sinh_conj_nonlinearity(x: Float[Array, ""], beta=0.0, mu=0.0, nu=0.0):
    """The bijective nonlinearity ``arcsinh(exp(mu) (exp(nu) sinh(x) + beta))``, whose
    inverse is the same map with ``(beta, mu, nu) -> (-beta, -nu, -mu)``.

    It is computed from the direct primitive ``jnp.arcsinh(z)`` in the normal regime,
    where autodiff is clean to all orders at the regular point ``z = 0``, with a
    double-``where`` fallback to the asymptote ``sign(z) (log|z| + log 2)`` once ``z``
    would overflow, so that ``jax.grad`` of any order is finite. The switch uses
    ``power = 4`` rather than 1: the value of ``arcsinh(z)`` is fine while ``z`` is
    representable, but its ``k``-th derivative forms ``z**(k+1)`` intermediates that
    overflow well before ``z`` does, silently to a wrong 0 or to ``nan`` at third
    order; switching once ``z**4`` would overflow keeps the first three derivatives
    representable, and there the asymptote holds with all its derivatives to full
    float precision.
    """
    big, log_abs_z, sign_z = _sinh_overflow_logabs_z(x, beta, mu, nu, power=4)

    # Normal branch: arcsinh(z) directly (smooth, odd through 0).  Size x/mu down
    # where ``big`` so z stays representable on the (unused) gradient path.
    x_safe = jnp.where(big, 0.0, x)
    mu_safe = jnp.where(big, 0.0, mu)
    z = jnp.exp(mu_safe) * (jnp.exp(nu) * jnp.sinh(x_safe) + beta)
    direct = jnp.arcsinh(z)

    # Overflow branch: arcsinh(z) ~= sign(z)*(log|z| + log2).
    log_abs_z_safe = jnp.where(big, log_abs_z, 0.0)
    asymp = sign_z * (log_abs_z_safe + jnp.log(2.0))

    return jnp.where(big, asymp, direct)


class SinhConjugationParams(NamedTuple):
    """The constrained parameters of `SinhConjugation`."""

    loc: Float[Array, ""]
    scale: Float[Array, ""]
    beta: Float[Array, ""]
    mu: Float[Array, ""]
    nu: Float[Array, ""]


class SinhConjugation(AbstractScalarBijection[SinhConjugationParams]):
    """The bijection
    ``arcsinh(exp(mu) (exp(nu) sinh((x - loc) / scale) + beta)) scale + loc``, a
    conjugation of ``sinh`` by ``arcsinh``.

    The raw parameters are ``(loc, scale, beta, mu, nu)``: ``scale`` is a `GreaterThan`
    with floor `eps_scale`, ``mu`` and ``nu`` pass through `arcsinh`, the rest are
    free. ``raw = 0`` is the identity. The inverse is the same map with
    ``(beta, mu, nu) -> (-beta, -nu, -mu)``.

    **Arguments:**

    - `raw`: The raw parameters; the identity when omitted.
    - `eps_scale`: The floor of the scale.
    """

    raw: Float[Array, " 5"] | None
    eps_scale: float = eqx.field(static=True)
    num_params: int = eqx.field(static=True, default=5, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(
        self, *, raw: Float[Array, " 5"] | None = None, eps_scale: float = 0.1
    ):
        self.eps_scale = eps_scale
        self.raw = self._init_raw(jnp.zeros(5) if raw is None else raw)

    def constrain(self, raw) -> SinhConjugationParams:
        """``scale`` through the floor, ``mu`` and ``nu`` through `arcsinh`, the rest
        free."""
        loc, scale, beta, mu, nu = raw
        return SinhConjugationParams(
            loc=free(loc),
            scale=GreaterThan(self.eps_scale, at_zero=1.0)(scale),
            beta=free(beta),
            mu=arcsinh(mu),
            nu=arcsinh(nu),
        )

    @classmethod
    def from_constrained(
        cls, loc, scale, beta, mu, nu, *, eps_scale: float = 0.1
    ) -> "SinhConjugation":
        """The map with the constrained parameters."""
        raw = jnp.stack(
            [
                jnp.asarray(loc),
                GreaterThan(eps_scale, at_zero=1.0).inverse(scale),
                jnp.asarray(beta),
                arcsinh.inverse(mu),
                arcsinh.inverse(nu),
            ]
        )
        return cls(raw=raw, eps_scale=eps_scale)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """The nonlinearity in the normalized coordinate, scaled and shifted back."""
        p = self.params
        x_norm = (x - p.loc) / p.scale
        return sinh_conj_nonlinearity(x_norm, p.beta, p.mu, p.nu) * p.scale + p.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """The same with ``(beta, mu, nu) -> (-beta, -nu, -mu)``."""
        p = self.params
        y_norm = (y - p.loc) / p.scale
        return sinh_conj_nonlinearity(y_norm, -p.beta, -p.nu, -p.mu) * p.scale + p.loc


def _cubic_forward(x, a, b):
    """The monotone cubic ``g(x) = a x + b x³``."""
    return (a + b * x**2) * x


def _cubic_reverse(y, a, b):
    """The inverse of `_cubic_forward`, the real root of ``b x³ + a x - y = 0``."""
    return solve_cubic(b, 0.0, a, -y)


def cubic_conj_nonlinearity(x, a=1, b=1, beta=0):
    """The conjugation ``g⁻¹(g(x) + beta)`` of the shift by `beta` by the monotone
    cubic ``g(x) = a x + b x³``."""
    return _cubic_reverse(_cubic_forward(x, a, b) + beta, a, b)


class CubicConjugationParams(NamedTuple):
    """The constrained parameters of `CubicConjugation`."""

    loc: Float[Array, ""]
    beta: Float[Array, ""]
    a: Float[Array, ""]
    b: Float[Array, ""]


class CubicConjugation(AbstractScalarBijection[CubicConjugationParams]):
    """The bijection ``g⁻¹(g(x - loc) + beta) + loc`` with ``g(x) = a x + b x³``.

    The raw parameters are ``(loc, beta, a, b)``. ``a`` and ``b`` are `BoundedPositive`
    with floors `eps_a` and `eps_b`, so a large conditioner output cannot drive them
    to infinity. ``raw = 0`` gives ``a = 1``, ``b = 0.3`` and ``beta = 0``: the
    identity.

    **Arguments:**

    - `raw`: The raw parameters; the identity when omitted.
    - `eps_a`, `eps_b`: The floors of ``a`` and ``b``.
    """

    raw: Float[Array, " 4"] | None
    eps_a: float = eqx.field(static=True)
    eps_b: float = eqx.field(static=True)
    num_params: int = eqx.field(static=True, default=4, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(
        self,
        *,
        raw: Float[Array, " 4"] | None = None,
        eps_a: float = 1e-2,
        eps_b: float = 1e-2,
    ):
        self.eps_a = eps_a
        self.eps_b = eps_b
        self.raw = self._init_raw(jnp.zeros(4) if raw is None else raw)

    def constrain(self, raw) -> CubicConjugationParams:
        """``a`` and ``b`` bounded positive, ``loc`` and ``beta`` free."""
        loc, beta, a, b = raw
        return CubicConjugationParams(
            loc=free(loc),
            beta=free(beta),
            a=BoundedPositive(self.eps_a, at_zero=1.0)(a),
            b=BoundedPositive(self.eps_b, at_zero=0.3)(b),
        )

    @classmethod
    def from_constrained(
        cls, loc, beta, a, b, *, eps_a: float = 1e-2, eps_b: float = 1e-2
    ) -> "CubicConjugation":
        """The map with the constrained parameters."""
        raw = jnp.stack(
            [
                jnp.asarray(loc),
                jnp.asarray(beta),
                BoundedPositive(eps_a, at_zero=1.0).inverse(a),
                BoundedPositive(eps_b, at_zero=0.3).inverse(b),
            ]
        )
        return cls(raw=raw, eps_a=eps_a, eps_b=eps_b)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """The conjugation about `loc`."""
        p = self.params
        return cubic_conj_nonlinearity(x - p.loc, p.a, p.b, p.beta) + p.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """The same with ``beta -> -beta``."""
        p = self.params
        return cubic_conj_nonlinearity(y - p.loc, p.a, p.b, -p.beta) + p.loc
