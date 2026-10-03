r"""Analytical bijections for normalizing flows, as introduced in [1]. The code largely
corresponds to that of the original [2], but with the neural network library changed
to equinox from flax.nnx.

[1] M. Gerdes and M. C. N. Cheng. Analytic bijections for smooth and interpretable
normalizing flows. ICML (2026).
[2] https://github.com/mathisgerdes/bijx/blob/master/src/bijx/bijections/analytic.py.
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
    """Solve cubic equation ax³ + bx² + cx + d = 0 using Cardano's formula.

    Uses numerically stable computation for the real root.
    """
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
    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    loc: Float[Array, ""]


class CubicRational(AbstractScalarBijection[CubicRationalParams]):
    """Modified rational transform with learnable parameters.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: x + α*x_/(1 + β*x_²), x_ = x - loc, with constrained α ∈ (-1, 8), β >
    eps.

    Raw parameters: ``(alpha, beta, loc)``; ``raw = 0`` is the identity (α = 0, β = 1).
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
        raw = jnp.stack(
            [
                Interval(-1 + eps_alpha, 8 - eps_alpha, at_zero=0.0).inverse(alpha),
                GreaterThan(eps_beta, at_zero=1.0).inverse(beta),
                jnp.asarray(loc),
            ]
        )
        return cls(raw=raw, eps_alpha=eps_alpha, eps_beta=eps_beta)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        p = self.params
        x_ = x - p.loc
        return x + p.alpha * x_ / (1 + p.beta * x_**2)

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        p = self.params
        y = y - p.loc
        x = solve_cubic(p.beta, -p.beta * y, p.alpha + 1, -y)
        return x + p.loc


def pow_overflow_log_arg(dtype, power):
    """log|arg| above which ``|arg|**power`` overflows ``dtype``.

    The clamp-free sinh helpers evaluate the singularity (arg=0) on a direct,
    gradient-clean path and only switch to the log-space asymptote once |arg| is
    too large for that path to stay finite.  ``power`` is 1 for the forward
    (forms ``arg``) and 2 for the log-Jac (forms ``arg**2``).  A 0.9 factor keeps
    a safety margin below the true overflow point and is dtype-aware so float32
    inputs switch to the asymptote well before float32 overflow.
    """
    return 0.9 * jnp.log(jnp.finfo(dtype).max) / power


def log_abs_sinh_stable(abs_x):
    """log|sinh(x)| = |x| - log 2 + log1p(-exp(-2|x|)); -> -inf at x=0 (sinh=0)."""
    return abs_x - jnp.log(2.0) + jnp.log1p(-jnp.exp(-2.0 * abs_x))


def _sinh_overflow_logabs_z(x, beta, mu, nu, power):
    """Shared overflow mask + ``log|z|`` for the all-order-safe sinh helpers.

    ``z = exp(mu) (exp(nu) sinh x + beta)``.  Returns ``(big, log_abs_z, sign_z)``
    where ``big`` marks ``|z|**power`` as too large for the direct primitive path
    (so the asymptote is used there), and ``log_abs_z`` / ``sign_z`` are computed
    with double-``where`` SAFE inputs so they are non-singular to all orders
    wherever ``big`` is False -- the unused overflow branch never poisons the
    reverse-mode gradient at the ``s=0`` / ``x=0`` singularity.

    Subtle higher-order point: in the overflow regime we must not form ``log|s|``
    from the raw ``s = exp(nu) sinh x + beta``.  Its value is fine, but its 2nd/3rd
    x-derivatives form ``cosh(x)^2`` / ``cosh(x)^3`` which overflow to inf (then
    inf/inf = NaN) for the very large ``|x|`` that put us in the overflow regime.
    So there we use the asymptote ``log|s| ~= nu + log|sinh x|`` (linear in x, clean
    derivatives); the dropped ``log|1 + beta/(exp(nu) sinh x)|`` term is ~0 with
    derivatives ~1/sinh x -> 0.  Exact ``log|s|`` is used only where ``big`` is
    False (s moderate), and the asymptote only where ``big`` is True.
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
    """Bijective nonlinearity: f(x) = arcsinh(exp(mu) * (exp(nu) * sinh(x) + beta)).

    Inverse is given by mu=-nu, nu=-mu, beta=-beta.

    Computed from the direct smooth primitive ``jnp.arcsinh(z)`` in the normal
    regime (autodiff-clean to all orders at the regular point ``z=0``/``s=0``), with
    a full double-``where`` fall-back to the asymptote ``sign(z)(log|z|+log2)`` once
    ``z`` would overflow.  Plain ``jax.grad^k`` of this is finite for every ``k``.

    ``power=4`` (not 1): ``arcsinh(z)``'s VALUE is fine while ``z`` is
    representable, but its k-th derivative forms ``z^(k+1)`` intermediates that
    overflow (silent wrong ``0`` derivative, or NaN at 3rd order) well before ``z``
    itself does; switching once ``z^4`` would overflow keeps grad, grad^2 and grad^3
    representable, and there ``arcsinh(z) = log|z| + log2`` (with all derivatives)
    holds to full float precision.
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
    loc: Float[Array, ""]
    scale: Float[Array, ""]
    beta: Float[Array, ""]
    mu: Float[Array, ""]
    nu: Float[Array, ""]


class SinhConjugation(AbstractScalarBijection[SinhConjugationParams]):
    """Sinh-based bijection using conjugation with arcsinh.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: arcsinh(exp(mu) * (exp(nu) * sinh((x-loc)/scale) + beta)) * scale + loc

    Raw parameters: ``(loc, scale, beta, mu, nu)``; ``scale`` is constrained to
    ``(eps_scale, inf)``, ``mu``/``nu`` are passed through ``arcsinh``. ``raw = 0`` is
    the identity. The inverse is the same map with ``(beta, mu, nu) -> (-beta, -nu,
    -mu)``.
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
        p = self.params
        x_norm = (x - p.loc) / p.scale
        return sinh_conj_nonlinearity(x_norm, p.beta, p.mu, p.nu) * p.scale + p.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        p = self.params
        y_norm = (y - p.loc) / p.scale
        return sinh_conj_nonlinearity(y_norm, -p.beta, -p.nu, -p.mu) * p.scale + p.loc


def _cubic_forward(x, a, b):
    return (a + b * x**2) * x


def _cubic_reverse(y, a, b):
    return solve_cubic(b, 0.0, a, -y)


def cubic_conj_nonlinearity(x, a=1, b=1, beta=0):
    return _cubic_reverse(_cubic_forward(x, a, b) + beta, a, b)


class CubicConjugationParams(NamedTuple):
    loc: Float[Array, ""]
    beta: Float[Array, ""]
    a: Float[Array, ""]
    b: Float[Array, ""]


class CubicConjugation(AbstractScalarBijection[CubicConjugationParams]):
    """Cubic polynomial-based bijection.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: g^{-1}(g(x - loc) + beta) + loc with g(x) = a x + b x³.

    Raw parameters: ``(loc, beta, a, b)``. ``a`` and ``b`` are bounded positive
    (``BoundedPositive``: ``eps + squashed_exp``), so a large conditioner output cannot
    drive them to infinity. ``raw = 0`` gives ``a = 1``, ``b = 0.3``, ``beta = 0``: the
    identity.
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
        p = self.params
        return cubic_conj_nonlinearity(x - p.loc, p.a, p.b, p.beta) + p.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        p = self.params
        return cubic_conj_nonlinearity(y - p.loc, p.a, p.b, -p.beta) + p.loc
