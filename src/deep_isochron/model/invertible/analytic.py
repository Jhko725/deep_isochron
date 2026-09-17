r"""Analytical bijections for normalizing flows, as introduced in [1]. The code largely
corresponds to that of the original [2], but with the neural network library changed
to equinox from flax.nnx.

[1] M. Gerdes and M. C. N. Cheng. Analytic bijections for smooth and interpretable
normalizing flows. ICML (2026).
[2] https://github.com/mathisgerdes/bijx/blob/master/src/bijx/bijections/analytic.py.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from deep_isochron.misc import inv_softplus, inv_squashed_exp, squashed_exp

from .base import AbstractBijection


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


class CubicRational(AbstractBijection):
    """Modified rational transform with learnable parameters.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: x + α*x/(1 + β*x²) with constrained α ∈ [-1,8], β > 0.
    """

    dim: ClassVar[int] = 1  # ty: ignore

    _alpha: Float[Array, ""]
    _beta: Float[Array, ""]
    loc: Float[Array, ""]
    eps_alpha: float = eqx.field(static=True)
    eps_beta: float = eqx.field(static=True)

    def __init__(
        self, alpha, beta, loc, eps_alpha: float = 1e-3, eps_beta: float = 1e-1
    ):
        self._alpha = alpha
        self._beta = beta
        self.loc = loc
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta

    @property
    def alpha(self) -> Float[Array, ""]:
        alpha_low, alpha_high = -1 + self.eps_alpha, 8 - self.eps_alpha
        alpha = alpha_low + (alpha_high - alpha_low) * jax.nn.sigmoid(
            self._alpha + jax.scipy.special.logit(-alpha_low / (alpha_high - alpha_low))
        )
        return alpha

    @property
    def beta(self) -> Float[Array, ""]:
        """Unlike the original implementation, the beta initialization was slightly
        changed from `softplus(beta+1)` to `softplus(beta+inv_softplus(1-eps_beta))`.
        This is so that alpha,beta,loc=(0,0,0) results in the identity transform."""
        beta = self.eps_beta + jax.nn.softplus(
            self._beta + inv_softplus(1.0 - self.eps_beta)
        )
        return beta

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        x_ = x - self.loc
        return x + self.alpha * x_ / (1 + self.beta * x_**2)

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        y = y - self.loc
        x = solve_cubic(
            self.beta,
            -self.beta * y,
            self.alpha + 1,
            -y,
        )
        return x + self.loc

    @classmethod
    def from_unnormalized_params(
        cls, alpha, beta, loc, eps_alpha: float = 1e-3, eps_beta: float = 1e-1
    ) -> Self:
        """Instantiate the transform using unconstrained parameters."""
        loc = loc
        beta = eps_beta + jax.nn.softplus(beta + inv_softplus(1.0 - eps_beta))
        alpha_low, alpha_high = -1 + eps_alpha, 8 - eps_alpha
        alpha = alpha_low + (alpha_high - alpha_low) * jax.nn.sigmoid(
            alpha + jax.scipy.special.logit(-alpha_low / (alpha_high - alpha_low))
        )
        return cls(alpha=alpha, beta=beta, loc=loc)


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


class SinhConjugation(AbstractBijection):
    """Sinh-based bijection using conjugation with arcsinh.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: arcsinh(exp(mu) * (exp(nu) * sinh((x-loc)/alpha) + beta)) * alpha + loc

    Parameters:
        loc: Location parameter (shift)
        scale: Scale parameter (must be positive)
        beta: Offset parameter in sinh space
        mu: Log-scale parameter for outer stretch
        nu: Log-scale parameter for inner stretch
    """

    dim: ClassVar[int] = 1  # ty: ignore

    loc: Float[Array, ""]
    _scale: Float[Array, ""]
    beta: Float[Array, ""]
    _mu: Float[Array, ""]
    _nu: Float[Array, ""]

    eps_scale: float = eqx.field(static=True)

    def __init__(self, loc, scale, beta, mu, nu, eps_scale: float = 0.1):
        self.loc = loc
        self._scale = scale
        self.beta = beta
        self._mu = mu
        self._nu = nu
        self.eps_scale = eps_scale

    @property
    def scale(self) -> Float[Array, ""]:
        """Unlike the original implementation, the scale initialization was slightly
        changed from `softplus(scale+1)` to `softplus(scale+inv_softplus(1-eps_scale))`.
        This is so that alpha,beta,loc=(0,0,0) results in the identity transform."""
        return self.eps_scale + jax.nn.softplus(
            self._scale + inv_softplus(1.0 - self.eps_scale)
        )

    @property
    def mu(self) -> Float[Array, ""]:
        return jnp.arcsinh(self._mu)

    @property
    def nu(self) -> Float[Array, ""]:
        return jnp.arcsinh(self._nu)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        x_norm = (x - self.loc) / self.scale
        return (
            sinh_conj_nonlinearity(x_norm, self.beta, self.mu, self.nu) * self.scale
            + self.loc
        )

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        y_norm = (y - self.loc) / self.scale
        return (
            sinh_conj_nonlinearity(y_norm, -self.beta, -self.nu, -self.mu) * self.scale
            + self.loc
        )


def _cubic_forward(x, a, b):
    return (a + b * x**2) * x


def _cubic_reverse(y, a, b):
    return solve_cubic(b, 0.0, a, -y)


def cubic_conj_nonlinearity(x, a=1, b=1, beta=0):
    return _cubic_reverse(_cubic_forward(x, a, b) + beta, a, b)


class CubicConjugation(AbstractBijection):
    """Cubic polynomial-based bijection.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: Based on cubic polynomial a*x + b*x³ with conjugation offset

    Parameters:
        loc: Location parameter (shift)
        beta: Offset parameter for conjugation
        a: Linear coefficient (must be positive)
        b: Cubic coefficient (must be positive)
    """

    dim: ClassVar[int] = 1  # ty: ignore

    loc: Float[Array, ""]
    beta: Float[Array, ""]
    _a: Float[Array, ""]
    _b: Float[Array, ""]

    eps_a: float = eqx.field(static=True)
    eps_b: float = eqx.field(static=True)

    def __init__(self, loc, beta, a, b, eps_a: float = 1e-2, eps_b=1e-2):
        self.loc = loc
        self.beta = beta
        self._a = a
        self._b = b
        self.eps_a = eps_a
        self.eps_b = eps_b

    @property
    def a(self) -> Float[Array, ""]:
        return self.eps_a + squashed_exp(self._a + inv_squashed_exp(1.0 - self.eps_a))

    @property
    def b(self) -> Float[Array, ""]:
        return self.eps_b + squashed_exp(self._b + inv_squashed_exp(0.3 - self.eps_b))

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        return (
            cubic_conj_nonlinearity(x - self.loc, self.a, self.b, self.beta) + self.loc
        )

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        return (
            cubic_conj_nonlinearity(y - self.loc, self.a, self.b, -self.beta) + self.loc
        )
