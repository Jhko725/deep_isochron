r"""Floquet multipliers and exponents of a ``LimitCycle`` (roadmap O2).

The monodromy $M = \partial_x\Phi_T(x^\ast)$ of a periodic orbit has the eigenvalue $1$
(eigenvector $f(x^\ast)$, the trivial multiplier) and the nontrivial multipliers
$\mu_j$; $\kappa_j = \log\mu_j/T$ are the Floquet exponents — for a planar attracting
cycle one $\mu \in (0, 1)$, the rate $\dot\psi = \kappa\psi$ of the amplitude coordinate
(Wilson & Moehlis 2016, Eq. 7; Kvalheim & Revzen 2021, Prop. 5). This module computes
them from the shooting mesh ``x0s`` of a ``LimitCycle`` by three routes, one the method
and two cross-checks, all jitted:

**Periodic Schur chain** (the method, ``method="periodic_schur"``). The monodromy is the
ordered product of the segment Jacobians $M_i = \partial_x\Phi_{T/m}(x_i)$. Forming the
product and taking eigenvalues loses the small multiplier: by Bauer–Fike the eigenvalues
of the product carry an absolute error $\varepsilon\Vert M\Vert$, so a multiplier $\mu$
loses $\log_{10}(\max\vert\lambda\vert/\vert\mu\vert)$ digits (Lust 2001, Eq. 10). The
periodic Schur decomposition (Bojanczyk, Golub & Van Dooren 1992) finds orthogonal $Q_i$
with every $R_i = Q_{i+1}^{\top}M_iQ_i$ upper triangular, cyclically, so that
$\mu_j = \prod_i (R_i)_{jj}$ — a product of per-segment factors of moderate size,
accumulated here as a sum of logarithms (Lust 2001, §3; his Table 5). Their general
algorithm iterates to find $Q_1$; here one Schur vector is known — the flow direction,
eigenvector of the trivial multiplier, dominant for an attracting cycle — so the chain
is seeded with $q_1 = f(x_1)/\Vert f(x_1)\Vert$ (Guckenheimer & Meloon 2000 transform to
cross-section coordinates in the same spirit) and built by the forward QR chain
$M_iQ_i = Q_{i+1}R_i$ with positive diagonals: the discrete QR method for Lyapunov
exponents over one period (Dieci, Russell & Van Vleck 1997, §3 and Claim 2: it "computes
the real parts of the Floquet exponents of the computed approximation to the monodromy
matrix exactly"). The forward direction propagates the dominant direction, which is
self-correcting; the chain run backward through the transposes propagates the
subdominant one and does not close (checked numerically; no literature statement found).
The closure $Q_{m+1} = Q_1 D$, $D = \mathrm{diag}(\pm1)$, carries the multipliers' signs
and its defect is the trivial-multiplier diagnostic (Guckenheimer & Meloon §5.4). In the
plane the transverse block is $1 \times 1$, so no periodic QR iteration remains; for
``dim > 2`` the transverse block would need one (SLICOT's ``MB03WD``, wrapped by
``slycot``) — not implemented.

**Monodromy** (``method="monodromy"``): the explicit product and ``jnp.linalg.eigvals``
— the textbook route, with the floor above (relative error
$\varepsilon\Vert M\Vert/\mu$; $\approx 10^{-4}$ for FitzHugh–Nagumo's
$\mu \approx 1.7\cdot10^{-12}$), kept as the diagnostic it is.

**Liouville** (``method="liouville"``, planar only):
$\det M = \exp\int_0^T\mathrm{tr}\thinspace Df(\gamma(t))\thinspace dt$
(Abel–Liouville), and with $\lambda_1 = 1$ exactly,
$\kappa = T^{-1}\int_0^T\mathrm{div}f\thinspace dt$ (Andronov et al.; Perko 3rd ed. pp.
210–214; Mauroy & Mezić 2018, Remark 4) — a scalar quadrature in log space, exact in the
plane, independent of all linear algebra: the $m \to \infty$ limit of the product of
segment determinants. Computed by an augmented solve
$(\dot x, \dot z) = (f, \mathrm{tr}\thinspace Df)$ at the cycle's tolerances.

**Return map** (``return_map_multiplier``, a diagnostic that takes a ``PoincareMap``):
the nontrivial eigenvalue of ``jax.jacfwd(P)`` — the single-section route, whose
floor is the solver's absolute error on $DP$; *not* its trace, which is
$\mathrm{tr}\thinspace M - n^{\top}Mv/ \langle n, v\rangle = (1 + \mu) - 1$
algebraically but leaks the trivial multiplier's error ($\sim 10^{-9}$) additively onto
$\mu$.

**Reverse time.** The chain must run in the direction in which the cycle attracts
(otherwise $f$ is the subdominant direction and the seed is the unstable one), so for a
cycle found with ``reverse_time`` the segment Jacobians are those of the reversed field
($M_i^{-1}$ of the forward segments) and the reversed flow's multipliers
$\lbrace 1, 1/\mu\rbrace$ are inverted at the end: reported multipliers and exponents
are always forward-time (ADR-0014 §4)."""

from __future__ import annotations

from typing import Literal, NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE, SolverConfig
from ...systems.base import diffeqsolve
from .limit_cycle import LimitCycle
from .poincare import PoincareMap


FloquetMethod = Literal["periodic_schur", "monodromy", "liouville"]


class PeriodicSchur(NamedTuple):
    """The periodic Schur decomposition of a product ``M_m ⋯ M_1`` (arrays)."""

    Q: Float[Array, "m+1 dim dim"]
    """``Q[i]`` is the orthogonal frame at mesh point ``i`` (``Q[0]`` the seed frame,
    ``Q[m]`` the frame the chain closes on)."""
    R: Float[Array, "m dim dim"]
    """``R[i] = Q[i+1]ᵀ M_i Q[i]``, upper triangular with positive diagonal."""
    signs: Float[Array, " dim"]
    """``D = diag(signs)`` with ``Q[m] ≈ Q[0] D``: the multipliers' signs."""

    @property
    def log_multipliers(self) -> Float[Array, " dim"]:
        """``Σ_i log (R[i])_jj`` — the multipliers' log-moduli, in log space."""
        return jnp.sum(jnp.log(jnp.diagonal(self.R, axis1=-2, axis2=-1)), axis=0)

    @property
    def multipliers(self) -> Float[Array, " dim"]:
        return self.signs * jnp.exp(self.log_multipliers)

    @property
    def closure(self) -> Float[Array, ""]:
        """``‖Q[m] − Q[0] D‖``: zero if the seed frame was a Schur basis of the
        product — the trivial-multiplier defect of the chain."""
        return jnp.linalg.norm(self.Q[-1] - self.Q[0] * self.signs[None, :])


class Floquet(eqx.Module):
    """The Floquet data of a ``LimitCycle``, forward-time.

    **Attributes:**

    - `multipliers`: The multipliers `(dim,)`, sorted by modulus descending for the
        monodromy route and in Schur order for the periodic Schur route (index 0 is the
        trivial multiplier in both, for an attracting planar cycle).
    - `exponents`: `log(multipliers) / T` `(dim,)`.
    - `trivial_error`: `|multipliers[0] − 1|`, the accuracy of the orbit and solver
        together (Guckenheimer & Meloon 2000, §5.4); `None` for the Liouville route,
        which assumes the trivial multiplier exactly.
    - `closure`: The periodic Schur chain's closure defect; `None` for other routes.
    - `vectors`: The right Floquet vectors `(m, dim, dim−1)` of the nontrivial
        multipliers at every mesh point, from the periodic Schur route (O5's hook);
        `None` for other routes.
    - `method`: The route used (static).
    """

    multipliers: Float[Array, " dim"]
    exponents: Float[Array, " dim"]
    trivial_error: Float[Array, ""] | None
    closure: Float[Array, ""] | None
    vectors: Float[Array, "m dim dim-1"] | None
    method: str = eqx.field(static=True)


# --------------------------------------------------------------- the kernels ----
@eqx.filter_jit
def _segment_jacobians(
    ode: AbstractODE,
    x0s: Float[Array, "m dim"],
    period: Float[Array, ""],
    config: SolverConfig,
    reverse_time: bool,
) -> Float[Array, "m dim dim"]:
    sign = -1.0 if reverse_time else 1.0
    duration = sign * period / x0s.shape[0]
    fwd = config.with_adjoint(dfx.ForwardMode())
    return jax.vmap(jax.jacfwd(lambda x: ode.flow_to(duration, x, config=fwd)))(x0s)


def segment_jacobians(cycle: LimitCycle) -> Float[Array, "m dim dim"]:
    """The Jacobians ``M_i = ∂Φ_{T/m}(x0s[i])`` of the flow over each shooting
    segment, in the time direction in which the orbit attracts (the reversed field
    for a reverse-time cycle): ``M_m ⋯ M_1`` is the monodromy at ``x0s[0]``."""
    return _segment_jacobians(
        cycle.ode, cycle.x0s, cycle.period, cycle.config, cycle.reverse_time
    )


def _frame_from(q1: Float[Array, " dim"]) -> Float[Array, "dim dim"]:
    """An orthogonal matrix whose first column is the unit vector ``q1``: the
    Householder reflector mapping ``e_1`` to ``q1`` (the identity if ``q1 = e_1``)."""
    dim = q1.shape[0]
    e1 = jnp.zeros(dim).at[0].set(1.0)
    v = e1 - q1
    vv = jnp.dot(v, v)
    safe = jnp.where(vv > 0, vv, 1.0)
    H = jnp.eye(dim) - 2.0 * jnp.outer(v, v) / safe
    return jnp.where(vv > 0, H, jnp.eye(dim))


def _qr_positive(A: Float[Array, "dim dim"]) -> tuple[Array, Array]:
    """``A = Q R`` with the diagonal of ``R`` positive (the convention of Dieci,
    Russell & Van Vleck 1997, Eq. 3.3): the sign of row ``i`` of ``R`` and column
    ``i`` of ``Q`` are flipped together, which keeps the factorization exact."""
    Q, R = jnp.linalg.qr(A)
    s = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)
    return Q * s[None, :], R * s[:, None]


@eqx.filter_jit
def periodic_schur(
    jacobians: Float[Array, "m dim dim"], q1: Float[ArrayLike, " dim"]
) -> PeriodicSchur:
    """The periodic Schur decomposition of ``M_m ⋯ M_1`` by the forward QR chain
    seeded with ``q1``; see the module docstring.

    **Arguments:**

    - `jacobians`: The factors `(m, dim, dim)`, `jacobians[i]` mapping mesh point `i`
        to `i + 1`.
    - `q1`: A unit vector that is an eigenvector of the product for its dominant
        eigenvalue — the flow direction at mesh point 0 for an attracting cycle.

    **Returns:**

    A `PeriodicSchur`; its `multipliers` are the eigenvalues of the product.

    **Raises:**

    - `NotImplementedError` for `dim > 2`: the seed fixes one Schur vector only, and
        the remaining transverse block would need the periodic QR iteration
        (SLICOT `MB03WD`, available in Python through `slycot.math.mb03wd`).
    """
    dim = jacobians.shape[-1]
    if dim > 2:
        raise NotImplementedError(
            "periodic_schur is implemented for planar systems; for dim > 2 the "
            "transverse block needs the periodic QR iteration (slycot.math.mb03wd)."
        )
    q1 = jnp.asarray(q1, dtype=float)
    Q1 = _frame_from(q1 / jnp.linalg.norm(q1))

    def step(Q, M):
        Z, R = _qr_positive(M @ Q)
        return Z, (Z, R)

    Qm, (Zs, Rs) = jax.lax.scan(step, Q1, jacobians)
    Qs = jnp.concatenate([Q1[None], Zs], axis=0)
    signs = jnp.sign(jnp.diag(Q1.T @ Qm))
    return PeriodicSchur(Qs, Rs, signs)


@eqx.filter_jit
def monodromy(jacobians: Float[Array, "m dim dim"]) -> Float[Array, "dim dim"]:
    """The ordered product ``M_m ⋯ M_1`` of the segment Jacobians — the monodromy at
    mesh point 0."""
    dim = jacobians.shape[-1]
    M, _ = jax.lax.scan(lambda M, J: (J @ M, None), jnp.eye(dim), jacobians)
    return M


def multipliers_from_monodromy(M: Float[Array, "dim dim"]) -> Float[Array, " dim"]:
    """The eigenvalues of ``M``: the one closest to 1 (the trivial multiplier) first,
    the others by modulus, descending; real parts (the planar multipliers are real; a
    complex pair in higher dimension would be reported by its real part). Floor:
    relative error ``ε‖M‖/|μ|`` on a multiplier ``μ`` (Lust 2001, Eq. 10)."""
    ev = jnp.linalg.eigvals(M)
    trivial = jnp.argmin(jnp.abs(ev - 1.0))
    rest = jnp.delete(ev, trivial, assume_unique_indices=True)
    rest = rest[jnp.argsort(-jnp.abs(rest))]
    return jnp.real(jnp.concatenate([ev[trivial][None], rest]))


def return_map_multiplier(
    pmap: PoincareMap, x: Float[ArrayLike, " dim"]
) -> Float[Array, ""]:
    """The nontrivial multiplier from the return map's Jacobian at the fixed point
    ``x``: the largest eigenvalue of ``jax.jacfwd(P)(x)`` (whose other eigenvalue is
    0, along the flow). A single-section diagnostic: both eigenvalues of ``DP`` are
    small, so its floor is the solver's *absolute* error on ``DP`` (``~1e-9`` at the
    shooting tolerances) — fine for a moderately attracting cycle, meaningless for
    ``μ`` below that. Its trace is *not* used (see the module docstring)."""
    DP = jax.jacfwd(lambda u: pmap(u)[0])(jnp.asarray(x, dtype=float))
    ev = jnp.linalg.eigvals(DP)
    return jnp.real(ev[jnp.argmax(jnp.abs(ev))])


@eqx.filter_jit
def _divergence_integral(
    ode: AbstractODE,
    x0: Float[Array, " dim"],
    period: Float[Array, ""],
    config: SolverConfig,
    reverse_time: bool,
) -> Float[Array, ""]:
    sign = -1.0 if reverse_time else 1.0
    dim = x0.shape[0]

    def rhs(t, u, args):
        x = u[:dim]
        div = jnp.trace(jax.jacfwd(lambda y: ode.rhs(t, y))(x))
        return jnp.concatenate([sign * ode.rhs(t, x), div[None]])

    ts = jnp.stack([jnp.zeros_like(period), period])
    sol = diffeqsolve(rhs, ts, jnp.concatenate([x0, jnp.zeros(1)]), None, config)
    ys = sol.ys
    assert ys is not None
    return ys[-1, dim]


def divergence_integral(cycle: LimitCycle) -> Float[Array, ""]:
    r"""$\int_0^T \mathrm{div} f(\gamma(t))\thinspace dt = \log\mu$ for a planar orbit
    (Liouville), by an augmented solve along the orbit at the cycle's tolerances, in
    the direction in which the orbit attracts (the integrand is the forward field's
    divergence either way).

    **Raises:**

    - `ValueError` unless `cycle.dim == 2`: in higher dimension the integral is the
        sum of all exponents, not the one we want.
    """
    if cycle.dim != 2:
        raise ValueError("the Liouville route is defined for planar orbits only.")
    return _divergence_integral(
        cycle.ode, cycle.x0s[0], cycle.period, cycle.config, cycle.reverse_time
    )


# ----------------------------------------------------------------- the driver ----
def _floquet_vectors(ps: PeriodicSchur) -> Float[Array, "m dim dim-1"]:
    """Right eigenvectors of the nontrivial multipliers at every mesh point, from the
    cyclic products of the triangular factors (planar: a 2 × 2 triangular
    eigenproblem in the ``Q[i]`` frame, mapped back to the ambient frame)."""
    R, Q = ps.R, ps.Q[:-1]
    m = R.shape[0]

    def product_at(i):
        # T_i = R_{i-1} ⋯ R_0 R_{m-1} ⋯ R_i, the monodromy in the Q[i] frame
        order = (jnp.arange(m) + i) % m
        T, _ = jax.lax.scan(lambda T, k: (R[k] @ T, None), jnp.eye(2), order)
        return T

    T = jax.vmap(product_at)(jnp.arange(m))  # (m, 2, 2), upper triangular
    # eigenvector of the (2,2) entry: (T01 / (T11 − T00), 1)
    v = jnp.stack([T[:, 0, 1] / (T[:, 1, 1] - T[:, 0, 0]), jnp.ones(m)], axis=1)
    v = jnp.einsum("mij,mj->mi", Q, v)
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    return v[:, :, None]


def floquet(cycle: LimitCycle, *, method: FloquetMethod = "periodic_schur") -> Floquet:
    """The Floquet multipliers and exponents of `cycle`, forward-time.

    **Arguments:**

    - `cycle`: The `LimitCycle` (its `ode`, `config` and `reverse_time` are used).
    - `method`: `"periodic_schur"` (the method: the seeded forward QR chain on the
        segment Jacobians), `"monodromy"` (eigenvalues of the explicit product; a
        diagnostic with the floor `ε‖M‖/μ`), or `"liouville"` (the planar divergence
        integral; a cross-check independent of all linear algebra).

    **Returns:**

    A `Floquet`.

    **Raises:**

    - `ValueError` for an unknown `method`, or `"liouville"` on a non-planar cycle.
    - `NotImplementedError` for `"periodic_schur"` with `cycle.dim > 2`.
    """
    T = cycle.period
    if method == "periodic_schur":
        J = segment_jacobians(cycle)
        f = cycle.ode.rhs(0.0, cycle.x0s[0])
        ps = periodic_schur(J, f / jnp.linalg.norm(f))
        log_mu = ps.log_multipliers * (-1.0 if cycle.reverse_time else 1.0)
        mult = ps.signs * jnp.exp(log_mu)
        return Floquet(
            multipliers=mult,
            exponents=log_mu / T,
            trivial_error=jnp.abs(mult[0] - 1.0),
            closure=ps.closure,
            vectors=_floquet_vectors(ps) if cycle.dim == 2 else None,
            method=method,
        )
    if method == "monodromy":
        mult = multipliers_from_monodromy(monodromy(segment_jacobians(cycle)))
        if cycle.reverse_time:  # the reversed flow's {1, 1/μ}
            mult = 1.0 / mult
        return Floquet(
            multipliers=mult,
            exponents=jnp.log(jnp.abs(mult)) / T,
            trivial_error=jnp.abs(mult[0] - 1.0),
            closure=None,
            vectors=None,
            method=method,
        )
    if method == "liouville":
        log_mu = divergence_integral(cycle)
        mult = jnp.stack([jnp.ones_like(log_mu), jnp.exp(log_mu)])
        return Floquet(
            multipliers=mult,
            exponents=jnp.stack([jnp.zeros_like(log_mu), log_mu]) / T,
            trivial_error=None,
            closure=None,
            vectors=None,
            method=method,
        )
    raise ValueError(
        f"unknown method {method!r}; use 'periodic_schur', 'monodromy' or 'liouville'."
    )


__all__ = [
    "Floquet",
    "FloquetMethod",
    "PeriodicSchur",
    "divergence_integral",
    "floquet",
    "monodromy",
    "multipliers_from_monodromy",
    "periodic_schur",
    "return_map_multiplier",
    "segment_jacobians",
]
