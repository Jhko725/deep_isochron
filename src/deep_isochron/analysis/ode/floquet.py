r"""Floquet multipliers and exponents of a ``LimitCycle`` (roadmap O2).

The monodromy of a periodic orbit has the trivial multiplier $1$, with eigenvector
$f(x^\ast)$, and the nontrivial multipliers $\mu_j$; the exponents are
$\kappa_j = \log\mu_j / T$. For a planar attracting cycle the one nontrivial exponent
is the decay rate of the amplitude coordinate (Wilson & Moehlis 2016, Eq. 7;
Kvalheim & Revzen 2021, Prop. 5).

``floquet(cycle, method=...)`` computes them from the shooting mesh of the cycle by one
of three routes, each also callable as ``floquet_<method>``: ``"periodic_schur"``, the
method, a periodic Schur decomposition of the segment Jacobians seeded with the flow
direction (``PeriodicSchur``), which also returns the Floquet vectors;
``"monodromy"``, the eigenvalues of the explicit product, a diagnostic limited by
rounding; and ``"liouville"``, planar only, the exponent from the divergence
integral. ``return_map_multiplier`` is a fourth diagnostic that takes a
``PoincareMap``. The kernels (``segment_jacobians``, ``periodic_schur``,
``monodromy``, ``divergence_integral``) are jit- and vmap-compatible functions of the
cycle and are left unjitted, so the caller chooses the level.

Two invariants span the routes. The chain runs in the time direction in which the
cycle attracts, so for a cycle found with ``reverse_time`` the segment Jacobians are
those of the reversed field and the result is inverted at the end: reported
multipliers and exponents are always forward-time (ADR-0014 §4). And only
``dim == 2`` is implemented for the periodic Schur route.

ADR-0015 has the choice of method, the direction rule and the rejected routes;
``docs/design/floquet.md`` has the survey, the accuracy floors and the citations.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ...systems.base import diffeqsolve
from .limit_cycle import LimitCycle
from .poincare import PoincareMap


FloquetMethod = Literal["periodic_schur", "monodromy", "liouville"]
"""The names `floquet` dispatches on."""


class PeriodicSchur(NamedTuple):
    """The periodic Schur decomposition of a product ``M_m ⋯ M_1``, as arrays.

    **Attributes:**

    - `Q`: The orthogonal frames, shape ``(m+1, dim, dim)``; ``Q[i]`` is the frame at
        mesh point ``i``, ``Q[0]`` the seed frame and ``Q[m]`` the frame the chain
        closes on.
    - `R`: The triangular factors ``R[i] = Q[i+1]ᵀ M_i Q[i]``, shape
        ``(m, dim, dim)``, upper triangular with positive diagonal.
    - `signs`: The diagonal of ``D`` in ``Q[m] ≈ Q[0] D``, shape ``(dim,)``: the
        multipliers' signs.
    """

    Q: Float[Array, "m+1 dim dim"]
    R: Float[Array, "m dim dim"]
    signs: Float[Array, " dim"]

    @property
    def log_multipliers(self) -> Float[Array, " dim"]:
        r"""The multipliers' log-moduli, ``Σ_i log (R[i])_jj``, accumulated in log
        space so that a multiplier as small as $10^{-16}$ keeps its digits."""
        return jnp.sum(jnp.log(jnp.diagonal(self.R, axis1=-2, axis2=-1)), axis=0)

    @property
    def multipliers(self) -> Float[Array, " dim"]:
        """The multipliers, ``signs · exp(log_multipliers)``."""
        return self.signs * jnp.exp(self.log_multipliers)

    @property
    def closure(self) -> Float[Array, ""]:
        """The closure defect ``‖Q[m] − Q[0] D‖``: zero if the seed frame was a Schur
        basis of the product, so the chain's trivial-multiplier diagnostic."""
        return jnp.linalg.norm(self.Q[-1] - self.Q[0] * self.signs[None, :])


class Floquet(eqx.Module):
    """The Floquet data of a ``LimitCycle``, in forward time.

    **Attributes:**

    - `multipliers`: The multipliers, shape ``(dim,)``: in Schur order for the
        periodic Schur route and sorted by modulus descending for the monodromy
        route; index 0 is the trivial multiplier in both, for an attracting planar
        cycle.
    - `exponents`: ``log(multipliers) / T``, shape ``(dim,)``.
    - `trivial_error`: ``|multipliers[0] − 1|``, the accuracy of orbit and solver
        together (Guckenheimer & Meloon 2000, §5.4); ``None`` for the Liouville
        route, which assumes the trivial multiplier exactly.
    - `closure`: The periodic Schur chain's closure defect; ``None`` for the other
        routes.
    - `vectors`: The right Floquet vectors of the nontrivial multipliers at every
        mesh point, shape ``(m, dim, dim−1)``, from the periodic Schur route (the
        hook for O5); ``None`` for the other routes.
    - `method`: The route used (static).
    """

    multipliers: Float[Array, " dim"]
    exponents: Float[Array, " dim"]
    trivial_error: Float[Array, ""] | None
    closure: Float[Array, ""] | None
    vectors: Float[Array, "m dim dim-1"] | None
    method: str = eqx.field(static=True)


# --------------------------------------------------------------- the kernels ----
def segment_jacobians(cycle: LimitCycle) -> Float[Array, "m dim dim"]:
    """The Jacobians ``M_i = ∂Φ_{T/m}(x0s[i])`` of the flow over each shooting segment
    of `cycle`, shape ``(m, dim, dim)``, by forward-mode AD through the solver.

    They are taken in the time direction in which the orbit attracts, the reversed
    field for a reverse-time cycle, so ``M_m ⋯ M_1`` is the monodromy at ``x0s[0]`` of
    the attracting flow.
    """
    sign = -1.0 if cycle.reverse_time else 1.0
    duration = sign * cycle.period / cycle.m
    fwd = cycle.config.with_adjoint(dfx.ForwardMode())
    flow = lambda x: cycle.ode.flow_to(duration, x, config=fwd)  # noqa: E731
    return jax.vmap(jax.jacfwd(flow))(cycle.x0s)


def _frame_from(q1: Float[Array, " dim"]) -> Float[Array, "dim dim"]:
    """An orthogonal matrix whose first column is the unit vector `q1`: the
    Householder reflector mapping ``e_1`` to `q1`, or the identity if ``q1 = e_1``."""
    dim = q1.shape[0]
    e1 = jnp.zeros(dim).at[0].set(1.0)
    v = e1 - q1
    vv = jnp.dot(v, v)
    safe = jnp.where(vv > 0, vv, 1.0)
    H = jnp.eye(dim) - 2.0 * jnp.outer(v, v) / safe
    return jnp.where(vv > 0, H, jnp.eye(dim))


def _qr_positive(A: Float[Array, "dim dim"]) -> tuple[Array, Array]:
    """``A = Q R`` with the diagonal of ``R`` positive, the convention of Dieci et al.
    1997, Eq. 3.3: the sign of row ``i`` of ``R`` and column ``i`` of ``Q`` are flipped
    together, which keeps the factorization exact."""
    Q, R = jnp.linalg.qr(A)
    s = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)
    return Q * s[None, :], R * s[:, None]


def periodic_schur(
    jacobians: Float[Array, "m dim dim"], q1: Float[ArrayLike, " dim"]
) -> PeriodicSchur:
    """The periodic Schur decomposition of ``M_m ⋯ M_1`` by the forward QR chain
    ``M_i Q_i = Q_{i+1} R_i`` with positive diagonals, seeded with `q1`.

    The chain is the discrete QR method for Lyapunov exponents over one period
    (Dieci et al. 1997, §3); the seed is one Schur vector known in advance, so in the
    plane no periodic QR iteration remains (ADR-0015 §1).

    **Arguments:**

    - `jacobians`: The factors, shape ``(m, dim, dim)``; ``jacobians[i]`` maps mesh
        point ``i`` to ``i + 1``.
    - `q1`: A unit vector that is an eigenvector of the product for its dominant
        eigenvalue: the flow direction at mesh point 0 for an attracting cycle.

    **Returns:**

    A `PeriodicSchur` whose `multipliers` are the eigenvalues of the product.

    **Raises:**

    `NotImplementedError` for ``dim > 2``: the seed fixes one Schur vector only, and the
    transverse block would need the periodic QR iteration (SLICOT ``MB03WD``, in Python
    through ``slycot.math.mb03wd``).
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


def monodromy(jacobians: Float[Array, "m dim dim"]) -> Float[Array, "dim dim"]:
    """The ordered product ``M_m ⋯ M_1`` of the segment Jacobians `jacobians`: the
    monodromy at mesh point 0. For square factors every association costs the same
    and rounds the same; the loss of accuracy is in the product itself, not in its
    order."""
    return jnp.linalg.multi_dot(list(jacobians[::-1]))


def multipliers_from_monodromy(M: Float[Array, "dim dim"]) -> Float[Array, " dim"]:
    """The eigenvalues of the monodromy `M`, shape ``(dim,)``: the one closest to 1,
    the trivial multiplier, first, then the others by modulus descending.

    Real parts are returned: the planar multipliers are real, and a complex pair in
    higher dimension would be reported by its real part. Rounding limits a multiplier
    ``μ`` to a relative error ``ε‖M‖/|μ|`` (Lust 2001, Eq. 10).
    """
    ev = jnp.linalg.eigvals(M)
    trivial = jnp.argmin(jnp.abs(ev - 1.0))
    rest = jnp.delete(ev, trivial, assume_unique_indices=True)
    rest = rest[jnp.argsort(-jnp.abs(rest))]
    return jnp.real(jnp.concatenate([ev[trivial][None], rest]))


def return_map_multiplier(
    pmap: PoincareMap, x: Float[ArrayLike, " dim"]
) -> Float[Array, ""]:
    """The nontrivial multiplier from the Jacobian of the return map `pmap` at its
    fixed point `x`: the largest eigenvalue of ``jax.jacfwd(P)(x)``, whose other
    eigenvalue is 0, along the flow.

    A single-section diagnostic. Both eigenvalues of ``DP`` are small, so its floor is
    the solver's absolute error on ``DP``, about ``1e-9`` at the shooting tolerances:
    fine for a moderately attracting cycle, meaningless for a ``μ`` below that. The
    trace of ``DP`` is not used, because it leaks the trivial multiplier's error onto
    ``μ`` (ADR-0015 §1).
    """
    DP = jax.jacfwd(lambda u: pmap(u)[0])(jnp.asarray(x, dtype=float))
    ev = jnp.linalg.eigvals(DP)
    return jnp.real(ev[jnp.argmax(jnp.abs(ev))])


def divergence_integral(cycle: LimitCycle) -> Float[Array, ""]:
    r"""The divergence integral $\int_0^T \mathrm{div} f(\gamma(t))\thinspace dt$ along
    `cycle`, which equals $\log\mu$ for a planar orbit by Liouville's formula.

    It is computed by one augmented solve
    $(\dot x, \dot z) = (f, \mathrm{tr}\thinspace Df)$ at the cycle's tolerances, in
    the direction in which the orbit attracts; the integrand is the forward field's
    divergence either way.

    **Raises:**

    `ValueError` unless ``cycle.dim == 2``, since in higher dimension the integral is
    the sum of all exponents.
    """
    if cycle.dim != 2:
        raise ValueError("the Liouville route is defined for planar orbits only.")
    sign = -1.0 if cycle.reverse_time else 1.0
    dim, ode = cycle.dim, cycle.ode

    def rhs(t, u, args):
        x = u[:dim]
        div = jnp.trace(jax.jacfwd(lambda y: ode.rhs(t, y))(x))
        return jnp.concatenate([sign * ode.rhs(t, x), div[None]])

    ts = jnp.stack([jnp.zeros_like(cycle.period), cycle.period])
    u0 = jnp.concatenate([cycle.x0s[0], jnp.zeros(1)])
    ys = diffeqsolve(rhs, ts, u0, None, cycle.config).ys
    assert ys is not None
    return ys[-1, dim]


# ----------------------------------------------------------------- the driver ----
def _floquet_vectors(ps: PeriodicSchur) -> Float[Array, "m dim dim-1"]:
    """The right eigenvectors of the nontrivial multipliers at every mesh point, shape
    ``(m, dim, dim-1)``, from the cyclic products of the triangular factors: in the
    plane a 2 × 2 triangular eigenproblem in the ``Q[i]`` frame, mapped back to the
    ambient frame."""
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


def floquet_periodic_schur(cycle: LimitCycle) -> Floquet:
    r"""The Floquet data of `cycle` by the seeded periodic Schur chain on its shooting
    mesh: the method.

    The multipliers are accumulated as sums of logarithms, so a multiplier as small as
    $10^{-16}$ keeps its digits; the chain runs in the direction in which the orbit
    attracts and the result is reported in forward time.

    **Returns:**

    A `Floquet` with `closure` and, in the plane, `vectors` set.
    """
    f = cycle.ode.rhs(0.0, cycle.x0s[0])
    ps = periodic_schur(segment_jacobians(cycle), f / jnp.linalg.norm(f))
    log_mu = ps.log_multipliers * (-1.0 if cycle.reverse_time else 1.0)
    mult = ps.signs * jnp.exp(log_mu)
    return Floquet(
        multipliers=mult,
        exponents=log_mu / cycle.period,
        trivial_error=jnp.abs(mult[0] - 1.0),
        closure=ps.closure,
        vectors=_floquet_vectors(ps) if cycle.dim == 2 else None,
        method="periodic_schur",
    )


def floquet_monodromy(cycle: LimitCycle) -> Floquet:
    r"""The Floquet data of `cycle` from the eigenvalues of the explicit monodromy.

    Rounding limits a small multiplier $\mu$ to a relative error
    $\varepsilon\Vert M\Vert/\mu$ (Lust 2001, Eq. 10), about $10^{-4}$ for
    FitzHugh–Nagumo where $\mu \approx 1.7\cdot10^{-12}$, so this route is a
    cross-check on `floquet_periodic_schur`, not the method.

    **Returns:**

    A `Floquet` with `multipliers` sorted by modulus descending; `closure` and `vectors`
    are ``None``.
    """
    mult = multipliers_from_monodromy(monodromy(segment_jacobians(cycle)))
    if cycle.reverse_time:  # the reversed flow's {1, 1/μ}
        mult = 1.0 / mult
    return Floquet(
        multipliers=mult,
        exponents=jnp.log(jnp.abs(mult)) / cycle.period,
        trivial_error=jnp.abs(mult[0] - 1.0),
        closure=None,
        vectors=None,
        method="monodromy",
    )


def floquet_liouville(cycle: LimitCycle) -> Floquet:
    """The Floquet data of `cycle` from the planar divergence integral: the
    cross-check independent of all linear algebra, exact in the plane.

    **Returns:**

    A `Floquet` whose trivial multiplier is assumed exactly, so `trivial_error`,
    `closure` and `vectors` are ``None``.

    **Raises:**

    `ValueError` for a non-planar cycle.
    """
    log_mu = divergence_integral(cycle)
    zero = jnp.zeros_like(log_mu)
    return Floquet(
        multipliers=jnp.stack([jnp.ones_like(log_mu), jnp.exp(log_mu)]),
        exponents=jnp.stack([zero, log_mu]) / cycle.period,
        trivial_error=None,
        closure=None,
        vectors=None,
        method="liouville",
    )


def floquet(cycle: LimitCycle, *, method: FloquetMethod = "periodic_schur") -> Floquet:
    """The Floquet multipliers and exponents of `cycle`, in forward time, by the
    chosen route.

    **Arguments:**

    - `cycle`: The `LimitCycle`, from `find_limit_cycle`; its `ode`, `config` and
        `reverse_time` are used.
    - `method`: ``"periodic_schur"`` (the method), ``"monodromy"`` or
        ``"liouville"``; each is also callable directly as ``floquet_<method>``.

    **Returns:**

    A `Floquet`, in forward time whatever the direction the cycle was found in.

    **Raises:**

    `ValueError` for an unknown `method` or for ``"liouville"`` on a non-planar cycle;
    `NotImplementedError` for ``"periodic_schur"`` with ``cycle.dim > 2``.
    """
    routes = {
        "periodic_schur": floquet_periodic_schur,
        "monodromy": floquet_monodromy,
        "liouville": floquet_liouville,
    }
    if method not in routes:
        raise ValueError(
            f"unknown method {method!r}; use 'periodic_schur', 'monodromy' or "
            "'liouville'."
        )
    return routes[method](cycle)


__all__ = [
    "Floquet",
    "FloquetMethod",
    "PeriodicSchur",
    "divergence_integral",
    "floquet",
    "floquet_liouville",
    "floquet_monodromy",
    "floquet_periodic_schur",
    "monodromy",
    "multipliers_from_monodromy",
    "periodic_schur",
    "return_map_multiplier",
    "segment_jacobians",
]
