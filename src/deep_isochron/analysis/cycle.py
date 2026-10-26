r"""``Cycle``: a planar limit cycle as a closed curve parameterized by phase.

The curve is a truncated Fourier series in the phase $\varphi \in [0, 2\pi)$,
$\gamma(\varphi) = c + \sum_{k=1}^{K} a_k \cos k\varphi + b_k \sin k\varphi$, with the
convention that $\varphi$ advances uniformly in time along the cycle,
$\varphi(t) = \varphi_0 + 2\pi t/T$. On the cycle, $\varphi$ is then the asymptotic
phase up to a constant, so a point of the cycle and its phase are interchangeable,
and a learned phase can be compared against ``nearest_phase``. Revzen & Guckenheimer
2008 and 2012 model the orbit the same way, as "an order 11 Fourier series model of
the orbit".

Both halves of ``analysis`` can produce a ``Cycle``: ``analysis.data`` from
trajectories, ``analysis.ode`` from a vector field. Distances and nearest phases are
computed against a dense sampling of the curve, ``resolution`` points, followed by one
Newton step; for the smooth curves of this project that is accurate to well below
every tolerance used here. ADR-0012 §1 records the object's role.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class Cycle:
    """A closed planar curve ``γ(φ)`` with the time ``period`` per revolution.

    **Attributes:**

    - `center`: The constant term ``c``, shape ``(2,)``; also the center the phase is
        defined about, which the curve winds around once.
    - `cos`, `sin`: The Fourier coefficients ``a_k`` and ``b_k`` for ``k = 1..K``,
        shape ``(K, 2)`` each.
    - `period`: The time for ``φ`` to advance by ``2π``.
    - `residual`: The RMS distance of the fitted samples to the curve; 0 for an
        exact construction.
    - `note`: How the cycle was obtained, free text.
    - `resolution`: The number of points of the dense sampling used by the point
        queries.

    **Raises:**

    `ValueError` at construction for shapes other than the above, fewer than one
    harmonic, or a non-positive period.
    """

    center: Float[np.ndarray, " 2"]
    cos: Float[np.ndarray, "K 2"]
    sin: Float[np.ndarray, "K 2"]
    period: float
    residual: float = 0.0
    note: str = ""
    resolution: int = field(default=2048, repr=False)

    def __post_init__(self):
        """Coerce the coefficients to float arrays and validate the shapes."""
        center = np.asarray(self.center, dtype=float)
        cos, sin = np.asarray(self.cos, dtype=float), np.asarray(self.sin, dtype=float)
        if center.shape != (2,) or cos.shape != sin.shape or cos.ndim != 2:
            raise ValueError("center must be (2,), cos and sin (K, 2).")
        if cos.shape[1] != 2 or cos.shape[0] < 1:
            raise ValueError("cos and sin must be (K, 2) with K >= 1.")
        if not self.period > 0:
            raise ValueError("period must be positive.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "cos", cos)
        object.__setattr__(self, "sin", sin)

    # ------------------------------------------------------------- the curve -----
    @property
    def harmonics(self) -> int:
        """The number of harmonics ``K``."""
        return self.cos.shape[0]

    @property
    def omega(self) -> float:
        """The angular frequency ``2π / period``."""
        return 2.0 * np.pi / self.period

    def __call__(self, phase: Float[np.ndarray, " *n"]) -> Float[np.ndarray, "*n 2"]:
        """The points ``γ(φ)`` at the phases `phase`, of any shape; the output has
        one trailing axis of length 2."""
        phi = np.asarray(phase, dtype=float)[..., None]  # (..., 1)
        k = np.arange(1, self.harmonics + 1, dtype=float)  # (K,)
        kphi = phi * k  # (..., K)
        return self.center + np.cos(kphi) @ self.cos + np.sin(kphi) @ self.sin

    def derivative(self, phase: Float[np.ndarray, " *n"]) -> Float[np.ndarray, "*n 2"]:
        """The tangent ``dγ/dφ`` at the phases `phase`, shaped as `__call__`."""
        phi = np.asarray(phase, dtype=float)[..., None]
        k = np.arange(1, self.harmonics + 1, dtype=float)
        kphi = phi * k
        return (-np.sin(kphi) * k) @ self.cos + (np.cos(kphi) * k) @ self.sin

    def phases(self, num: int | None = None) -> Float[np.ndarray, " n"]:
        """`num` phases uniform on ``[0, 2π)``; `resolution` of them by default."""
        return np.linspace(0.0, 2.0 * np.pi, num or self.resolution, endpoint=False)

    def points(self, num: int | None = None) -> Float[np.ndarray, "n 2"]:
        """The curve sampled at ``phases(num)``."""
        return self(self.phases(num))

    # --------------------------------------------------------- point queries -----
    def nearest_phase(self, x: Float[np.ndarray, "*n 2"]) -> Float[np.ndarray, " *n"]:
        """The phase of the curve point nearest to each point of `x`, by the dense
        sampling and then one Newton step on ``d/dφ |γ(φ) − x|²``.

        **Arguments:**

        - `x`: The query points, shape ``(..., 2)``.

        **Returns:**

        The phases in ``[0, 2π)``, shape ``(...)``.
        """
        x = np.asarray(x, dtype=float)
        pts = self.points()  # (R, 2)
        d2 = np.sum((x[..., None, :] - pts) ** 2, axis=-1)  # (..., R)
        idx = np.argmin(d2, axis=-1)
        phi = self.phases()[idx]
        # Newton on f(φ) = (γ(φ) − x)·γ'(φ) = 0
        g, dg = self(phi), self.derivative(phi)
        ddg = self._second_derivative(phi)
        f = np.sum((g - x) * dg, axis=-1)
        df = np.sum(dg * dg, axis=-1) + np.sum((g - x) * ddg, axis=-1)
        step = np.where(np.abs(df) > 1e-12, f / np.where(df == 0, 1.0, df), 0.0)
        return np.mod(phi - step, 2.0 * np.pi)

    def _second_derivative(self, phase):
        """``d²γ/dφ²`` at the phases `phase`, shaped as `__call__`."""
        phi = np.asarray(phase, dtype=float)[..., None]
        k = np.arange(1, self.harmonics + 1, dtype=float)
        kphi = phi * k
        return (-np.cos(kphi) * k**2) @ self.cos + (-np.sin(kphi) * k**2) @ self.sin

    def nearest_point(self, x: Float[np.ndarray, "*n 2"]) -> Float[np.ndarray, "*n 2"]:
        """The curve point nearest to each point of `x`, ``γ(nearest_phase(x))``."""
        return self(self.nearest_phase(x))

    def distance(self, x: Float[np.ndarray, "*n 2"]) -> Float[np.ndarray, " *n"]:
        """The Euclidean distance from each point of `x` to the curve."""
        x = np.asarray(x, dtype=float)
        return np.linalg.norm(x - self.nearest_point(x), axis=-1)

    # --------------------------------------------------------- constructors -----
    @classmethod
    def from_points(
        cls,
        phases: Float[np.ndarray, " n"],
        points: Float[np.ndarray, "n 2"],
        period: float,
        harmonics: int = 11,
        note: str = "",
    ) -> Cycle:
        """The least-squares Fourier fit to `points` observed at `phases`.

        **Arguments:**

        - `phases`: The phases, shape ``(n,)``, in any order and spacing; repeated
            phases are allowed, since pooled trajectories are the normal input.
        - `points`: The points, shape ``(n, 2)``.
        - `period`: The period to record.
        - `harmonics`: The Fourier order ``K``.
        - `note`: Free text recorded on the result.

        **Returns:**

        The `Cycle`, with `residual` the RMS misfit.

        **Raises:**

        `ValueError` for mismatched shapes, or fewer than ``2K + 1`` points.
        """
        phi = np.asarray(phases, dtype=float)
        pts = np.asarray(points, dtype=float)
        if phi.ndim != 1 or pts.shape != (phi.shape[0], 2):
            raise ValueError("phases must be (n,) and points (n, 2).")
        if pts.shape[0] < 2 * harmonics + 1:
            raise ValueError(
                f"{pts.shape[0]} points cannot determine {2 * harmonics + 1} "
                "coefficients; lower harmonics."
            )
        k = np.arange(1, harmonics + 1, dtype=float)
        design = np.concatenate(
            [
                np.ones((phi.shape[0], 1)),
                np.cos(phi[:, None] * k),
                np.sin(phi[:, None] * k),
            ],
            axis=1,
        )  # (n, 1 + 2K)
        coef, *_ = np.linalg.lstsq(design, pts, rcond=None)  # (1 + 2K, 2)
        fitted = design @ coef
        residual = float(np.sqrt(np.mean(np.sum((fitted - pts) ** 2, axis=-1))))
        return cls(
            center=coef[0],
            cos=coef[1 : 1 + harmonics],
            sin=coef[1 + harmonics :],
            period=float(period),
            residual=residual,
            note=note,
        )

    @classmethod
    def circle(cls, radius: float, period: float, center=(0.0, 0.0)) -> Cycle:
        """The circle of `radius` about `center` traversed counterclockwise,
        ``γ(φ) = c + r (cos φ, sin φ)``: the normal forms' cycle in their own
        coordinates."""
        return cls(
            center=np.asarray(center, dtype=float),
            cos=np.array([[radius, 0.0]]),
            sin=np.array([[0.0, radius]]),
            period=float(period),
            note="circle",
        )
