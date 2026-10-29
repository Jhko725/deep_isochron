r"""``Cycle``: a planar limit cycle as a closed curve parameterized by phase.

The phase $\varphi \in [0, 1)$ is a fraction of the period (ADR-0020, decision 1),
and the curve is a truncated Fourier series in the angle $2\pi\varphi$,
$\gamma(\varphi) = c + \sum_{k=1}^{K} a_k \cos 2\pi k\varphi + b_k \sin 2\pi k\varphi$,
with the convention that $\varphi$ advances uniformly in time along the cycle,
$\varphi(t) = \varphi_0 + t/T \pmod 1$. The factor $2\pi$ is written in `__call__`,
the two derivatives that the Newton step of `nearest_phase` uses, and `from_points`:
the one seam between the phase and the angle of the series (ADR-0020, decision 2). On
the cycle, $\varphi$ is then the asymptotic phase up to a constant, so a point of the
cycle and its phase are interchangeable, and a learned phase can be compared against
``nearest_phase``. Revzen & Guckenheimer 2008 and 2012 model the orbit the same way,
as "an order 11 Fourier series model of the orbit".

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

from ..utils.numerics import wrap_phase


@dataclass(frozen=True)
class Cycle:
    """A closed planar curve ``γ(φ)`` with the time ``period`` per revolution.

    The phase ``φ`` is a fraction of the period in ``[0, 1)``, and ``γ`` is a Fourier
    series in the angle ``2πφ``.

    **Attributes:**

    - `center`: The constant term ``c``, shape ``(2,)``; also the center the phase is
        defined about, which the curve winds around once.
    - `cos`, `sin`: The Fourier coefficients ``a_k`` and ``b_k`` for ``k = 1..K``,
        shape ``(K, 2)`` each.
    - `period`: The time for ``φ`` to advance by 1.
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
        """The angular frequency ``2π / period`` of the series in ``2πφ``."""
        return 2.0 * np.pi / self.period

    def __call__(self, phase: Float[np.ndarray, " *n"]) -> Float[np.ndarray, "*n 2"]:
        """The points ``γ(φ)`` at the phases `phase` in turns, of any shape; the output
        has one trailing axis of length 2."""
        angle = 2.0 * np.pi * np.asarray(phase, dtype=float)[..., None]  # (..., 1)
        k = np.arange(1, self.harmonics + 1, dtype=float)  # (K,)
        kphi = angle * k  # (..., K)
        return self.center + np.cos(kphi) @ self.cos + np.sin(kphi) @ self.sin

    def derivative(self, phase: Float[np.ndarray, " *n"]) -> Float[np.ndarray, "*n 2"]:
        """The tangent ``dγ/dφ`` at the phases `phase`, shaped as `__call__`.

        The derivative is with respect to the phase in turns, so it is ``2π`` times
        the derivative of the series in its angle ``2πφ``.
        """
        angle = 2.0 * np.pi * np.asarray(phase, dtype=float)[..., None]
        k = np.arange(1, self.harmonics + 1, dtype=float)
        kphi = angle * k
        dk = 2.0 * np.pi * k  # (K,): d(kθ)/dφ with θ = 2πφ
        return (-np.sin(kphi) * dk) @ self.cos + (np.cos(kphi) * dk) @ self.sin

    def phases(self, num: int | None = None) -> Float[np.ndarray, " n"]:
        """`num` phases uniform on ``[0, 1)``; `resolution` of them by default."""
        return np.linspace(0.0, 1.0, num or self.resolution, endpoint=False)

    def points(self, num: int | None = None) -> Float[np.ndarray, "n 2"]:
        """The curve sampled at ``phases(num)``."""
        return self(self.phases(num))

    # --------------------------------------------------------- point queries -----
    def nearest_phase(self, x: Float[np.ndarray, "*n 2"]) -> Float[np.ndarray, " *n"]:
        """The phase of the curve point nearest to each point of `x`, by the dense
        sampling and then one Newton step on ``d/dφ |γ(φ) − x|²``, taken in turns with
        `derivative` and its second derivative.

        **Arguments:**

        - `x`: The query points, shape ``(..., 2)``.

        **Returns:**

        The phases in ``[0, 1)``, shape ``(...)``.
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
        return wrap_phase(phi - step)

    def _second_derivative(self, phase):
        """``d²γ/dφ²`` at the phases `phase` in turns, shaped as `__call__`; the
        derivative of `derivative`."""
        angle = 2.0 * np.pi * np.asarray(phase, dtype=float)[..., None]
        k = np.arange(1, self.harmonics + 1, dtype=float)
        kphi = angle * k
        dk2 = (2.0 * np.pi * k) ** 2
        return (-np.cos(kphi) * dk2) @ self.cos + (-np.sin(kphi) * dk2) @ self.sin

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

        - `phases`: The phases in turns, shape ``(n,)``, in any order and spacing;
            repeated phases are allowed, since pooled trajectories are the normal
            input. The series is fitted at the angles ``2π phases``.
        - `points`: The points, shape ``(n, 2)``.
        - `period`: The period to record.
        - `harmonics`: The Fourier order ``K``.
        - `note`: Free text recorded on the result.

        **Returns:**

        The `Cycle`, with `residual` the RMS misfit.

        **Raises:**

        `ValueError` for mismatched shapes, or fewer than ``2K + 1`` points.
        """
        angle = 2.0 * np.pi * np.asarray(phases, dtype=float)
        pts = np.asarray(points, dtype=float)
        if angle.ndim != 1 or pts.shape != (angle.shape[0], 2):
            raise ValueError("phases must be (n,) and points (n, 2).")
        if pts.shape[0] < 2 * harmonics + 1:
            raise ValueError(
                f"{pts.shape[0]} points cannot determine {2 * harmonics + 1} "
                "coefficients; lower harmonics."
            )
        k = np.arange(1, harmonics + 1, dtype=float)
        design = np.concatenate(
            [
                np.ones((angle.shape[0], 1)),
                np.cos(angle[:, None] * k),
                np.sin(angle[:, None] * k),
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
        ``γ(φ) = c + r (cos 2πφ, sin 2πφ)``: the normal forms' cycle in their own
        coordinates."""
        return cls(
            center=np.asarray(center, dtype=float),
            cos=np.array([[radius, 0.0]]),
            sin=np.array([[0.0, radius]]),
            period=float(period),
            note="circle",
        )
