r"""``estimate_cycle``: the limit cycle from sampled trajectories (roadmap E1).

The estimate works on trajectory tails and needs no notion of settling time: the last
``tail_periods`` revolutions of each trajectory are taken to lie on the cycle, and
the estimate checks that assumption afterwards (``CycleEstimate.converged``). Noise
is handled by averaging rather than filtering: a robust center, interpolated crossing
times for the period samples, a density estimate for the phase, least squares for the
curve.

The steps, all planar, are the center (the median of the second half of every
trajectory); the protophase, the unwrapped polar angle about the center, oriented to
increase with time; the tail, the samples of the last ``tail_periods`` revolutions,
with one period sample per revolution from the crossing times of the multiples of
$2\pi$; the phase, by the protophase-to-phase transformation of Kralemann et al. 2008
(``phase_from_protophase``), with the density coefficients averaged in time over
exactly ``tail_periods`` revolutions (``density_coefficients``); the period, from the
slope of the unwrapped phase against time by least squares over every tail sample;
the curve, ``Cycle.from_points``; and the self-consistency check, that the earlier
revolutions fit the curve of the last revolution as well as the last one does.

``docs/design/analysis.md`` §1 has the literature and the reason for each step;
ADR-0012 the layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import ArrayLike, Complex, Float

from ..cycle import Cycle


@dataclass(frozen=True)
class CycleEstimate:
    """The result of ``estimate_cycle``: the ``Cycle`` and how it was arrived at.

    **Attributes:**

    - `cycle`: The fitted curve, from the whole tail.
    - `winding`: ``+1`` when the protophase increases with time (counterclockwise),
        ``−1`` otherwise.
    - `period_samples`: One period estimate per completed tail revolution per
        trajectory, pooled, shape ``(m,)``.
    - `revolutions`: The revolutions each trajectory completes about the center,
        shape ``(N,)``.
    - `residual_last`: The RMS misfit of the last revolution of every trajectory to
        the curve fitted to those samples alone: the noise floor.
    - `residual_earlier`: The RMS misfit of the earlier tail revolutions to that same
        last-revolution curve.
    - `converged`: ``residual_earlier <= tolerance · residual_last``: the earlier
        revolutions lie on the curve the last one traces, so the tail is on the
        cycle. ``False`` says the tail still contains transient; shorten
        ``tail_periods`` or lengthen the trajectories. Trivially ``True`` for
        ``tail_periods == 1``.
    """

    cycle: Cycle
    winding: int
    period_samples: Float[np.ndarray, " m"]
    revolutions: Float[np.ndarray, " N"]
    residual_last: float
    residual_earlier: float
    converged: bool

    @property
    def period(self) -> float:
        """The period of `cycle`."""
        return self.cycle.period


def _as_batch(ys: ArrayLike) -> np.ndarray:
    """`ys` as a float array of shape ``(N, T, 2)``, with one trajectory promoted."""
    ys = np.asarray(ys, dtype=float)
    if ys.ndim == 2:
        ys = ys[None]
    if ys.ndim != 3 or ys.shape[-1] != 2:
        raise ValueError(f"trajectories must be (T, 2) or (N, T, 2); got {ys.shape}.")
    return ys


def orientation(points: Float[ArrayLike, "n 2"]) -> int:
    """``+1`` for a counterclockwise closed curve of `points`, by its signed area,
    ``−1`` for a clockwise one."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return 1 if area > 0 else -1


def protophase(
    ys: Float[np.ndarray, "N T 2"], center: Float[np.ndarray, " 2"]
) -> Float[np.ndarray, "N T"]:
    """The unwrapped polar angle about `center` along each trajectory of `ys`."""
    rel = ys - center
    return np.unwrap(np.arctan2(rel[..., 1], rel[..., 0]), axis=-1)


def _crossing_times(t: np.ndarray, psi: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """The times at which `psi`, monotone on average, first exceeds each of
    `levels`, by linear interpolation between the bracketing samples."""
    out = np.empty(len(levels))
    for j, level in enumerate(levels):
        idx = np.argmax(psi >= level)  # first index at or above the level
        if idx == 0:
            out[j] = t[0]
            continue
        p0, p1 = psi[idx - 1], psi[idx]
        w = 0.0 if p1 == p0 else (level - p0) / (p1 - p0)
        out[j] = t[idx - 1] + w * (t[idx] - t[idx - 1])
    return out


def density_coefficients(
    psi: Float[np.ndarray, " n"],
    harmonics: int,
    t: Float[np.ndarray, " n"] | None = None,
    window: tuple[float, float] | None = None,
) -> Complex[np.ndarray, " K"]:
    """The Fourier coefficients ``F_k = ⟨e^{−ikψ}⟩``, ``k = 1..K``, of the
    protophase's density in time.

    Without `t` this is the sample mean, the estimator of Kralemann et al. 2008. With
    `t` it is the time average by the trapezoid rule over `window`, with the ends
    outside the samples filled by linear interpolation of ``ψ``, so that an integer
    number of revolutions can be averaged exactly even when the samples do not start
    and end on a revolution boundary; a mismatch of one sample at the window's edges
    would bias every ``F_k`` by ``1/n``.

    **Arguments:**

    - `psi`: The protophase samples, shape ``(n,)``.
    - `harmonics`: ``K``.
    - `t`: The sample times, increasing, shape ``(n,)``; ``None`` for the sample mean.
    - `window`: ``(t_start, t_end)``; the whole of `t` when ``None``.

    **Returns:** The complex coefficients, shape ``(K,)``.
    """
    k = np.arange(1, harmonics + 1)
    if t is None:
        return np.mean(np.exp(-1j * np.outer(psi, k)), axis=0)
    t_start, t_end = window if window is not None else (t[0], t[-1])
    tt = np.concatenate([[t_start], t, [t_end]])
    pp = np.concatenate([[np.interp(t_start, t, psi)], psi, [np.interp(t_end, t, psi)]])
    keep = (tt >= t_start) & (tt <= t_end)
    tt, pp = tt[keep], pp[keep]
    values = np.exp(-1j * np.outer(pp, k))  # (n+2, K)
    dt = np.diff(tt)[:, None]
    return np.sum(0.5 * dt * (values[1:] + values[:-1]), axis=0) / (t_end - t_start)


def phase_from_protophase(
    psi: Float[np.ndarray, " n"],
    harmonics: int | None = None,
    *,
    coefficients: Complex[np.ndarray, " K"] | None = None,
) -> tuple[Float[np.ndarray, " n"], Complex[np.ndarray, " K"]]:
    """The protophase-to-phase transformation of Kralemann et al. 2008,
    ``φ = ψ + 2 Re Σ_k F_k (e^{ikψ} − 1) / (ik)``: the cumulative density of ``ψ``
    rescaled to ``2π``, so that ``φ`` is uniform in time.

    The correction is ``2π``-periodic in ``ψ``, so the same formula transforms an
    unwrapped protophase.

    **Arguments:**

    - `psi`: The protophase samples, shape ``(n,)``.
    - `harmonics`: ``K``, when the coefficients are estimated from `psi` itself.
    - `coefficients`: The coefficients ``F``, when already known.

    **Returns:** ``(φ, F)``.

    **Raises:** `ValueError` when neither `harmonics` nor `coefficients` is given.
    """
    if coefficients is None:
        if harmonics is None:
            raise ValueError("give harmonics or coefficients.")
        coefficients = density_coefficients(np.mod(psi, 2.0 * np.pi), harmonics)
    k = np.arange(1, coefficients.shape[0] + 1)
    correction = np.exp(1j * np.outer(psi, k)) - 1.0  # (n, K)
    phi = psi + 2.0 * np.real(correction @ (coefficients / (1j * k)))
    return phi, coefficients


def estimate_cycle(
    ys: Float[ArrayLike, "*N T 2"],
    ts: Float[ArrayLike, " T"],
    *,
    tail_periods: float = 3.0,
    harmonics: int = 11,
    density_harmonics: int | None = None,
    tolerance: float = 2.0,
) -> CycleEstimate:
    """The limit cycle from the last `tail_periods` revolutions of each trajectory.

    **Arguments:**

    - `ys`: One trajectory, shape ``(T, 2)``, or a collection, shape ``(N, T, 2)``.
    - `ts`: The common time grid, shape ``(T,)``.
    - `tail_periods`: The number of revolutions taken to lie on the cycle, at least 1.
    - `harmonics`: The Fourier order of the curve; 11 after Revzen & Guckenheimer
        2008, while the FitzHugh–Nagumo relaxation cycle needs about 24 for a
        residual below 1e-2.
    - `density_harmonics`: The order of the protophase-density expansion for the
        phase; ``2 harmonics + 2`` by default. It is cheap, and its truncation is the
        phase's error: the density of a strongly eccentric loop needs more terms than
        its curve.
    - `tolerance`: The factor within which the earlier tail revolutions must fit the
        last revolution's curve for `converged`.

    **Returns:** The `CycleEstimate`. Its `cycle` is always the fit to the whole tail,
    which averages the most; when `converged` is ``False`` it is biased toward the
    transient and `tail_periods` should be reduced.

    **Raises:** `ValueError` for mismatched shapes, ``tail_periods < 1``, or a
    trajectory that completes fewer than `tail_periods` revolutions.
    """
    ys = _as_batch(ys)
    t = np.asarray(ts, dtype=float)
    if t.shape != (ys.shape[1],):
        raise ValueError("ts must match the trajectories' time axis.")
    if tail_periods < 1:
        raise ValueError("tail_periods must be at least 1.")
    N, T, _ = ys.shape

    # 1. center from the second halves (robust to the transient when it is shorter)
    center = np.median(ys[:, T // 2 :].reshape(-1, 2), axis=0)

    # 2. protophase, oriented to increase with time
    psi = protophase(ys, center)
    total = psi[:, -1] - psi[:, 0]
    winding = 1 if np.median(total) >= 0 else -1
    psi = winding * psi
    revolutions = (psi[:, -1] - psi[:, 0]) / (2.0 * np.pi)
    if np.any(revolutions < tail_periods):
        raise ValueError(
            f"every trajectory needs at least {tail_periods} revolutions about the "
            f"center; the shortest completes {revolutions.min():.2f}."
        )

    # 3. tail samples; one period sample per completed revolution from the crossing
    #    times of 2π-levels; the density of ψ averaged over exactly ``tail_periods``
    #    revolutions per trajectory (pooled, weighted by duration)
    K_density = density_harmonics or 2 * harmonics + 2
    periods: list[np.ndarray] = []
    tail_psi_parts: list[np.ndarray] = []
    tail_t_parts: list[np.ndarray] = []
    tail_pts_parts: list[np.ndarray] = []
    last_parts: list[np.ndarray] = []
    F = np.zeros(K_density, dtype=complex)
    duration = 0.0
    n_levels = int(np.floor(tail_periods))
    for i in range(N):
        end = psi[i, -1]
        levels = end - 2.0 * np.pi * np.arange(n_levels, -1, -1)  # ascending to end
        times = _crossing_times(t, psi[i], levels)
        periods.append(np.diff(times))
        tail_start = end - 2.0 * np.pi * tail_periods
        t_start = _crossing_times(t, psi[i], np.array([tail_start]))[0]
        in_tail = psi[i] > tail_start
        tail_psi_parts.append(psi[i, in_tail])
        tail_t_parts.append(t[in_tail])
        tail_pts_parts.append(ys[i, in_tail])
        last_parts.append(psi[i, in_tail] > end - 2.0 * np.pi)
        span = t[-1] - t_start
        F += span * density_coefficients(psi[i], K_density, t, (t_start, t[-1]))
        duration += span
    F /= duration
    period_samples = np.concatenate(periods)
    tail_psi = np.concatenate(tail_psi_parts)
    tail_pts = np.concatenate(tail_pts_parts)
    last_mask = np.concatenate(last_parts)

    # 4. protophase -> phase (uniform in time on the cycle); the correction is
    #    2π-periodic in ψ, so the same formula gives the unwrapped φ
    phi_unwrapped, _ = phase_from_protophase(tail_psi, coefficients=F)
    phi = np.mod(phi_unwrapped, 2.0 * np.pi)

    # 5. the period: slope of the unwrapped phase against time over the tail, pooled
    #    over trajectories with one intercept each (every sample contributes, so
    #    measurement noise averages out; the crossing-interval samples are kept as a
    #    per-revolution diagnostic)
    num = den = 0.0
    start = 0
    for part_t in tail_t_parts:
        n_i = len(part_t)
        dt = part_t - part_t.mean()
        dphi = phi_unwrapped[start : start + n_i]
        num += float(np.dot(dt, dphi - dphi.mean()))
        den += float(np.dot(dt, dt))
        start += n_i
    period = 2.0 * np.pi * den / num

    # 6. the curve from the whole tail
    cycle = Cycle.from_points(
        phi, tail_pts, period, harmonics=harmonics, note="estimate_cycle (tails)"
    )

    # 7. self-consistency: the curve fitted to the last revolution alone is the
    #    reference; the earlier tail revolutions must fit it as well as the last does
    last = Cycle.from_points(phi[last_mask], tail_pts[last_mask], period, harmonics)
    misfit = np.sum((last(phi) - tail_pts) ** 2, axis=-1)
    residual_last = float(np.sqrt(np.mean(misfit[last_mask])))
    earlier = misfit[~last_mask]
    residual_earlier = float(np.sqrt(np.mean(earlier))) if earlier.size else 0.0
    converged = bool(residual_earlier <= tolerance * max(residual_last, 1e-300))
    return CycleEstimate(
        cycle=cycle,
        winding=winding,
        period_samples=period_samples,
        revolutions=revolutions,
        residual_last=residual_last,
        residual_earlier=residual_earlier,
        converged=converged,
    )
