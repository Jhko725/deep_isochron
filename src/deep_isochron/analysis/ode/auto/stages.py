r"""The four stages of the isochron pipeline (``docs/design/isochrons.md`` §3).

Each stage is a function of a `Runner` and the previous stage's `AutoSolution`, and
returns an `AutoSolution`; the driver in ``foliation`` strings them together. The
boundary value problem is the rotating multi-segment set-up of Hannam, Krauskopf and
Osinga, as reproduced by Andrus Giraldo (``isochron2pbvp/``, his project in this
repository; the version read is commit ``3af99f2``), stated in one Fortran file in four
nested sizes selected by ``NDIM``.

**Stage 1, `periodic_orbit`** (``NDIM = 4``): O1's cycle cut at half its period into
two segments, the first starting on the section through phase 0, handed to AUTO as a
start solution.

**Stage 2, `adjoint_bundle`** (``NDIM = 8``): the adjoint equation
$\dot w = -\tau A^{\mathsf T}w$ on both segments; the trivial bundle $w = 0$ walked in
the multiplier from 1.2 to the branch point at 1, then the switch onto the bundle
normalized at the junction, whose rotation $Jw$ is the stable direction there.

**Stage 3, `rotate_phase`** (``NDIM = 8``): the junction's segment length continued to
$\tau_2 = (1 - \theta/2\pi)T$ for every requested phase $\theta$, which moves the base
point and its bundle together around the cycle.

**Stage 4, `isochron`** (``NDIM = 10``): a segment of ``n_periods`` periods appended,
ending on the stable direction at distance $\eta$ from the base point, and continued
in $\eta$ in both directions; its start traces the isochron of the base point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from ....utils.numerics import arclength
from ...errors import NotConvergedError
from ..limit_cycle import LimitCycle
from .codegen import Idx
from .results import Isochron, IsochronSettings
from .runner import AutoSolution, bvp, check_stage, Runner


Box = tuple[Float[np.ndarray, "2"], Float[np.ndarray, "2"]]
"""``(lower, upper)`` corners of the box that ends an isochron's halves."""


# -------------------------------------------------------------- stage 1 ----
def periodic_orbit(run: Runner, cycle: LimitCycle, n_points: int = 601) -> AutoSolution:
    r"""Stage 1: O1's cycle as two segments of half a period each, segment 1 starting
    at phase 0 on the cycle's section, as an AUTO solution.

    The cycle is integrated from ``x0s[0]`` over one period by O1's solver and split at
    $T/2$; one run with ``NDIM = 4``, ``NBC = 6`` and the free parameters ``seg2``,
    ``seg1``, ``period`` stores it on AUTO's mesh. That run has ``NMX = 1``, and AUTO
    stores its first point as the start interpolated onto its mesh, not
    Newton-corrected: the stored orbit is off by the interpolation error,
    $3.4\times 10^{-6}$ on Bautin's unit circle with `n_points` 601. Stage 2's first
    continuation step corrects it, to $7\times 10^{-12}$ there.

    **Arguments:**

    - `run`: The session's runner.
    - `cycle`: An attracting planar cycle from `find_limit_cycle`, with phase 0 at
        ``x0s[0]`` and `section` through it.
    - `n_points`: The number of points of each segment's start.

    **Returns:**

    The two-segment orbit, with ``period``, ``seg1``, ``seg2`` and the section's
    point and normal in its parameters.
    """
    idx, period = run.idx, float(cycle.period)
    times = jnp.linspace(0.0, period, 2 * n_points - 1)
    orbit = np.asarray(cycle.ode.flow(times, cycle.x0s[0], config=cycle.config).ys)
    seg1, seg2 = orbit[:n_points], orbit[n_points - 1 :]
    p, n = np.asarray(cycle.section.point), np.asarray(cycle.section.normal)
    par = dict(run.problem.pars)
    par |= {idx.period: period, idx.seg1: period / 2, idx.seg2: period / 2}
    par |= dict(zip(idx.section_point, p)) | dict(zip(idx.section_normal, n))
    start = AutoSolution(np.linspace(0.0, 1.0, n_points), np.hstack([seg1, seg2]), par)
    branch = run(start, ICP=[idx.seg2, idx.seg1, idx.period], NMX=1, **bvp(4, 6))
    return run.solution(branch)


# -------------------------------------------------------------- stage 2 ----
def adjoint_bundle(run: Runner, orbit: AutoSolution) -> AutoSolution:
    r"""Stage 2: the adjoint bundle of the orbit, normalized at the junction.

    The orbit gets four components $w$, zero at first, with the adjoint equation
    $\dot w = -\tau A^{\mathsf T}w$ on each segment and the multiplier ``mult``
    between them (``NDIM = 8``, ``NBC = 11``; free: ``mult``, ``norm``, ``seg1``,
    ``period``). The trivial bundle $w = 0$ is walked in ``mult`` from 1.2 down to the
    branch point at 1, where the bundle with multiplier 1 branches off; AUTO's branch
    switching (``ISW = -1``) follows it until ``norm`` $= \vert w(0)\vert^2$ reaches 1.
    Its rotation $Jw$ at the junction is the stable direction there (§3.1, D1).

    **Arguments:**

    - `run`: The session's runner.
    - `orbit`: Stage 1's orbit.

    **Returns:**

    The normalized bundle, with ``mult`` $= 1$ and ``norm`` $= 1$.

    **Raises:**

    `NotConvergedError` when the first branch point is not at multiplier 1, or the
    normalization is not reached.
    """
    idx = run.idx
    icp = [idx.mult, idx.norm, idx.seg1, idx.period]
    u = np.hstack([orbit.u, np.zeros_like(orbit.u)])  # w = 0 on both segments
    par = orbit.par | {idx.mult: 1.2, idx.norm: 0.0}
    trivial = run(
        AutoSolution(orbit.t, u, par), ICP=icp, DS=-0.1, DSMAX=0.1, NMX=100,
        SP=["BP1"], **bvp(8, 11),
    )  # fmt: skip
    branch_points = trivial("BP")
    found = [float(bp[f"PAR({idx.mult})"]) for bp in branch_points]
    check_stage(
        "adjoint bundle",
        bool(found) and abs(found[0] - 1) < 1e-6,
        f"no branch point at multiplier 1; found {found}.",
    )
    # The switch restarts from AUTO's own point, which carries the null vector that
    # ISW = -1 needs; an array start would lose it.
    switched = run(
        branch_points[0], ICP=icp, ISW=-1, UZSTOP={idx.norm: 1.0}, NMX=200,
        **bvp(8, 11),
    )  # fmt: skip
    bundle = run.solution(switched)
    check_stage(
        "adjoint bundle",
        abs(bundle.par[idx.norm] - 1) < 1e-6,
        "the bundle's normalization did not reach 1.",
    )
    return bundle


# -------------------------------------------------------------- stage 3 ----
def rotate_phase(
    run: Runner, bundle: AutoSolution, phases: Sequence[float]
) -> dict[float, AutoSolution]:
    r"""Stage 3: the bundle rotated to every requested phase.

    The junction's segment length ``seg2`` is continued in both directions over
    $[0, 1.01T]$ with ``mult``, ``seg1`` and ``period`` free, and a user point at
    $\tau_2 = (1 - \theta/2\pi)T$ for every requested phase $\theta$; at each user
    point the junction, and with it the base point ``U(3:4)`` and its bundle
    ``U(7:8)``, sits at that phase (§3.4). AUTO reports a user point where its function
    changes sign between two steps, never at the starting value, so the phase whose
    target is the bundle's own ``seg2`` (θ = π, for stage 1's split at $T/2$) takes
    the bundle itself.

    **Arguments:**

    - `run`: The session's runner.
    - `bundle`: Stage 2's normalized bundle.
    - `phases`: The requested phases, wrapped to $(-\pi, \pi]$.

    **Returns:**

    The rotated solution of every phase whose user point was found, by phase.
    """
    idx = run.idx
    period = bundle.par[idx.period]
    targets = {theta: (1 - (theta % (2 * math.pi)) / (2 * math.pi)) * period
               for theta in phases}  # fmt: skip
    targets = {theta: (period if tau == 0 else tau) for theta, tau in targets.items()}
    start = bundle.par[idx.seg2]
    found = {
        theta: bundle
        for theta, target in targets.items()
        if abs(target - start) < 1e-8 * period
    }
    for ds in (-0.01, 0.01):
        branch = run(
            bundle, ICP=[idx.seg2, idx.mult, idx.seg1, idx.period], SP=[], DS=ds,
            DSMAX=0.05, NMX=2000, NPR=0, UZR={idx.seg2: list(targets.values())},
            RL0=0.0, RL1=1.01 * period, **bvp(8, 11),
        )  # fmt: skip
        for solution in run.labeled(branch, "UZ"):
            tau = solution.par[idx.seg2]
            for theta, target in targets.items():
                if theta not in found and abs(tau - target) < 1e-8 * period:
                    found[theta] = solution
    return found


# -------------------------------------------------------------- stage 4 ----
def bounding_box(cycle: LimitCycle, scale: float, n_points: int = 512) -> Box:
    """The cycle's bounding box enlarged `scale` times about its center, which ends an
    isochron's halves (`IsochronSettings.box_scale`); called inside a session."""
    _, points = cycle.sample(n_points)
    points = np.asarray(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    return center - scale * half, center + scale * half


def _isochron_start(
    rotated: AutoSolution, cycle: LimitCycle, idx: Idx, n_periods: int
) -> AutoSolution:
    """The start of stage 4 on AUTO's stored grid (decision D2): ``U(1:8)`` as stored
    and ``U(9:10)`` the cycle from the base point at the same $t$, scaled to
    `n_periods` periods."""
    period = rotated.par[idx.period]
    base = rotated.u[0, 2:4]
    orbit = cycle.ode.flow(
        jnp.asarray(rotated.t * n_periods * period), jnp.asarray(base),
        config=cycle.config,
    )  # fmt: skip
    u = np.column_stack([rotated.u, np.asarray(orbit.ys)])
    par = rotated.par | {idx.copies: n_periods * period, idx.eta: 0.0}
    return AutoSolution(rotated.t, u, par)


def _stop_reason(
    end: AutoSolution, idx: Idx, settings: IsochronSettings, box: Box
) -> str:
    r"""Why a half stopped, from the last point of its run.

    A user point is a box stop when the traced point lies on an edge of `box`, and an
    $\eta$ stop otherwise. AUTO locates a user point only to its own tolerance, so the
    test is on the box, whose edges AUTO reaches to round-off; an $\eta$ stop was once
    classified by $\vert\eta\vert \ge$ ``eta_stop`` and a point at
    $\eta = -9.99999995\times 10^{-7}$ was reported as a box stop.

    **Raises:**

    `NotConvergedError` when AUTO stopped for any other reason, such as a failed
    Newton correction (``MX``).
    """
    if end.label == "UZ":
        point = np.asarray([end.par[i] for i in idx.point], dtype=float)
        scale = float(np.max(box[1] - box[0]))
        on_edge = np.isclose(point, box[0], rtol=0, atol=1e-9 * scale) | np.isclose(
            point, box[1], rtol=0, atol=1e-9 * scale
        )
        return "box" if on_edge.any() else "eta"
    if end.label == "EP" and end.step >= settings.max_steps:
        return "max_steps"
    raise NotConvergedError("isochron", f"AUTO stopped with label {end.label}.", None)


def isochron(
    run: Runner,
    rotated: AutoSolution,
    cycle: LimitCycle,
    phase: float,
    settings: IsochronSettings,
    box: Box,
) -> Isochron:
    r"""Stage 4: the isochron of the rotated solution's base point.

    A segment of ``settings.n_periods`` periods is appended (``NDIM = 10``,
    ``NBC = 13``), starting on the cycle at the base point and ending at the base point
    plus $\eta\,Jw$; one run with ``NMX = 1`` corrects that start at $\eta = 0$, and two
    runs continue it in $\eta$ with ``DS`` $= \pm$ ``settings.ds`` (free: ``eta``,
    ``mult``, ``seg1``, ``period``; printed: the base point and the traced point). The
    segment's start is the isochron. A half stops at $\vert\eta\vert =$
    ``settings.eta_stop``, when the traced point leaves `box`, or after
    ``settings.max_steps`` steps, and the `Isochron` records which (D7); folds and
    branch points are not detected (``SP = [LP0, BP0]``).

    **Arguments:**

    - `run`: The session's runner.
    - `rotated`: Stage 3's solution at `phase`.
    - `cycle`: The cycle, for the segment's start by O1's solver.
    - `phase`: The phase of the base point, recorded on the result.
    - `settings`: How the isochron is traced.
    - `box`: The box that ends a half, from `bounding_box`.

    **Returns:**

    The `Isochron`, its points ordered along the curve through the base point, with
    arclength 0 there.

    **Raises:**

    `NotConvergedError` when a half stops for a reason other than the three recorded.
    """
    idx = run.idx
    start = _isochron_start(rotated, cycle, idx, settings.n_periods)
    icp = [idx.eta, idx.mult, idx.seg1, idx.period, *idx.base, *idx.point]
    common = dict(ICP=icp, NTST=settings.mesh_intervals, **bvp(10, 13))
    setup = run(start, NMX=1, **common)
    corrected = setup()[-1]  # AUTO's own point, which restarts the two halves
    base = run.solution(setup).par
    lo, hi = box
    stop = {
        idx.eta: [settings.eta_stop, -settings.eta_stop],
        idx.point[0]: [float(lo[0]), float(hi[0])],
        idx.point[1]: [float(lo[1]), float(hi[1])],
    }
    halves = []
    for ds in (-settings.ds, settings.ds):
        branch = run(
            corrected, DS=ds, DSMAX=10.0, NMX=settings.max_steps, NPR=200,
            UZSTOP=stop, SP=["LP0", "BP0"], **common,
        )  # fmt: skip
        points = np.column_stack(
            [branch[f"PAR({idx.point[0]})"], branch[f"PAR({idx.point[1]})"]]
        )
        eta = np.asarray(branch[f"PAR({idx.eta})"], dtype=float)
        end = run.solution(branch)
        halves.append((points, eta, _stop_reason(end, idx, settings, box)))
    (p1, e1, s1), (p2, e2, s2) = halves
    points = np.concatenate([p1[::-1], p2[1:]])
    eta = np.concatenate([e1[::-1], e2[1:]])
    s = np.array(arclength(points))  # a copy; a JAX array is read-only
    s -= s[len(p1) - 1]  # zero at the base point
    base_point = np.asarray([base[i] for i in idx.base], dtype=float)
    return Isochron(phase, base_point, points, eta, s, (s1, s2))


__all__ = [
    "Box",
    "adjoint_bundle",
    "bounding_box",
    "isochron",
    "periodic_orbit",
    "rotate_phase",
]
