r"""Isochrons of a planar limit cycle by AUTO-07p continuation (roadmap O7).

``isochron_foliation(cycle, phases, settings)`` is the one entry point. It runs the
pipeline of ``docs/design/isochrons.md`` §6, a rotating multi-segment boundary value
problem after Hannam, Krauskopf and Osinga, as reproduced by Andrus Giraldo in
``src/Isochron2PBVP`` (removed after commit ``3af99f2``):

1. `_periodic_orbit` cuts O1's cycle at half its period into two segments, the first
   starting on the section through phase 0, and hands them to AUTO, which stores them
   on its mesh; the first step of stage 2 corrects them.
2. `_adjoint_bundle` adds the adjoint equation $\dot w = -\tau A^{\mathsf T}w$ on both
   segments, finds the branch point at multiplier 1 and switches onto the normalized
   bundle, whose rotation $Jw$ is the stable direction at the junction.
3. `_rotate_phase` continues the junction's segment length to every requested phase.
4. `_isochron` appends a segment of ``n_periods`` periods that ends on the stable
   direction at distance $\eta$ from the junction, and continues it in $\eta$ in both
   directions; its start traces the isochron of the junction.

Stages 1–3 run once in the calling process; stage 4 runs per phase in a pool of
``spawn`` processes. AUTO writes its files into the working directory, which belongs to
the whole process, so every stage runs inside a temporary directory that it enters and
leaves, and the module is not thread-safe. AUTO's output goes to this module's logger
at DEBUG.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import io
import json
import logging
import math
import multiprocessing
import os
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import ArrayLike, Float

from ....data.dataset import ENGINE
from ...errors import NotConvergedError
from ..limit_cycle import LimitCycle
from ._env import AutoHandle, load_auto
from .codegen import AutoProblem, write_problem


logger = logging.getLogger(__name__)

LOG_TAIL = 50
"""The number of lines of AUTO's output that a failed phase keeps."""

STOP_REASONS = ("eta", "box", "max_steps")
r"""Why a half of an isochron stopped: $\vert\eta\vert$ reached ``eta_stop``, the
traced point left the box, or the continuation ran out of steps."""


# ------------------------------------------------------------------- settings ----
@dataclass(frozen=True)
class IsochronSettings:
    r"""How the isochrons are traced.

    **Attributes:**

    - `n_periods`: The number of periods of the isochron segment; more periods reach
        further toward the phaseless set (``docs/design/isochrons.md`` §7).
    - `eta_stop`: The distance along the stable direction at which a half stops.
    - `box_scale`: A half also stops when the traced point leaves the cycle's
        bounding box enlarged `box_scale` times about its center.
    - `ntst`: AUTO's number of mesh intervals for the isochron runs; `None` means
        ``60 * n_periods``, so that the mesh per period does not change with
        `n_periods`.
    - `max_workers`: The number of processes tracing phases; `None` means
        ``os.process_cpu_count()``, the cores this process may use.
    - `max_steps`: The bound on the continuation steps of a half (AUTO's ``NMX``).
    - `ds`: The first continuation step in $\eta$ (AUTO's ``DS``).
    """

    n_periods: int = 5
    eta_stop: float = 1e-6
    box_scale: float = 2.5
    ntst: int | None = None
    max_workers: int | None = None
    max_steps: int = 8000
    ds: float = 1e-5

    def __post_init__(self):
        if self.n_periods < 1:
            raise ValueError("n_periods must be at least 1.")
        if not (self.eta_stop > 0 and self.box_scale > 1 and self.ds > 0):
            raise ValueError("eta_stop and ds must be positive, box_scale above 1.")
        if self.ntst is not None and self.ntst < 1:
            raise ValueError("ntst must be positive.")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")

    @property
    def mesh_intervals(self) -> int:
        """`ntst`, or ``60 * n_periods`` when it is `None`."""
        return 60 * self.n_periods if self.ntst is None else self.ntst

    @property
    def workers(self) -> int:
        """`max_workers`, or the cores this process may use when it is `None`."""
        return self.max_workers or os.process_cpu_count() or 1


# -------------------------------------------------------------------- results ----
@dataclass(frozen=True)
class Isochron:
    r"""One traced isochron, as host-side NumPy arrays.

    **Attributes:**

    - `phase`: The phase $\theta \in (-\pi, \pi]$ of the base point.
    - `base_point`: The point of the cycle at that phase, shape ``(2,)``.
    - `points`: The traced points, shape ``(n, 2)``, ordered along the curve from the
        end of one half, through the base point, to the end of the other.
    - `eta`: The distance along the stable direction that traced each point, shape
        ``(n,)``; negative on the first half.
    - `arclength`: The arclength of each point from the base point, shape ``(n,)``;
        negative on the first half.
    - `stop`: Why each half stopped, one of `STOP_REASONS` each.
    """

    phase: float
    base_point: Float[np.ndarray, "2"]
    points: Float[np.ndarray, "n 2"]
    eta: Float[np.ndarray, " n"]
    arclength: Float[np.ndarray, " n"]
    stop: tuple[str, str]


@dataclass(frozen=True)
class IsochronFailure:
    """A phase whose isochron could not be traced.

    **Attributes:**

    - `phase`: The requested phase.
    - `message`: What went wrong.
    - `log_tail`: The last `LOG_TAIL` lines of AUTO's output.
    - `constants`: The run constants of the failed run.
    """

    phase: float
    message: str
    log_tail: str
    constants: dict[str, Any]


@dataclass(frozen=True)
class IsochronFoliation:
    """The isochrons of a cycle at the requested phases.

    **Attributes:**

    - `cycle`: The `LimitCycle`, whose ``x0s[0]`` is phase 0.
    - `isochrons`: One `Isochron` per traced phase, in the order requested.
    - `failures`: One `IsochronFailure` per phase that could not be traced.
    - `settings`: The `IsochronSettings` used.
    - `provenance`: Where the result came from: the package's git state, the AUTO
        commit, the hash of the Fortran sources, the compiler and its flags.
    """

    cycle: LimitCycle
    isochrons: tuple[Isochron, ...]
    failures: tuple[IsochronFailure, ...]
    settings: IsochronSettings
    provenance: dict[str, str] = field(default_factory=dict)

    def dataset(self) -> xr.Dataset:
        """The foliation as one `xarray.Dataset`: the isochrons concatenated along the
        dimension ``point`` and delimited by ``offsets``, the cycle's mesh, period and
        section as data, and the system, settings, failures and provenance as JSON
        attributes."""
        lengths = [len(iso.points) for iso in self.isochrons]
        cat = lambda f, shape: (  # noqa: E731
            np.concatenate([f(iso) for iso in self.isochrons])
            if self.isochrons
            else np.zeros(shape)
        )
        cycle = self.cycle
        return xr.Dataset(
            {
                "points": (("point", "xy"), cat(lambda i: i.points, (0, 2))),
                "eta": ("point", cat(lambda i: i.eta, (0,))),
                "arclength": ("point", cat(lambda i: i.arclength, (0,))),
                "offsets": ("boundary", np.concatenate([[0], np.cumsum(lengths)])),
                "phase": ("isochron", [iso.phase for iso in self.isochrons]),
                "base_point": (
                    ("isochron", "xy"),
                    np.reshape([iso.base_point for iso in self.isochrons], (-1, 2)),
                ),
                "stop": (
                    ("isochron", "half"),
                    np.reshape([iso.stop for iso in self.isochrons], (-1, 2)),
                ),
                "cycle_x0s": (("mesh", "xy"), np.asarray(cycle.x0s)),
                "cycle_period": float(cycle.period),
                "cycle_residual": float(cycle.residual),
                "cycle_section_point": ("xy", np.asarray(cycle.section.point)),
                "cycle_section_normal": ("xy", np.asarray(cycle.section.normal)),
            },
            attrs={
                "system": json.dumps(
                    {"name": type(cycle.ode).__name__, "params": cycle.ode.params()}
                ),
                "settings": json.dumps(dataclasses.asdict(self.settings)),
                "failures": json.dumps(
                    [dataclasses.asdict(f) for f in self.failures], default=str
                ),
                "provenance": json.dumps(self.provenance),
            },
        )

    def save(self, path: str | Path) -> Path:
        """Write the foliation to the netCDF file `path` (decision 16) and return it.

        **Raises:**

        `FileExistsError` if `path` exists.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists; choose another path.")
        self.dataset().to_netcdf(path, engine=ENGINE)
        return path

    @classmethod
    def load(cls, path: str | Path, cycle: LimitCycle) -> IsochronFoliation:
        """Read a foliation that `save` wrote, attached to `cycle`.

        The file stores the cycle as data, not as a `LimitCycle`, which would need the
        system object and its solver. The caller passes the cycle, from
        `find_limit_cycle` and `reanchor` at the stored ``cycle_x0s[0]``, and it is
        checked against the file.

        **Arguments:**

        - `path`: The file.
        - `cycle`: The cycle the foliation was traced for.

        **Returns:**

        The `IsochronFoliation`.

        **Raises:**

        `ValueError` when `cycle` is of another system, other parameters, another
        period or another phase origin than the file records.
        """
        with xr.open_dataset(path, engine=ENGINE) as ds:
            ds = ds.load()
        system = json.loads(ds.attrs["system"])
        mismatches = []
        if system["name"] != type(cycle.ode).__name__:
            mismatches.append(f"system {system['name']} != {type(cycle.ode).__name__}")
        if not _same_params(system["params"], cycle.ode.params()):
            mismatches.append(f"parameters {system['params']} != {cycle.ode.params()}")
        if not math.isclose(
            float(ds["cycle_period"]), float(cycle.period), rel_tol=1e-8
        ):
            mismatches.append(
                f"period {float(ds['cycle_period'])} != {float(cycle.period)}"
            )
        origin = ds["cycle_x0s"].values[0]
        if not np.allclose(origin, np.asarray(cycle.x0s[0]), atol=1e-8):
            mismatches.append(f"phase origin {origin} != {np.asarray(cycle.x0s[0])}")
        if mismatches:
            raise ValueError(f"{path} was traced for another cycle: {mismatches}.")
        offsets = ds["offsets"].values
        isochrons = tuple(
            Isochron(
                phase=float(ds["phase"].values[i]),
                base_point=ds["base_point"].values[i],
                points=ds["points"].values[a:b],
                eta=ds["eta"].values[a:b],
                arclength=ds["arclength"].values[a:b],
                stop=(str(ds["stop"].values[i, 0]), str(ds["stop"].values[i, 1])),
            )
            for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:]))
        )
        failures = tuple(IsochronFailure(**f) for f in json.loads(ds.attrs["failures"]))
        return cls(
            cycle=cycle,
            isochrons=isochrons,
            failures=failures,
            settings=IsochronSettings(**json.loads(ds.attrs["settings"])),
            provenance=json.loads(ds.attrs["provenance"]),
        )


def _same_params(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Whether two parameter dicts agree, numbers to a relative ``1e-12``."""
    if stored.keys() != current.keys():
        return False
    return all(
        math.isclose(float(stored[k]), float(current[k]), rel_tol=1e-12) for k in stored
    )


# ----------------------------------------------------------- AUTO plumbing ----
class _Solution(NamedTuple):
    """An AUTO solution as plain arrays, so that it crosses process boundaries.

    **Attributes:**

    - `t`: The stored mesh of $[0, 1]$, shape ``(n,)``.
    - `u`: The state at those points, shape ``(n, ndim)``.
    - `par`: The parameters by slot.
    """

    t: np.ndarray
    u: np.ndarray
    par: dict[int, float]

    @classmethod
    def from_auto(cls, solution: Any, ndim: int, npar: int) -> _Solution:
        """Copy an AUTO solution's mesh, first `ndim` components and `npar`
        parameters."""
        t = np.asarray(solution["t"], dtype=float)
        u = np.column_stack([solution[f"U({i})"] for i in range(1, ndim + 1)])
        par = {i: float(solution[f"PAR({i})"]) for i in range(1, npar + 1)}
        return cls(t, u, par)

    def start(self) -> np.ndarray:
        """The array start solution AUTO's ``run`` takes: rows ``t, U(1), ...``."""
        return np.vstack([self.t, self.u.T])


@contextlib.contextmanager
def _workdir(ode, name: str) -> Generator[AutoProblem]:
    """A temporary directory holding the AUTO problem of `ode`, as the working
    directory for the duration of the block."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="deep_isochron_auto_") as tmp:
        problem = write_problem(ode, tmp, name=name)
        os.chdir(tmp)
        try:
            yield problem
        finally:
            os.chdir(previous)


@contextlib.contextmanager
def _one_thread() -> Generator[None]:
    """``OMP_NUM_THREADS=1`` for the duration of the block, restored afterward.

    AUTO compiles each equation file with ``-fopenmp`` and starts the executable with
    the process's environment, so without the variable it runs one thread per core.
    On a 24-core node, stages 1–3 of one FitzHugh–Nagumo phase ran on 24 threads for
    about 8.5 minutes; on two cores they take 82 s. A planar problem leaves each thread
    little work to share.
    """
    previous = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = previous


class _Runner:
    """Runs AUTO on one thread (`_one_thread`) with its output captured: logged at
    DEBUG and kept for failures."""

    def __init__(self, auto: AutoHandle, problem: AutoProblem):
        self.auto = auto
        self.problem = problem
        self.output: list[str] = []
        self.constants: dict[str, Any] = {}

    def __call__(self, start: Any, **constants: Any) -> Any:
        """One AUTO run from `start` with the run constants `constants`."""
        self.constants = constants
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), _one_thread():
                return self.auto.commands.run(
                    start,
                    e=self.problem.name,
                    c=self.problem.name,
                    runner=self.auto.runner,
                    **constants,
                )
        finally:
            text = buffer.getvalue()
            self.output.extend(text.splitlines())
            if text:
                logger.debug("AUTO %s:\n%s", self.problem.name, text)

    def tail(self) -> str:
        """The last `LOG_TAIL` lines of AUTO's output."""
        return "\n".join(self.output[-LOG_TAIL:])


def _bvp(ndim: int, nbc: int) -> dict[str, int]:
    """The run constants of the staged boundary value problem with `ndim` components
    and `nbc` boundary conditions. Every run passes them, because each run reloads the
    constants file and nothing carries over from the run before."""
    return {"IPS": 4, "NDIM": ndim, "NBC": nbc, "NINT": 0}


def _wrap_phase(theta: float) -> float:
    r"""`theta` wrapped to $(-\pi, \pi]$."""
    wrapped = math.remainder(theta, 2 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


def _check_phases(
    phases: Sequence[float] | Float[ArrayLike, " k"],
) -> tuple[float, ...]:
    r"""The requested phases wrapped to $(-\pi, \pi]$.

    **Raises:**

    `ValueError` naming any phase that is not finite or that repeats another after
    wrapping.
    """
    values = [float(p) for p in np.atleast_1d(np.asarray(phases, dtype=float))]
    bad = [p for p in values if not math.isfinite(p)]
    if bad:
        raise ValueError(f"phases must be finite; got {bad}.")
    wrapped = [_wrap_phase(p) for p in values]
    seen: dict[float, float] = {}
    repeats = []
    for original, w in zip(values, wrapped):
        key = round(w, 12)
        if key in seen:
            repeats.append((seen[key], original))
        seen.setdefault(key, original)
    if repeats:
        raise ValueError(f"phases repeat after wrapping to (-π, π]: {repeats}.")
    return tuple(wrapped)


# ------------------------------------------------------------------ the stages ----
def _periodic_orbit(run: _Runner, cycle: LimitCycle, n_points: int = 601) -> Any:
    r"""Stage 1: O1's cycle as two segments of half a period each, segment 1 starting
    at phase 0 on the cycle's section, as an AUTO solution (``NDIM = 4``,
    ``NBC = 6``; free: ``seg2``, ``seg1``, ``period``).

    The run has ``NMX = 1``, and AUTO stores its first point as the start interpolated
    onto its mesh, not Newton-corrected: the stored orbit is off by the interpolation
    error, $3.4\times 10^{-6}$ on Bautin's unit circle with `n_points` 601. Stage 2's
    first continuation step corrects it, to $7\times 10^{-12}$ there."""
    idx, period = run.problem.idx, float(cycle.period)
    with jax.enable_x64(True):
        times = jnp.linspace(0.0, period, 2 * n_points - 1)
        orbit = np.asarray(cycle.ode.flow(times, cycle.x0s[0], config=cycle.config).ys)
    seg1, seg2 = orbit[:n_points], orbit[n_points - 1 :]
    start = np.vstack([np.linspace(0.0, 1.0, n_points), seg1.T, seg2.T])
    p, n = np.asarray(cycle.section.point), np.asarray(cycle.section.normal)
    par = dict(run.problem.pars)
    par |= {idx.period: period, idx.seg1: period / 2, idx.seg2: period / 2}
    par |= dict(zip(idx.section_point, p)) | dict(zip(idx.section_normal, n))
    branch = run(
        start, PAR=par, ICP=[idx.seg2, idx.seg1, idx.period], NMX=1, **_bvp(4, 6),
    )  # fmt: skip
    return branch()[-1]


def _adjoint_bundle(run: _Runner, orbit: Any) -> Any:
    """Stage 2: the trivial bundle ``w = 0`` walked in the multiplier from 1.2 down to
    the branch point at 1, then the switch onto the bundle normalized to
    ``|w|² = 1`` at the junction (``NDIM = 8``, ``NBC = 11``; free: ``mult``,
    ``norm``, ``seg1``, ``period``).

    **Raises:**

    `NotConvergedError` when the first branch point is not at multiplier 1 or the
    normalization is not reached.
    """
    idx = run.problem.idx
    icp = [idx.mult, idx.norm, idx.seg1, idx.period]
    trivial = run(
        orbit, ICP=icp, PAR={idx.mult: 1.2, idx.norm: 0.0}, DS=-0.1, DSMAX=0.1,
        NMX=100, SP=["BP1"], **_bvp(8, 11),
    )  # fmt: skip
    branch_points = trivial("BP")
    if not branch_points or abs(branch_points[0][f"PAR({idx.mult})"] - 1) > 1e-6:
        found = [bp[f"PAR({idx.mult})"] for bp in branch_points]
        raise NotConvergedError(
            "adjoint bundle", f"no branch point at multiplier 1; found {found}.", None
        )
    bundle = run(
        branch_points[0], ICP=icp, ISW=-1, UZSTOP={idx.norm: 1.0}, NMX=200,
        **_bvp(8, 11),
    )()[-1]  # fmt: skip
    if abs(bundle[f"PAR({idx.norm})"] - 1) > 1e-6:
        raise NotConvergedError(
            "adjoint bundle", "the bundle's normalization did not reach 1.", None
        )
    return bundle


def _rotate_phase(
    run: _Runner, bundle: Any, phases: Sequence[float], npar: int
) -> dict[float, _Solution]:
    r"""Stage 3: the junction's segment length continued in both directions over
    $[0, 1.01T]$, with a user point at $\tau_2 = (1 - \theta/2\pi)T$ for every
    requested phase $\theta$ (``docs/design/isochrons.md`` §6).

    **Returns:**

    The rotated solution for each phase that a user point was found for.
    """
    idx = run.problem.idx
    period = float(bundle[f"PAR({idx.period})"])
    targets = {theta: (1 - (theta % (2 * math.pi)) / (2 * math.pi)) * period
               for theta in phases}  # fmt: skip
    targets = {theta: (period if tau == 0 else tau) for theta, tau in targets.items()}
    # A user point is reported where its function changes sign between two steps, so
    # one at the starting value is never reported: the bundle itself is that phase,
    # θ = π for the split at half the period of stage 1.
    start = float(bundle[f"PAR({idx.seg2})"])
    found: dict[float, _Solution] = {
        theta: _Solution.from_auto(bundle, ndim=8, npar=npar)
        for theta, target in targets.items()
        if abs(target - start) < 1e-8 * period
    }
    for ds in (-0.01, 0.01):
        branch = run(
            bundle, ICP=[idx.seg2, idx.mult, idx.seg1, idx.period], SP=[], DS=ds,
            DSMAX=0.05, NMX=2000, NPR=0, UZR={idx.seg2: list(targets.values())},
            RL0=0.0, RL1=1.01 * period, **_bvp(8, 11),
        )  # fmt: skip
        for uz in branch("UZ"):
            tau = float(uz[f"PAR({idx.seg2})"])
            for theta, target in targets.items():
                if theta not in found and abs(tau - target) < 1e-8 * period:
                    found[theta] = _Solution.from_auto(uz, ndim=8, npar=npar)
    return found


def _bounding_box(
    cycle: LimitCycle, scale: float, n_points: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """The cycle's bounding box enlarged `scale` times about its center:
    ``(lower, upper)``, each of shape ``(2,)``."""
    with jax.enable_x64(True):
        _, points = cycle.sample(n_points)
    points = np.asarray(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    return center - scale * half, center + scale * half


def _isochron_start(
    rotated: _Solution, cycle: LimitCycle, idx: Any, n_periods: int
) -> _Solution:
    """The start of stage 4 on AUTO's stored grid (decision D2): ``U(1:8)`` as stored
    and ``U(9:10)`` the cycle from the base point at the same $t$, scaled to
    `n_periods` periods."""
    period = rotated.par[idx.period]
    base = rotated.u[0, 2:4]
    with jax.enable_x64(True):
        orbit = cycle.ode.flow(
            jnp.asarray(rotated.t * n_periods * period), jnp.asarray(base),
            config=cycle.config,
        )  # fmt: skip
    u = np.column_stack([rotated.u, np.asarray(orbit.ys)])
    par = dict(rotated.par) | {idx.copies: n_periods * period, idx.eta: 0.0}
    return _Solution(rotated.t, u, par)


def _stop_reason(
    end: Any, idx: Any, settings: IsochronSettings, box: tuple[np.ndarray, np.ndarray]
) -> str:
    r"""Why a half stopped, from the last solution of its run.

    A user point is a box stop when the traced point lies on an edge of `box`, and an
    $\eta$ stop otherwise. AUTO locates a user point only to its own tolerance, so the
    test is on the box, whose edges AUTO reaches to round-off; an $\eta$ stop was once
    classified by $\vert\eta\vert \ge$ ``eta_stop`` and a point at
    $\eta = -9.99999995\times 10^{-7}$ was reported as a box stop.

    **Raises:**

    `NotConvergedError` when AUTO stopped for any other reason, such as a failed
    Newton correction (``MX``).
    """
    kind = end["TY"]
    if kind == "UZ":
        point = np.asarray([end[f"PAR({i})"] for i in idx.point], dtype=float)
        scale = float(np.max(box[1] - box[0]))
        on_edge = np.isclose(point, box[0], rtol=0, atol=1e-9 * scale) | np.isclose(
            point, box[1], rtol=0, atol=1e-9 * scale
        )
        return "box" if on_edge.any() else "eta"
    if kind == "EP" and abs(int(end["PT"])) >= settings.max_steps:
        return "max_steps"
    raise NotConvergedError("isochron", f"AUTO stopped with label {kind}.", None)


def _isochron(
    run: _Runner,
    rotated: _Solution,
    cycle: LimitCycle,
    phase: float,
    settings: IsochronSettings,
    box: tuple[np.ndarray, np.ndarray],
) -> Isochron:
    r"""Stage 4: the isochron of the rotated solution's base point, traced in both
    directions of $\eta$ (``NDIM = 10``, ``NBC = 13``; free: ``eta``, ``mult``,
    ``seg1``, ``period``; printed: the base point and the traced point)."""
    idx = run.problem.idx
    start = _isochron_start(rotated, cycle, idx, settings.n_periods)
    icp = [idx.eta, idx.mult, idx.seg1, idx.period, *idx.base, *idx.point]
    common = dict(ICP=icp, NTST=settings.mesh_intervals, **_bvp(10, 13))
    corrected = run(start.start(), PAR=start.par, NMX=1, **common)()[-1]
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
        halves.append((points, eta, _stop_reason(branch()[-1], idx, settings, box)))
    (p1, e1, s1), (p2, e2, s2) = halves
    points = np.concatenate([p1[::-1], p2[1:]])
    eta = np.concatenate([e1[::-1], e2[1:]])
    base = np.asarray(
        [corrected[f"PAR({idx.base[0]})"], corrected[f"PAR({idx.base[1]})"]]
    )
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    arclength -= arclength[len(p1) - 1]  # zero at the base point
    return Isochron(phase, base, points, eta, arclength, (s1, s2))


# ------------------------------------------------------------------ the driver ----
def _init_worker() -> None:
    """A pool worker traces one phase at a time on one core (decision 17), with JAX on
    the CPU.

    Each worker unpickles the cycle's arrays and integrates the start of stage 4, so on
    a machine with a GPU each one would start a GPU backend and preallocate most of its
    memory; five workers ran the GPU out of memory five times. The worker's
    integrations are small and two-dimensional. JAX is already imported when the
    initializer runs, so the platform is set through its configuration, not the
    environment.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    jax.config.update("jax_platforms", "cpu")


def _trace_phase(
    cycle: LimitCycle,
    rotated: _Solution,
    phase: float,
    settings: IsochronSettings,
    box: tuple[np.ndarray, np.ndarray],
) -> Isochron | IsochronFailure:
    """Stage 4 for one phase in its own temporary directory; a failure is returned,
    not raised, so that the other phases go on."""
    auto = load_auto()
    with _workdir(cycle.ode, "isochron") as problem:
        run = _Runner(auto, problem)
        try:
            return _isochron(run, rotated, cycle, phase, settings, box)
        except Exception as exc:  # reported by name with AUTO's output (G4)
            return IsochronFailure(phase, repr(exc), run.tail(), run.constants)


def _provenance(auto: AutoHandle, problem: AutoProblem) -> dict[str, str]:
    """The git state, the AUTO commit, the Fortran hash, the compiler and its flags
    (decision G6)."""
    import subprocess

    from ....utils.provenance import git_state
    from ._env import pinned_commit

    sha, dirty = git_state()
    compiler = subprocess.run(
        ["gfortran", "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    flags = [
        line.strip()
        for line in (auto.directory / "cmds" / "cmds.make").read_text().splitlines()
        if line.split("=")[0].strip() in ("FC", "FFLAGS", "OPT")
    ]
    return {
        "git_sha": sha,
        "git_dirty": str(dirty),
        "auto_commit": pinned_commit() or "",
        "fortran_sha": problem.fortran_sha,
        "compiler": compiler[0] if compiler else "",
        "compile_flags": "; ".join(flags),
    }


def isochron_foliation(
    cycle: LimitCycle,
    phases: Sequence[float] | Float[ArrayLike, " k"],
    settings: IsochronSettings = IsochronSettings(),
) -> IsochronFoliation:
    r"""The isochrons of `cycle` at `phases`, by AUTO-07p continuation.

    Phase 0 is ``cycle.x0s[0]``; `reanchor` moves it. Stages 1–3 run once here, and
    the isochron of each phase is traced in a pool of ``settings.workers`` processes
    started with ``spawn``, since JAX runs threads and a forked copy of a threaded
    process can deadlock (decision G2).

    **Arguments:**

    - `cycle`: An attracting planar cycle from `find_limit_cycle`.
    - `phases`: The phases, in radians; they are wrapped to $(-\pi, \pi]$.
    - `settings`: How the isochrons are traced.

    **Returns:**

    An `IsochronFoliation`, with one `Isochron` per traced phase in the order
    requested and one `IsochronFailure` per phase that failed.

    **Raises:**

    `ValueError` for a cycle found in reverse time, or for a phase that is not finite
    or that repeats another after wrapping; `AutoNotInstalled` without a built AUTO
    tree; `NotConvergedError` when a stage before the per-phase tracing fails.
    """
    if cycle.reverse_time:
        raise ValueError("isochrons are traced for attracting cycles only.")
    wrapped = _check_phases(phases)
    auto = load_auto()
    with _workdir(cycle.ode, "isochron") as problem:
        run = _Runner(auto, problem)
        npar = max(problem.idx.last, 36)
        orbit = _periodic_orbit(run, cycle)
        bundle = _adjoint_bundle(run, orbit)
        rotated = _rotate_phase(run, bundle, wrapped, npar)
        provenance = _provenance(auto, problem)
    failures = [
        IsochronFailure(theta, "no user point at this phase in stage 3", "", {})
        for theta in wrapped
        if theta not in rotated
    ]
    box = _bounding_box(cycle, settings.box_scale)
    todo = [theta for theta in wrapped if theta in rotated]
    results: dict[float, Isochron | IsochronFailure] = {}
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(settings.workers, max(len(todo), 1)),
        mp_context=context,
        initializer=_init_worker,
    ) as pool:
        futures = {
            pool.submit(
                _trace_phase, cycle, rotated[theta], theta, settings, box
            ): theta
            for theta in todo
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    isochrons = tuple(r for t in todo if isinstance(r := results[t], Isochron))
    failures += [r for t in todo if isinstance(r := results[t], IsochronFailure)]
    return IsochronFoliation(cycle, isochrons, tuple(failures), settings, provenance)


__all__ = [
    "LOG_TAIL",
    "STOP_REASONS",
    "Isochron",
    "IsochronFailure",
    "IsochronFoliation",
    "IsochronSettings",
    "isochron_foliation",
]
