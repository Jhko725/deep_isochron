r"""The driver: the isochrons of a cycle at requested phases (roadmap O7).

``isochron_foliation(cycle, phases, settings)`` is the one entry point. It runs stages
1–3 of ``stages`` once in the calling process, then `trace_isochron`, stage 4 in its own
AUTO session, for each phase in a pool of ``spawn`` processes, and returns an
`IsochronFoliation` with the isochrons, the phases that failed, the settings and the
provenance. AUTO's output goes to the loggers of this subpackage at DEBUG; one INFO
line reports stages 1–3 and one each finished phase.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
import subprocess
import time
from collections.abc import Sequence

import jax
import numpy as np
from jaxtyping import ArrayLike, Float

from ....utils.numerics import wrap_phase
from ....utils.provenance import git_state
from ..limit_cycle import LimitCycle
from ._env import pinned_commit
from .results import Isochron, IsochronFailure, IsochronFoliation, IsochronSettings
from .runner import auto_session, AutoSolution, Runner
from .stages import (
    adjoint_bundle,
    bounding_box,
    Box,
    isochron,
    periodic_orbit,
    rotate_phase,
)


logger = logging.getLogger(__name__)


def wrap_phases(
    phases: Sequence[float] | Float[ArrayLike, " k"],
) -> tuple[float, ...]:
    r"""The requested phases wrapped to $(-\pi, \pi]$.

    **Raises:**

    `ValueError` naming any phase that is not finite or that repeats another after
    wrapping.
    """
    values = np.atleast_1d(np.asarray(phases, dtype=float))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"phases must be finite; got {values[~np.isfinite(values)]}.")
    wrapped = np.asarray(wrap_phase(values))  # NumPy in, NumPy out: float64 as given
    _, first, counts = np.unique(
        np.round(wrapped, 12), return_index=True, return_counts=True
    )
    if np.any(counts > 1):
        raise ValueError(
            "phases repeat after wrapping to (-π, π]: "
            f"{values[first[counts > 1]]} each occur more than once."
        )
    return tuple(wrapped.tolist())


def trace_isochron(
    cycle: LimitCycle,
    rotated: AutoSolution,
    phase: float,
    settings: IsochronSettings,
    box: Box,
) -> Isochron | IsochronFailure:
    """Stage 4 for one phase, in an AUTO session of its own.

    A failure is returned, not raised, so that the other phases of a foliation go on;
    it carries the exception, AUTO's last lines and the run constants (decision G4).

    **Arguments:**

    - `cycle`, `rotated`, `phase`, `settings`, `box`: As for `stages.isochron`.

    **Returns:**

    The `Isochron`, or the `IsochronFailure`.
    """
    with auto_session(cycle.ode) as run:
        try:
            return isochron(run, rotated, cycle, phase, settings, box)
        except Exception as exc:
            return IsochronFailure(phase, repr(exc), run.tail(), run.constants)


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


def provenance(run: Runner) -> dict[str, str]:
    """Where a foliation came from: the package's git state, the AUTO commit, the hash
    of the generated Fortran, the compiler and its flags (decision G6)."""
    sha, dirty = git_state()
    compiler = subprocess.run(
        ["gfortran", "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    make = run.auto.directory / "cmds" / "cmds.make"
    flags = [
        line.strip()
        for line in make.read_text().splitlines()
        if line.split("=")[0].strip() in ("FC", "FFLAGS", "OPT")
    ]
    return {
        "git_sha": sha,
        "git_dirty": str(dirty),
        "auto_commit": pinned_commit() or "",
        "fortran_sha": run.problem.fortran_sha,
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
    `trace_isochron` runs for each phase in a pool of ``settings.workers`` processes
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
    start = time.perf_counter()
    with auto_session(cycle.ode) as run:
        wrapped = wrap_phases(phases)
        orbit = periodic_orbit(run, cycle)
        bundle = adjoint_bundle(run, orbit)
        rotated = rotate_phase(run, bundle, wrapped)
        origin = provenance(run)
        box = bounding_box(cycle, settings.box_scale)
    logger.info(
        "stages 1-3: %d of %d phases rotated in %.0f s",
        len(rotated), len(wrapped), time.perf_counter() - start,
    )  # fmt: skip
    failures = [
        IsochronFailure(theta, "no user point at this phase in stage 3", "", {})
        for theta in wrapped
        if theta not in rotated
    ]
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
                trace_isochron, cycle, rotated[theta], theta, settings, box
            ): theta
            for theta in todo
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            theta = futures[future]
            results[theta] = result = future.result()
            outcome = (
                f"stopped by {result.stop}"
                if isinstance(result, Isochron)
                else f"failed: {result.message}"
            )
            logger.info(
                "phase %+.6f %s (%d/%d, %.0f s)",
                theta, outcome, done, len(todo), time.perf_counter() - start,
            )  # fmt: skip
    isochrons = tuple(r for t in todo if isinstance(r := results[t], Isochron))
    failures += [r for t in todo if isinstance(r := results[t], IsochronFailure)]
    return IsochronFoliation(cycle, isochrons, tuple(failures), settings, origin)


__all__ = ["isochron_foliation", "provenance", "trace_isochron", "wrap_phases"]
