"""Running AUTO-07p on a system's generated problem, from a session.

``auto_session(ode)`` loads AUTO, writes the problem files of `ode` into a temporary
directory, makes that directory the working directory and yields a `Runner`, which
runs AUTO on one thread with its output captured. Every stage of the pipeline goes
through the runner, and every run passes the constants of `bvp`, because AUTO reloads
the constants file on each run. An `AutoSolution` is an AUTO solution as plain arrays,
the one shape that the stages exchange and that crosses a process boundary.

AUTO writes its files into the working directory, which belongs to the whole process,
so a session is not thread-safe; the pipeline runs its sessions in processes.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jaxtyping import Float

from ...errors import NotConvergedError
from ._env import AutoHandle, load_auto
from .codegen import AutoProblem, Idx, write_problem


logger = logging.getLogger(__name__)

LOG_TAIL = 50
"""The number of lines of AUTO's output that a failed phase keeps."""

MIN_NPAR = 36
"""The number of parameter slots an `AutoSolution` copies at least: AUTO's internal
slots up to the period, ``PAR(11)``, and the pipeline's own block."""


@dataclass(frozen=True)
class AutoSolution:
    """An AUTO solution as plain arrays.

    **Attributes:**

    - `t`: The stored mesh of $[0, 1]$, shape ``(n,)``.
    - `u`: The state at those points, shape ``(n, ndim)``.
    - `par`: The parameters by slot.
    - `label`: AUTO's type label of the point, such as ``"UZ"``, ``"BP"`` or
        ``"EP"``; empty for a start solution built by a stage.
    - `step`: AUTO's point number on its branch; 0 for a start solution.
    """

    t: Float[np.ndarray, " n"]
    u: Float[np.ndarray, "n ndim"]
    par: dict[int, float]
    label: str = ""
    step: int = 0

    @classmethod
    def from_auto(cls, solution: Any, npar: int = MIN_NPAR) -> AutoSolution:
        """Copy an AUTO solution's mesh, every stored component and the first `npar`
        parameters."""
        t = np.asarray(solution["t"], dtype=float)
        ndim = len(solution.coordnames)
        u = np.column_stack([solution[f"U({i})"] for i in range(1, ndim + 1)])
        par = {i: float(solution[f"PAR({i})"]) for i in range(1, npar + 1)}
        return cls(t, u, par, str(solution["TY"]), abs(int(solution["PT"])))

    def start(self) -> Float[np.ndarray, "rows n"]:
        """The array start solution AUTO's ``run`` takes: rows ``t, U(1), ...``."""
        return np.vstack([self.t, self.u.T])


class Runner:
    """Runs AUTO on the session's problem, one run per call.

    Each run sets ``OMP_NUM_THREADS=1`` (`one_thread`), captures AUTO's output, logs it
    at DEBUG and keeps it for a failure report, and remembers the run constants of the
    last run.

    **Attributes:**

    - `auto`: The loaded AUTO.
    - `problem`: The generated problem of the session's system.
    - `idx`: The problem's parameter slots.
    - `output`: Every line AUTO printed in this session.
    - `constants`: The run constants of the last run.
    """

    def __init__(self, auto: AutoHandle, problem: AutoProblem):
        self.auto = auto
        self.problem = problem
        self.idx: Idx = problem.idx
        self.output: list[str] = []
        self.constants: dict[str, Any] = {}

    @property
    def npar(self) -> int:
        """The number of parameter slots to copy from a solution: the problem's block
        or `MIN_NPAR`, whichever is larger."""
        return max(self.idx.last, MIN_NPAR)

    def __call__(self, start: Any, **constants: Any) -> Any:
        """One AUTO run from `start`, an `AutoSolution`, an AUTO solution or a start
        array, with the run constants `constants`.

        **Returns:**

        AUTO's branch object; `solution` and `labeled` turn its points into
        `AutoSolution`s.
        """
        if isinstance(start, AutoSolution):
            constants = {"PAR": start.par} | constants
            start = start.start()
        self.constants = constants
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), one_thread():
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

    def solution(self, branch: Any, index: int = -1) -> AutoSolution:
        """The `index`-th labeled point of `branch` as an `AutoSolution`; the last by
        default."""
        return AutoSolution.from_auto(branch()[index], self.npar)

    def labeled(self, branch: Any, label: str) -> list[AutoSolution]:
        """The points of `branch` with AUTO's type `label`, such as ``"UZ"`` or
        ``"BP"``, as `AutoSolution`s."""
        return [AutoSolution.from_auto(s, self.npar) for s in branch(label)]

    def tail(self) -> str:
        """The last `LOG_TAIL` lines of AUTO's output."""
        return "\n".join(self.output[-LOG_TAIL:])


@contextlib.contextmanager
def auto_session(ode, name: str = "isochron") -> Generator[Runner]:
    """A `Runner` on the AUTO problem of `ode`, written into a temporary directory that
    is the working directory for the duration of the block and deleted afterward. JAX
    runs in float64 inside the block, whatever the calling process's setting, since
    the stages integrate with O1's solver at its tolerances.

    **Raises:**

    `AutoNotInstalled` without a built AUTO tree.
    """
    auto = load_auto()
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="deep_isochron_auto_") as tmp:
        problem = write_problem(ode, tmp, name=name)
        os.chdir(tmp)
        try:
            with jax.enable_x64(True):
                yield Runner(auto, problem)
        finally:
            os.chdir(previous)


@contextlib.contextmanager
def one_thread() -> Generator[None]:
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


def bvp(ndim: int, nbc: int) -> dict[str, int]:
    """The run constants of the staged boundary value problem with `ndim` components
    and `nbc` boundary conditions. AUTO's ``run`` applies the constants file first and
    every keyword passed to the call on top, and nothing survives from the run before,
    so a stage states what differs from the file on every run, these included."""
    return {"IPS": 4, "NDIM": ndim, "NBC": nbc, "NINT": 0}


def check_stage(stage: str, condition: bool, message: str) -> None:
    """Raise `NotConvergedError` for `stage` with `message` unless `condition`."""
    if not condition:
        raise NotConvergedError(stage, message, None)


__all__ = [
    "LOG_TAIL",
    "MIN_NPAR",
    "AutoSolution",
    "Runner",
    "auto_session",
    "bvp",
    "check_stage",
    "one_thread",
]
