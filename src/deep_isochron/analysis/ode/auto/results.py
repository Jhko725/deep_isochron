"""The results of the isochron pipeline and how they are traced, saved and read.

`IsochronSettings` says how the isochrons are traced; `Isochron` is one traced curve,
`IsochronFailure` a phase that could not be traced, and `IsochronFoliation` the
isochrons of a cycle with the settings and the provenance, saved to one netCDF file
(decision 16) and read back against the cycle it was traced for. The stages that
produce them are in ``stages``, the driver in ``foliation``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from jaxtyping import Float, Int

from ....data.dataset import ENGINE
from ..limit_cycle import LimitCycle
from .codegen import AutoTolerances


STOP_REASONS = ("eta", "box", "max_steps")
r"""Why a half of an isochron stopped: $\vert\eta\vert$ reached ``eta_stop``, the
traced point left the box, or the continuation ran out of steps."""


@dataclass(frozen=True)
class IsochronSettings:
    r"""How the isochrons are traced.

    **Attributes:**

    - `n_periods`: The number of periods of the isochron segment; more periods reach
        further toward the phaseless set (``docs/design/isochrons.md`` §4).
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
    - `epsl`, `epsu`, `epss`: AUTO's convergence tolerances (`AutoTolerances`), in
        force on every run of every stage; the defaults are the coworker's.
    """

    n_periods: int = 5
    eta_stop: float = 1e-6
    box_scale: float = 2.5
    ntst: int | None = None
    max_workers: int | None = None
    max_steps: int = 8000
    ds: float = 1e-5
    epsl: float = AutoTolerances.epsl
    epsu: float = AutoTolerances.epsu
    epss: float = AutoTolerances.epss

    def __post_init__(self):
        if self.n_periods < 1:
            raise ValueError("n_periods must be at least 1.")
        if not (self.eta_stop > 0 and self.box_scale > 1 and self.ds > 0):
            raise ValueError("eta_stop and ds must be positive, box_scale above 1.")
        self.tolerances  # validates the three tolerances
        if self.ntst is not None and self.ntst < 1:
            raise ValueError("ntst must be positive.")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")

    @property
    def tolerances(self) -> AutoTolerances:
        """`epsl`, `epsu` and `epss` as the `AutoTolerances` a session takes."""
        return AutoTolerances(self.epsl, self.epsu, self.epss)

    @property
    def mesh_intervals(self) -> int:
        """`ntst`, or ``60 * n_periods`` when it is `None`."""
        return 60 * self.n_periods if self.ntst is None else self.ntst

    @property
    def workers(self) -> int:
        """`max_workers`, or the cores this process may use when it is `None`."""
        return self.max_workers or os.process_cpu_count() or 1


@dataclass(frozen=True)
class Isochron:
    r"""One traced isochron, as host-side NumPy arrays.

    **Attributes:**

    - `phase`: The phase $\Theta \in [0, 1)$ of the base point, in turns.
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

    def resample(self, spacing: float) -> Isochron:
        r"""The isochron at points evenly spaced in arclength.

        The grid is $s_k = k\,$`spacing`, anchored at the base point ($s = 0$, kept
        exactly) and covering the traced arclength; the traced ends are dropped where
        they fall between grid points. Points are interpolated linearly in arclength,
        $\eta$ linearly in $\log\vert\eta\vert$ with its sign, since it varies
        geometrically along a curve. Between two traced points the chord is off the
        curve by about $\kappa h^2/8$ for a step $h$ and curvature $\kappa$, under
        $10^{-5}$ for AUTO's steps on FitzHugh–Nagumo. A `spacing` coarser than the
        curve's windings, which near the phaseless set have radii of $10^{-3}$, cuts
        across them.

        **Arguments:**

        - `spacing`: The arclength between consecutive points.

        **Returns:**

        An `Isochron` with the same phase, base point and stop reasons.

        **Raises:**

        `ValueError` unless `spacing` is positive and finite.
        """
        if not (math.isfinite(spacing) and spacing > 0):
            raise ValueError(f"spacing must be positive and finite; got {spacing}.")
        s = self.arclength
        grid = spacing * np.arange(
            math.ceil(s.min() / spacing - 1e-12),
            math.floor(s.max() / spacing + 1e-12) + 1,
        )
        points = np.column_stack(
            [np.interp(grid, s, self.points[:, j]) for j in (0, 1)]
        )
        base = np.argmin(np.abs(grid))
        points[base] = self.base_point
        eta = np.zeros_like(grid)
        for side in (-1.0, 1.0):  # each half on its own; eta is 0 at the base point
            traced = np.sign(s) == side
            wanted = np.sign(grid) == side
            if traced.any() and wanted.any():
                magnitude = np.log(np.abs(self.eta[traced]))
                eta[wanted] = side * np.exp(
                    np.interp(grid[wanted], s[traced], magnitude)
                )
        return copy.replace(self, points=points, eta=eta, arclength=grid)

    def log_polar(
        self, center: Float[np.ndarray, "2"]
    ) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
        r"""The traced points in log-polar coordinates about `center`, along the
        curve: $\log_{10}$ of the distance, and the polar angle unwrapped along the
        curve, in windings. An isochron spirals onto the phaseless point, so about the
        equilibrium the distance falls geometrically and the angle winds; the pair
        shows both on linear axes.

        **Arguments:**

        - `center`: The point the coordinates are taken about, the equilibrium.

        **Returns:**

        ``(log_radius, turns)``, each of shape ``(n,)``.
        """
        d = self.points - center
        angle = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        return np.log10(np.hypot(d[:, 0], d[:, 1])), angle / (2 * math.pi)

    def hairpins(
        self, center: Float[np.ndarray, "2"], min_excursion: float = 0.05
    ) -> Int[np.ndarray, " k"]:
        r"""The indices of the hairpins of the curve about `center`: the reversals of
        its unwrapped polar angle, keeping a reversal only when the angle has moved by
        more than `min_excursion` turns since the last kept one (or the start), so
        that a wiggle is not a hairpin.

        **Arguments:**

        - `center`: As for `log_polar`.
        - `min_excursion`: The least angular excursion, in turns, between hairpins.

        **Returns:**

        The indices into `points`, in order along the curve, shape ``(k,)``.
        """
        _, turns = self.log_polar(center)
        # The local-extrema test of the unwrapped angle.
        extrema = np.flatnonzero(np.diff(np.sign(np.diff(turns)))) + 1
        kept: list[int] = []
        last = 0
        for i in extrema:
            if abs(turns[i] - turns[last]) > min_excursion:
                kept.append(int(i))
                last = i
        return np.asarray(kept, dtype=int)


@dataclass(frozen=True)
class IsochronFailure:
    """A phase whose isochron could not be traced.

    **Attributes:**

    - `phase`: The requested phase.
    - `message`: What went wrong.
    - `log_tail`: The last lines of AUTO's output (``runner.LOG_TAIL`` of them).
    - `constants`: The run constants of the failed run.
    - `workdir`: The session's directory, kept with the generated files and AUTO's
        output files for inspection; empty when no session ran for the phase.
    """

    phase: float
    message: str
    log_tail: str
    constants: dict[str, Any]
    workdir: str


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
                "phase": (
                    "isochron",
                    [iso.phase for iso in self.isochrons],
                    {"units": "turns"},
                ),
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

        `ValueError` when the file records no phase unit (a file written before
        ADR-0020 holds angles), or when `cycle` is of another system, other
        parameters, another period or another phase origin than the file records.
        """
        with xr.open_dataset(path, engine=ENGINE) as ds:
            ds = ds.load()
        if ds["phase"].attrs.get("units") != "turns":
            raise ValueError(
                f"{path} records no phase unit; files written before ADR-0020 hold "
                "angles and are not read as turns."
            )
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


__all__ = [
    "STOP_REASONS",
    "Isochron",
    "IsochronFailure",
    "IsochronFoliation",
    "IsochronSettings",
]
