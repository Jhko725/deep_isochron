"""Trajectory generation: ``generate`` turns a system, a sampler of initial conditions
and a time grid into a ``TimeSeriesDataSource``.

Initial conditions are drawn by an ``AbstractICSampler``, a small equinox module so
that its configuration is recorded in the metadata; the system's ``flow`` is vmapped
over them with ``throw=False``, and a failed integration is reported with the indices
of the offending initial conditions rather than dropped. The returned source carries a
grouped ``DatasetMetadata`` whose ``config_hash`` names the file, ``<name>-<hash>.nc``;
``generation_metadata`` computes that metadata without generating, which is how the
experiment layer finds a dataset. ADR-0008 has the format, ADR-0013 the basin rule.
"""

from __future__ import annotations

import copy
import datetime
import warnings
from pathlib import Path
from typing import Any, Literal

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..systems.base import AbstractODE, DEFAULT_SOLVER_CONFIG, SolverConfig
from ..systems.normal_forms import (
    AbstractFlowIntegration,
    AbstractNormalForm,
    resolve_integration,
)
from ..utils.provenance import git_state, package_version
from .dataset import (
    DatasetMetadata,
    GridSpec,
    Provenance,
    SamplingSpec,
    SolveSpec,
    SystemSpec,
    TimeSeriesDataSource,
)
from .initial_conditions import AbstractICSampler


def generate(
    system: AbstractODE,
    ic_sampler: AbstractICSampler,
    ts: Float[Array, " time"],
    n_trajectories: int,
    *,
    seed: int,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    integration: AbstractFlowIntegration | str | None = None,
    extra: dict[str, Any] | None = None,
    outside_basin: Literal["error", "resample"] = "error",
) -> TimeSeriesDataSource:
    """Integrate `n_trajectories` initial conditions of `system` on the grid `ts`.

    **Arguments:**

    - `system`: The ODE.
    - `ic_sampler`: The sampler of initial conditions.
    - `ts`: The time grid, shape ``(time,)``.
    - `n_trajectories`: How many trajectories to generate.
    - `seed`: The seed of the initial-condition draw, recorded in the metadata.
    - `config`: The solver settings; `throw` is forced off so that failures are
        reported per trajectory.
    - `integration`: For an ``AbstractNormalForm`` only; ``None`` is the system's
        `default_integration`.
    - `extra`: Free metadata recorded under ``extra``.
    - `outside_basin`: What to do with initial conditions outside ``system.in_basin``:
        ``"error"`` raises; ``"resample"`` replaces them by further draws until
        `n_trajectories` lie inside, with a warning and the count in
        ``metadata.provenance.rejected_ics``.

    **Returns:**

    The `TimeSeriesDataSource` with its metadata and provenance.

    **Raises:**

    `RuntimeError` naming the indices of failed integrations; `ValueError` for initial
    conditions outside the basin under ``"error"``."""
    ts = jnp.asarray(ts)
    u0, rejected = _sample_in_basin(
        system, ic_sampler, jax.random.key(seed), n_trajectories, outside_basin
    )
    config = copy.replace(config, throw=False)
    meta = generation_metadata(
        system,
        ic_sampler,
        ts,
        n_trajectories,
        seed=seed,
        config=config,
        integration=integration,
        extra=extra,
        outside_basin=outside_basin,
    )

    if isinstance(system, AbstractNormalForm):
        method = resolve_integration(integration or system.default_integration)

        def flow(u):
            sol = system.flow(ts, u, config=config, integration=method)
            return sol.ys, sol.result

    else:

        def flow(u):
            sol = system.flow(ts, u, config=config)
            return sol.ys, sol.result

    ys, result = eqx.filter_vmap(flow)(u0)
    ok = np.asarray(result == dfx.RESULTS.successful)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        raise RuntimeError(
            f"{len(bad)} of {n_trajectories} integrations failed (indices {bad[:10]}"
            f"{'...' if len(bad) > 10 else ''}); tighten SolverConfig or change the "
            "initial-condition sampler."
        )

    sha, dirty = git_state()
    meta = copy.replace(
        meta,
        provenance=Provenance(
            created=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            git_sha=sha,
            git_dirty=dirty,
            package_version=package_version(),
            dtype=str(ys.dtype),
            rejected_ics=rejected,
        ),
    )
    return TimeSeriesDataSource(np.asarray(ts), np.asarray(ys), meta)


RESAMPLE_ROUNDS = 100
"""Upper bound on redraws in ``generate(outside_basin="resample")``."""


def _sample_in_basin(
    system: AbstractODE,
    ic_sampler: AbstractICSampler,
    key: Any,
    n: int,
    outside_basin: str,
) -> tuple[Array, int]:
    """`n` initial conditions inside the basin of `system`, and how many draws were
    rejected on the way. Under ``"error"`` any rejection raises; under ``"resample"``
    the sampler is redrawn, `n` at a time with fresh keys, until `n` are accepted, and
    a warning reports the count."""
    if outside_basin not in ("error", "resample"):
        raise ValueError(
            f"outside_basin must be 'error' or 'resample', got {outside_basin!r}."
        )
    in_basin = eqx.filter_jit(eqx.filter_vmap(system.in_basin))
    u0 = ic_sampler(key, n)
    ok = np.asarray(in_basin(u0))
    rejected = int((~ok).sum())
    if rejected == 0:
        return u0, 0
    if outside_basin == "error":
        bad = np.flatnonzero(~ok)
        raise ValueError(
            f"{rejected} of {n} initial conditions lie outside the basin of "
            f"{type(system).__name__} (indices {bad[:10]}"
            f"{'...' if rejected > 10 else ''}); change the sampler, or pass "
            "outside_basin='resample'."
        )
    accepted = [u0[ok]]
    have = n - rejected
    for _ in range(RESAMPLE_ROUNDS):
        if have >= n:
            break
        key, sub = jax.random.split(key)
        more = ic_sampler(sub, n)
        ok = np.asarray(in_basin(more))
        rejected += int((~ok).sum())
        accepted.append(more[ok])
        have += int(ok.sum())
    else:
        raise RuntimeError(
            f"could not draw {n} initial conditions inside the basin of "
            f"{type(system).__name__} in {RESAMPLE_ROUNDS} rounds; the sampler barely "
            "overlaps the basin."
        )
    warnings.warn(
        f"{rejected} initial condition(s) outside the basin of {type(system).__name__} "
        f"were rejected and replaced by further draws (outside_basin='resample').",
        stacklevel=3,
    )
    return jnp.concatenate(accepted)[:n], rejected


def generation_metadata(
    system: AbstractODE,
    ic_sampler: AbstractICSampler,
    ts: Float[Array, " time"],
    n_trajectories: int,
    *,
    seed: int,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    integration: AbstractFlowIntegration | str | None = None,
    extra: dict[str, Any] | None = None,
    outside_basin: str = "error",
) -> DatasetMetadata:
    """The metadata that `generate` would record for these arguments, without
    generating, with the provenance left at its defaults.

    Its ``config_hash`` is the one the generated file carries, so
    ``dataset_path(root, name, generation_metadata(...))`` is where a dataset of this
    configuration lives; ``experiment.build_source`` relies on that. The arguments are
    those of `generate`.

    **Raises:**

    `ValueError` for an `integration` on a system that is not a normal form."""
    ts = jnp.asarray(ts)
    config = copy.replace(config, throw=False)
    if isinstance(system, AbstractNormalForm):
        method = resolve_integration(integration or system.default_integration)
        integration_name = type(method).__name__
    else:
        if integration is not None:
            raise ValueError("integration applies to AbstractNormalForm systems only.")
        integration_name = ""
    return DatasetMetadata(
        system=SystemSpec(type(system).__name__, system.params()),
        sampling=SamplingSpec(
            type(ic_sampler).__name__,
            ic_sampler.params(),
            seed,
            n_trajectories,
            outside_basin=outside_basin,
        ),
        grid=GridSpec(float(ts[0]), float(ts[-1]), int(ts.shape[0])),
        solve=SolveSpec(**config.params(), integration=integration_name),
        extra=extra or {},
    )


def dataset_path(root: str | Path, name: str, meta: DatasetMetadata) -> Path:
    """The path ``<root>/<name>-<config_hash>.nc`` of the dataset `meta` describes."""
    return Path(root) / f"{name}-{meta.config_hash}.nc"
