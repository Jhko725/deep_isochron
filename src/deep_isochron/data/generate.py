"""Trajectory generation: ``generate(system, ic_sampler, ts, n, ...)`` -> a data source.

Initial conditions are drawn by an ``AbstractICSampler`` (a small equinox module so that
its configuration is recorded in the metadata), the system's ``flow_result`` is vmapped
over them with ``throw=False``, and any failed integration is reported *loudly* with the
indices of the offending initial conditions rather than silently dropped. The returned
source carries a ``DatasetMetadata`` whose ``config_hash`` (first 8 hex digits of the
SHA-256 of the generation-defining fields) names the file: ``<name>-<hash>.nc``.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from ..systems.base import AbstractODE, DEFAULT_SOLVER_CONFIG, SolverConfig
from ..systems.normal_forms import (
    AbstractFlowIntegration,
    AbstractNormalForm,
    resolve_integration,
)
from .dataset import DatasetMetadata, TimeSeriesDataSource


# ------------------------------------------------------------------ IC samplers ---
class AbstractICSampler(eqx.Module):
    @abc.abstractmethod
    def __call__(self, key: PRNGKeyArray, n: int) -> Float[Array, "n dim"]: ...

    def params(self) -> dict[str, Any]:
        """JSON-able configuration for the metadata."""
        return {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in vars(self).items()
        }


class UniformBox(AbstractICSampler):
    """Uniform on the box ``[lo, hi]`` (per-coordinate bounds)."""

    lo: Float[Array, " dim"]
    hi: Float[Array, " dim"]

    def __init__(self, lo, hi):
        self.lo, self.hi = jnp.asarray(lo, dtype=float), jnp.asarray(hi, dtype=float)
        if self.lo.shape != self.hi.shape or not bool(jnp.all(self.lo < self.hi)):
            raise ValueError("lo and hi must have the same shape with lo < hi.")

    def __call__(self, key, n):
        return jax.random.uniform(
            key, (n, self.lo.shape[0]), minval=self.lo, maxval=self.hi
        )


class UniformAnnulus(AbstractICSampler):
    """Uniform in angle and in radius on ``[r_min, r_max]`` about ``center`` (planar);
    avoids the origin of a normal form."""

    r_min: float
    r_max: float
    center: Float[Array, " 2"]

    def __init__(self, r_min: float, r_max: float, center=(0.0, 0.0)):
        if not 0 <= r_min < r_max:
            raise ValueError("need 0 <= r_min < r_max.")
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.center = jnp.asarray(center, dtype=float)

    def __call__(self, key, n):
        kr, kt = jax.random.split(key)
        r = jax.random.uniform(kr, (n,), minval=self.r_min, maxval=self.r_max)
        theta = jax.random.uniform(kt, (n,), minval=-jnp.pi, maxval=jnp.pi)
        return self.center + jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), -1)


# ------------------------------------------------------------------- provenance ---
def _git_state() -> tuple[str, int]:
    """``(sha, dirty)`` of the checkout this package is imported from (not the cwd)."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha, int(bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return "", 0


def _package_version() -> str:
    try:
        return importlib.metadata.version("deep-isochron")
    except importlib.metadata.PackageNotFoundError:
        return ""


_HASHED = (
    "system",
    "system_params",
    "ic_sampler",
    "ic_sampler_params",
    "seed",
    "n_trajectories",
    "n_time",
    "t0",
    "t1",
    "solver",
    "rtol",
    "atol",
    "max_steps",
    "strategy",
    "dtype",
)


def config_hash(meta: DatasetMetadata) -> str:
    """First 8 hex digits of SHA-256 over the generation-defining fields (provenance
    fields — timestamps, git state — excluded, so regenerating gives the same name)."""
    d = {k: getattr(meta, k) for k in _HASHED}
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


# --------------------------------------------------------------------- generate ---
def generate(
    system: AbstractODE,
    ic_sampler: AbstractICSampler,
    ts: Float[Array, " time"],
    n_trajectories: int,
    *,
    seed: int,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    integration: AbstractFlowIntegration | str | None = None,
    window_size: int | None = None,
    extra: dict[str, Any] | None = None,
) -> TimeSeriesDataSource:
    """Integrate ``n_trajectories`` initial conditions of ``system`` on the grid ``ts``.

    ``seed`` seeds the initial-condition draw (recorded in the metadata).
    ``integration`` applies to ``AbstractNormalForm`` systems only (default
    ``"r_squared"``). Failed integrations raise ``RuntimeError`` naming the indices.
    """
    ts = jnp.asarray(ts)
    u0 = ic_sampler(jax.random.key(seed), n_trajectories)
    config = dataclasses.replace(config, throw=False)

    if isinstance(system, AbstractNormalForm):
        method = resolve_integration(integration or "r_squared")

        def flow(u):
            sol = system.flow(ts, u, config=config, integration=method)
            return sol.ys, sol.result

        integration_name = type(method).__name__
    else:
        if integration is not None:
            raise ValueError("integration applies to AbstractNormalForm systems only.")

        def flow(u):
            sol = system.flow(ts, u, config=config)
            return sol.ys, sol.result

        integration_name = ""

    ys, result = eqx.filter_vmap(flow)(u0)
    ok = np.asarray(result == dfx.RESULTS.successful)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        raise RuntimeError(
            f"{len(bad)} of {n_trajectories} integrations failed (indices {bad[:10]}"
            f"{'...' if len(bad) > 10 else ''}); tighten SolverConfig or change the "
            "initial-condition sampler."
        )

    meta = DatasetMetadata(
        system=type(system).__name__,
        system_params=system.params(),
        ic_sampler=type(ic_sampler).__name__,
        ic_sampler_params=ic_sampler.params(),
        seed=seed,
        n_trajectories=n_trajectories,
        n_time=int(ts.shape[0]),
        dim=system.dim,
        t0=float(ts[0]),
        t1=float(ts[-1]),
        solver=type(config.solver).__name__,
        rtol=config.rtol,
        atol=config.atol,
        max_steps=config.max_steps,
        strategy=integration_name,
        dtype=str(ys.dtype),
        created=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        git_sha=_git_state()[0],
        git_dirty=_git_state()[1],
        package_version=_package_version(),
        extra=extra or {},
    )
    meta = dataclasses.replace(meta, config_hash=config_hash(meta))
    return TimeSeriesDataSource(ts, ys, window_size, meta)


def dataset_path(root: str | Path, name: str, meta: DatasetMetadata) -> Path:
    """``<root>/<name>-<config_hash>.nc``."""
    return Path(root) / f"{name}-{meta.config_hash}.nc"
