"""Trajectory generation: ``generate(system, ic_sampler, ts, n, ...)`` -> a data source.

Initial conditions are drawn by an ``AbstractICSampler`` (a small equinox module so that
its configuration is recorded in the metadata), the system's ``flow`` is vmapped over
them with ``throw=False``, and any failed integration is reported *loudly* with the
indices of the offending initial conditions rather than silently dropped. The returned
source carries a grouped ``DatasetMetadata`` (``system``, ``sampling``, ``grid``,
``solve``, ``provenance``) whose ``config_hash`` names the file: ``<name>-<hash>.nc``.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import importlib.metadata
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
from .dataset import (
    DatasetMetadata,
    GridSpec,
    Provenance,
    SamplingSpec,
    SolveSpec,
    SystemSpec,
    TimeSeriesDataSource,
)


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


class OnCycleGaussian(AbstractICSampler):
    r"""Yawata et al. (Chaos 34, 063111, 2024), Eqs. (27)–(28): a point of the limit
    cycle, drawn uniformly from ``cycle_points``, plus Gaussian noise
    $\gamma_2\,\sigma \odot \xi$ with $\sigma$ the per-coordinate standard deviation of
    the cycle points and $\xi \sim \mathcal N(0, I)$ (their $\gamma_2 = 0.5$). The
    paper evolves each initial state for $\gamma_1 T = 3T$; that is the ``ts`` passed
    to ``generate``.

    ``cycle_points`` come from ``AbstractNormalForm.limit_cycle`` (``from_normal_form``)
    or, for an observed system, from a long integration of one orbit.
    """

    cycle_points: Float[Array, "points dim"]
    gamma2: float

    def __init__(self, cycle_points, gamma2: float = 0.5):
        self.cycle_points = jnp.asarray(cycle_points, dtype=float)
        if self.cycle_points.ndim != 2 or self.cycle_points.shape[0] < 2:
            raise ValueError("cycle_points must be (points, dim) with >= 2 points.")
        if gamma2 < 0:
            raise ValueError("gamma2 must be non-negative.")
        self.gamma2 = float(gamma2)

    @classmethod
    def from_normal_form(
        cls,
        normal_form: AbstractNormalForm,
        num_points: int = 1000,
        gamma2: float = 0.5,
    ) -> "OnCycleGaussian":
        theta = jnp.linspace(-jnp.pi, jnp.pi, num_points, endpoint=False)
        return cls(jax.vmap(normal_form.limit_cycle)(theta), gamma2)

    @property
    def sigma(self) -> Float[Array, " dim"]:
        return jnp.std(self.cycle_points, axis=0)

    def params(self) -> dict[str, Any]:
        return {
            "gamma2": self.gamma2,
            "num_cycle_points": int(self.cycle_points.shape[0]),
            "sigma": self.sigma.tolist(),
        }

    def __call__(self, key, n):
        k_idx, k_xi = jax.random.split(key)
        idx = jax.random.randint(k_idx, (n,), 0, self.cycle_points.shape[0])
        xi = jax.random.normal(k_xi, (n, self.cycle_points.shape[1]))
        return self.cycle_points[idx] + self.gamma2 * self.sigma * xi


# ------------------------------------------------------------------- provenance ---
def _git_state() -> tuple[str, bool]:
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
        return sha, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "", False


def _package_version() -> str:
    try:
        return importlib.metadata.version("deep-isochron")
    except importlib.metadata.PackageNotFoundError:
        return ""


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

    sha, dirty = _git_state()
    meta = DatasetMetadata(
        system=SystemSpec(type(system).__name__, system.params()),
        sampling=SamplingSpec(
            type(ic_sampler).__name__, ic_sampler.params(), seed, n_trajectories
        ),
        grid=GridSpec(float(ts[0]), float(ts[-1]), int(ts.shape[0])),
        solve=SolveSpec(**config.params(), integration=integration_name),
        provenance=Provenance(
            created=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            git_sha=sha,
            git_dirty=dirty,
            package_version=_package_version(),
            dtype=str(ys.dtype),
        ),
        extra=extra or {},
    )
    return TimeSeriesDataSource(np.asarray(ts), np.asarray(ys), meta)


def dataset_path(root: str | Path, name: str, meta: DatasetMetadata) -> Path:
    """``<root>/<name>-<config_hash>.nc``."""
    return Path(root) / f"{name}-{meta.config_hash}.nc"
