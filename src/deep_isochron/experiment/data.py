"""From the ``data`` and ``windows``/``validation`` config groups to loaders.

The ``data`` group is the *generation* config (``configs/data/*.yaml``, shared with
``scripts/generate_data.py``): ``build_source`` instantiates its system, sampler and
solver exactly as the generation script does, computes the metadata hash with
``generation_metadata`` and loads ``<out_dir>/<name>-<hash>.nc``. It never generates —
a missing file is an error naming the command to run (ADR-0011: generation is explicit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from ..data import (
    dataset_path,
    generate,
    generation_metadata,
    TimeSeriesDataSource,
    to_device,
    transient_weight,
    validation_windows,
)
from ..data.windows import mixed_ranges, WindowBatchSource
from ..systems.normal_forms import AbstractNormalForm
from .instantiate import instantiate


def _generation_args(data_cfg: DictConfig) -> tuple:
    system = instantiate(data_cfg.system)
    ic_sampler = instantiate(data_cfg.ic_sampler)
    solver = instantiate(data_cfg.solver)
    ts = jnp.linspace(data_cfg.time.t0, data_cfg.time.t1, data_cfg.time.n)
    return system, ic_sampler, ts, solver


def dataset_file(data_cfg: DictConfig) -> Path:
    """Where the dataset of this generation config lives (whether or not it exists):
    ``<out_dir>/<name>-<config_hash>.nc`` with the hash ``generate`` would record.

    Datasets are float64 (``generate_data.py`` enables x64), and the hash covers the
    system's parameters as the arrays hold them, so it is computed under x64 whatever
    the caller's setting — otherwise a run with ``x64: false`` would look for a file
    whose name was hashed from float32-rounded parameters."""
    with jax.enable_x64(True):
        system, ic_sampler, ts, solver = _generation_args(data_cfg)
        meta = generation_metadata(
            system,
            ic_sampler,
            ts,
            data_cfg.n_trajectories,
            seed=data_cfg.seed,
            config=solver,
            integration=data_cfg.integration,
        )
    return dataset_path(data_cfg.out_dir, data_cfg.name, meta)


def generate_dataset(data_cfg: DictConfig) -> Path:
    """Generate the dataset of a ``data`` config and save it at ``dataset_file(cfg)``
    (what ``scripts/generate_data.py`` does; the resolved config goes into the file's
    ``extra`` metadata). The one place generation happens."""
    with jax.enable_x64(True):  # datasets are float64, whatever the caller runs in
        system, ic_sampler, ts, solver = _generation_args(data_cfg)
        source = generate(
            system,
            ic_sampler,
            ts,
            data_cfg.n_trajectories,
            seed=data_cfg.seed,
            config=solver,
            integration=data_cfg.integration,
            extra={"config": OmegaConf.to_container(data_cfg, resolve=True)},
        )
    assert source.metadata is not None
    return source.save(dataset_path(data_cfg.out_dir, data_cfg.name, source.metadata))


def build_source(data_cfg: DictConfig) -> TimeSeriesDataSource:
    """Load the dataset the ``data`` config describes; fail loudly if it was not
    generated (``scripts/generate_data.py --config-name <name>``)."""
    path = dataset_file(data_cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset {path} not found. Training never generates data; run\n"
            "    uv run python scripts/generate_data.py --config-name "
            f"{data_cfg.name}\n"
            "(with the same overrides) first."
        )
    return TimeSeriesDataSource.load(path)


def reference_normal_form(data_cfg: DictConfig) -> AbstractNormalForm | None:
    """The normal form the data were generated from, when they were — the
    ``Evaluator``'s ground truth for phase and amplitude; ``None`` otherwise (FHN)."""
    system = instantiate(data_cfg.system)
    return system if isinstance(system, AbstractNormalForm) else None


def validation_t_split(cfg: DictConfig, source: TimeSeriesDataSource) -> float | None:
    """``validation.t_split`` as a time: the configured value; or, when it is ``null``
    and the training sampler is ``mixed``, the sampler's own boundary ``ts[split_idx]``
    so training and validation agree on what the transient is; else ``None``."""
    v = cfg.validation
    if v.t_split is not None:
        return float(v.t_split)
    s = cfg.windows.sampling
    if s.kind == "mixed" and s.split_idx is not None:
        return float(source.ts[int(s.split_idx)])
    return None


@dataclass(frozen=True)
class Loaders:
    """What ``build_loaders`` returns. ``train`` is the device-prefetched stream the
    trainer consumes; ``train_batches`` the underlying finite ``MapDataset`` (sliced at
    ``start_step`` already) for inspection; ``val`` the finite validation dataset for an
    ``Evaluator``."""

    train: grain.IterDataset
    train_batches: grain.MapDataset
    val: grain.MapDataset
    batches_per_epoch: int
    num_windows: int
    train_source: TimeSeriesDataSource
    val_source: TimeSeriesDataSource


def build_window_source(
    cfg: DictConfig, source: TimeSeriesDataSource
) -> WindowBatchSource:
    """The run's batched window source (``windows.sampling.kind``: uniform / weighted /
    mixed), sized to cover ``num_steps``."""
    w = cfg.windows
    s = w.sampling
    kind = s.kind
    if kind == "uniform":
        start_range = None if s.start_range is None else tuple(s.start_range)
        return WindowBatchSource(
            source,
            w.length,
            w.batch,
            seed=w.seed,
            num_steps=cfg.num_steps,
            start_range=start_range,
        )
    if kind == "weighted":
        return WindowBatchSource(
            source,
            w.length,
            w.batch,
            seed=w.seed,
            num_steps=cfg.num_steps,
            weight=transient_weight(s.boost, s.tau),
        )
    if kind == "mixed":
        if s.split_idx is None:
            raise ValueError("windows.sampling.split_idx is required for kind=mixed.")
        return WindowBatchSource(
            source,
            w.length,
            w.batch,
            seed=w.seed,
            num_steps=cfg.num_steps,
            ranges=mixed_ranges(source, w.length, int(s.split_idx)),
            range_weights=(float(s.weights[0]), float(s.weights[1])),
        )
    raise ValueError(f"unknown windows.sampling.kind {kind!r}.")


def build_loaders(
    cfg: DictConfig,
    source: TimeSeriesDataSource,
    device: jax.Device,
    *,
    start_step: int = 0,
) -> Loaders:
    """Split the trajectories, build the batched training stream from ``start_step``
    (resume = the slice ``[start_step:]`` of the same deterministic dataset) and the
    finite validation dataset."""
    v = cfg.validation
    train_source, val_source = source.split_trajectories(v.fraction, seed=v.seed)
    wsrc = build_window_source(cfg, train_source)
    batches = grain.MapDataset.source(wsrc)
    if start_step:
        if start_step >= len(batches):
            raise ValueError(
                f"resume step {start_step} is beyond the run's {len(batches)} batches; "
                "raise num_steps."
            )
        batches = batches[start_step:]
    val = validation_windows(val_source, cfg.windows.length, stride=v.stride).batch(
        v.batch
    )
    return Loaders(
        train=to_device(batches, device),
        train_batches=batches,
        val=val,
        batches_per_epoch=wsrc.batches_per_epoch,
        num_windows=wsrc.num_windows,
        train_source=train_source,
        val_source=val_source,
    )
