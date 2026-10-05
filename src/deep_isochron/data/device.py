"""Consuming a grain pipeline next to JAX, whatever it yields: reading without a reader
pool (``single_threaded``), the prefetching transfer to the accelerator (``to_device``),
and the one sanctioned way to pick the device (``resolve_device``). Background:
``docs/design/training-step-performance.md``."""

import os

import grain
import jax


def single_threaded(dataset: grain.MapDataset | grain.IterDataset) -> grain.IterDataset:
    """An ``IterDataset`` that reads with one thread and no read-ahead. Iterating a
    ``MapDataset`` directly starts grain's default reader (16 threads, 500-element
    buffer), which competes with a JAX training loop for the GIL; use this (or
    ``to_device``, which does the same) wherever a pipeline is consumed next to JAX
    work, e.g. validation batches."""
    if isinstance(dataset, grain.IterDataset):
        return dataset
    return dataset.to_iter_dataset(
        grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    )


def to_device(
    dataset: grain.MapDataset | grain.IterDataset,
    device: jax.Device,
    *,
    cpu_buffer_size: int = 4,
    device_buffer_size: int = 2,
) -> grain.IterDataset:
    """grain's two-stage prefetch to the accelerator (its JAX training tutorial's
    "option C", the recommended pattern for real training): one CPU-side thread
    prepares ``cpu_buffer_size`` batches ahead while a second thread keeps
    ``device_buffer_size`` batches already transferred to ``device`` (exactly two
    prefetch threads, one per stage, plus one reader thread; no pool). The transfer
    then overlaps the training step instead of blocking the loop between steps.
    ``device`` is explicit — on a shared machine the caller (or the scheduler, through
    ``CUDA_VISIBLE_DEVICES``) decides which card a job uses, never this function. Apply
    after ``.batch(...)``."""
    ds = single_threaded(dataset)  # one reader; the two prefetch stages buffer
    return grain.experimental.device_put(
        ds,
        device,
        cpu_buffer_size=cpu_buffer_size,
        device_buffer_size=device_buffer_size,
    )


_VISIBLE_DEVICES_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def resolve_device(index: int | None = None) -> jax.Device:
    """The device a run should use, chosen so that nothing ever grabs an accelerator the
    scheduler did not assign.

    - ``index`` given: ``jax.devices()[index]`` — the explicit choice always wins.
    - One device visible (a CPU-only machine, or a job the scheduler scoped to one
      card), or several visible but a ``*_VISIBLE_DEVICES`` variable set (Slurm's
      ``--gpus`` / ``--gres`` export ``CUDA_VISIBLE_DEVICES`` into the job): JAX's own
      default device, ``jax.devices()[0]`` — the first device of the default backend,
      where uncommitted arrays and jitted computations land anyway, so the loader's
      batches and the model agree without any further configuration.
    - Several accelerators visible and no such variable — a login node, an interactive
      shell on a shared machine: refuse, with the list, rather than pick one."""
    devices = jax.devices()
    if index is not None:
        return devices[index]
    if len(devices) == 1 or devices[0].platform == "cpu":
        return devices[0]
    if any(os.environ.get(v) for v in _VISIBLE_DEVICES_VARS):
        return devices[0]
    visible = [f"{d.platform}:{d.id} {d.device_kind}" for d in devices]
    raise ValueError(
        f"{len(devices)} accelerators are visible and none of {_VISIBLE_DEVICES_VARS} "
        "is set, so no scheduler scoped this process to a card; pass the device index "
        f"explicitly. Visible: {visible}"
    )
