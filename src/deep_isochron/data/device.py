"""Consuming a grain pipeline next to JAX: reading without a reader pool
(``single_threaded``), the prefetching transfer to the accelerator (``to_device``), and
the one sanctioned way to pick the device (``resolve_device``).

``docs/design/training-step-performance.md`` has the measurements behind the three.
"""

import os

import grain
import jax


def single_threaded(dataset: grain.MapDataset | grain.IterDataset) -> grain.IterDataset:
    """`dataset` as an ``IterDataset`` that reads with one thread and no read-ahead.

    Iterating a ``MapDataset`` directly starts grain's default reader, 16 threads with
    a 500-element buffer, which competes with a JAX training loop for the GIL. Use
    this, or `to_device`, which does the same, wherever a pipeline is consumed next to
    JAX work, such as the validation batches. An ``IterDataset`` is returned as it is.
    """
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
    """`dataset` with grain's two-stage prefetch to `device`, the pattern its JAX
    training tutorial recommends for real training ("option C").

    One CPU-side thread prepares `cpu_buffer_size` batches ahead while a second thread
    keeps `device_buffer_size` batches already transferred, so the transfer overlaps
    the training step instead of blocking the loop between steps; with the single
    reader thread that is three threads and no pool. Apply it after ``.batch(...)``.

    **Arguments:**

    - `dataset`: The batched pipeline.
    - `device`: The accelerator. It is explicit because on a shared machine the caller
        or the scheduler decides which card a job uses, never this function.
    - `cpu_buffer_size`, `device_buffer_size`: The two buffer depths.

    **Returns:**

    The prefetching ``IterDataset``."""
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
    """The device a run should use, chosen so that nothing ever grabs an accelerator
    the scheduler did not assign.

    With `index` given, the choice is ``jax.devices()[index]``. With one device
    visible, on a CPU-only machine or in a job the scheduler scoped to one card, or
    with several visible but a ``*_VISIBLE_DEVICES`` variable set (Slurm's ``--gpus``
    exports ``CUDA_VISIBLE_DEVICES`` into the job), the choice is JAX's own default
    device, ``jax.devices()[0]``, where uncommitted arrays and jitted computations land
    anyway, so the loader's batches and the model agree without further
    configuration.

    **Raises:**

    `ValueError`, listing the devices, when several accelerators are visible and no such
    variable is set, as on a login node or in an interactive shell on a shared machine.
    """
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
