"""Loggers for the trainer (roadmap C1, ADR-0009).

A ``Logger`` receives ``log(metrics, step)`` with scalar metrics (JAX arrays or floats)
and is closed at the end of training. Converting an on-device scalar to a Python float
blocks the host until that step's computation is done (JAX, *Asynchronous dispatch*), so
the loop never does it directly:

- ``DelayedLogger(inner)`` holds one step of metrics and forwards the *previous* step's
  — the JAX training cookbook's ``RecordWriter``. By the time the previous step's values
  are read, the current step has already been enqueued, so the device stays busy while
  the host waits and loads the next batch; ``close()`` flushes the last step.
- ``every`` on the concrete loggers thins the stream (``log_every``).

``NullLogger`` is for tests, ``PrintLogger`` for the terminal, ``WandbLogger`` wraps a
``wandb`` run created by the caller (the script owns the run; Phase D)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax


Metrics = Mapping[str, Any]


@runtime_checkable
class Logger(Protocol):
    def log(self, metrics: Metrics, step: int) -> None: ...

    def close(self) -> None: ...


def to_floats(metrics: Metrics) -> dict[str, float]:
    """Device scalars to Python floats (this is the host sync)."""
    return {k: float(v) for k, v in metrics.items()}


class NullLogger:
    def log(self, metrics: Metrics, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class ListLogger:
    """Keeps every ``(step, metrics)`` it receives; for tests and notebooks."""

    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, float]]] = []

    def log(self, metrics: Metrics, step: int) -> None:
        self.records.append((step, to_floats(metrics)))

    def close(self) -> None:
        pass


class PrintLogger:
    def __init__(self, every: int = 1, keys: tuple[str, ...] | None = None) -> None:
        self.every, self.keys = every, keys

    def log(self, metrics: Metrics, step: int) -> None:
        if step % self.every:
            return
        values = to_floats(metrics)
        keys = self.keys or tuple(values)
        body = " | ".join(f"{k}: {values[k]:.4g}" for k in keys if k in values)
        print(f"step {step} | {body}")

    def close(self) -> None:
        pass


class WandbLogger:
    """``run.log(metrics, step=step)`` on a ``wandb`` run the caller created and will
    finish (``close`` does not finish it)."""

    def __init__(self, run: Any, every: int = 1) -> None:
        self.run, self.every = run, every

    def log(self, metrics: Metrics, step: int) -> None:
        if step % self.every == 0:
            self.run.log(to_floats(metrics), step=step)

    def close(self) -> None:
        pass


class DelayedLogger:
    """Forward each ``log`` call one call later (see the module docstring)."""

    def __init__(self, inner: Logger) -> None:
        self.inner = inner
        self._pending: tuple[Metrics, int] | None = None

    def log(self, metrics: Metrics, step: int) -> None:
        if self._pending is not None:
            self.inner.log(*self._pending)
        self._pending = (metrics, step)

    def close(self) -> None:
        if self._pending is not None:
            self.inner.log(*self._pending)
            self._pending = None
        self.inner.close()


def block(metrics: Metrics) -> Metrics:
    """Wait for every metric to be computed (tests; never in the loop)."""
    for v in metrics.values():
        if isinstance(v, jax.Array):
            v.block_until_ready()
    return metrics
