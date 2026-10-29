"""Loggers for the trainer (roadmap C1, ADR-0009).

A ``Logger`` receives ``log(metrics, step)`` with scalar metrics, JAX arrays or
floats, and is closed at the end of training. Converting an on-device scalar to a
Python float blocks the host until that step's computation is done, so the loop never
does it directly: ``DelayedLogger`` holds one step of metrics and forwards the
previous step's, the JAX training cookbook's ``RecordWriter``, so that by the time the
values are read the current step has been enqueued and the device stays busy.
``MultiLogger`` fans out to several loggers, ``EpochLogger`` adds the epoch to each
record, and ``every`` on the concrete loggers thins the stream. Every logger is a
context manager whose exit calls ``close()``, also when the loop raised. ``NullLogger``
is for tests, ``ListLogger`` for tests and notebooks, ``PrintLogger`` for the terminal,
and ``WandbLogger`` wraps a ``wandb`` run that the caller created and owns.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any

import jax


Metrics = Mapping[str, Any]
"""Scalar metrics by name: JAX arrays or floats."""


class Logger(abc.ABC):
    """The base class: a subclass implements `log`. Every logger is a context manager
    whose exit calls `close`, a no-op unless overridden, also when the loop raised;
    `close` stays public for callers managing the lifetime themselves."""

    @abc.abstractmethod
    def log(self, metrics: Metrics, step: int) -> None:
        """Record `metrics` for `step`."""

    def close(self) -> None:
        """Release whatever the logger holds; a no-op by default."""
        pass

    def __enter__(self):  # beartype rejects typing.Self outside @beartype classes
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def to_floats(metrics: Metrics) -> dict[str, float]:
    """`metrics` as Python floats; this is where the host waits for the device."""
    return {k: float(v) for k, v in metrics.items()}


class NullLogger(Logger):
    """A logger that discards everything, for tests."""

    def log(self, metrics: Metrics, step: int) -> None:
        """Discard."""
        pass


class ListLogger(Logger):
    """A logger that keeps every ``(step, metrics)`` it receives in `records`, for
    tests and notebooks."""

    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, float]]] = []

    def log(self, metrics: Metrics, step: int) -> None:
        """Append ``(step, metrics)`` as floats."""
        self.records.append((step, to_floats(metrics)))


class PrintLogger(Logger):
    """A logger that prints one line per step to the terminal.

    **Arguments:**

    - `every`: Print every this many steps.
    - `keys`: The metrics to print, in order; all of them when ``None``.
    """

    def __init__(self, every: int = 1, keys: tuple[str, ...] | None = None) -> None:
        self.every, self.keys = every, keys

    def log(self, metrics: Metrics, step: int) -> None:
        """Print the selected metrics when `step` is a multiple of `every`."""
        if step % self.every:
            return
        values = to_floats(metrics)
        keys = self.keys or tuple(values)
        body = " | ".join(f"{k}: {values[k]:.4g}" for k in keys if k in values)
        print(f"step {step} | {body}")


class WandbLogger(Logger):
    """A logger that calls ``run.log(metrics, step=step)`` on a ``wandb`` run the
    caller created and will finish; `close` does not finish it.

    **Arguments:**

    - `run`: The ``wandb`` run.
    - `every`: Log every this many steps.
    """

    def __init__(self, run: Any, every: int = 1) -> None:
        self.run, self.every = run, every

    def log(self, metrics: Metrics, step: int) -> None:
        """Forward to the run when `step` is a multiple of `every`."""
        if step % self.every == 0:
            self.run.log(to_floats(metrics), step=step)


class DelayedLogger(Logger):
    """A logger that forwards each `log` call to `inner` one call later, so that the
    host reads a step's values only after the next step has been enqueued; `close`
    flushes the pending step."""

    def __init__(self, inner: Logger) -> None:
        self.inner = inner
        self._pending: tuple[Metrics, int] | None = None

    def log(self, metrics: Metrics, step: int) -> None:
        """Forward the pending call and hold this one."""
        if self._pending is not None:
            self.inner.log(*self._pending)
        self._pending = (metrics, step)

    def close(self) -> None:
        """Flush the pending call, then close `inner`."""
        if self._pending is not None:
            self.inner.log(*self._pending)
            self._pending = None
        self.inner.close()


class MultiLogger(Logger):
    """A logger that fans out to `loggers`, as in
    ``MultiLogger(PrintLogger(), WandbLogger(run))``, and closes all of them."""

    def __init__(self, *loggers: Logger) -> None:
        self.loggers = loggers

    def log(self, metrics: Metrics, step: int) -> None:
        """Forward to every logger."""
        for logger in self.loggers:
            logger.log(metrics, step)

    def close(self) -> None:
        """Close every logger."""
        for logger in self.loggers:
            logger.close()


class EpochLogger(Logger):
    """A logger that adds ``epoch = step / batches_per_epoch`` to every record before
    forwarding to `inner`: the run's unit is the step (ADR-0011 §4), and the epoch is
    reported alongside.

    **Raises:**

    `ValueError` for a non-positive `batches_per_epoch`."""

    def __init__(self, inner: Logger, batches_per_epoch: int) -> None:
        if batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive.")
        self.inner, self.batches_per_epoch = inner, batches_per_epoch

    def log(self, metrics: Metrics, step: int) -> None:
        """Forward with the epoch added."""
        self.inner.log({**metrics, "epoch": step / self.batches_per_epoch}, step)

    def close(self) -> None:
        """Close `inner`."""
        self.inner.close()


def block(metrics: Metrics) -> Metrics:
    """Wait for every metric to be computed and return them; for tests, never in the
    loop."""
    for v in metrics.values():
        if isinstance(v, jax.Array):
            v.block_until_ready()
    return metrics
