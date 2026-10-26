"""The training loop (roadmap C1, ADR-0009).

``Trainer`` holds what defines a step: the optimizer, the loss and the loss-weight
schedule. ``TrainerState`` holds everything that changes, the model, the optimizer
state, the step, the key and the schedule state, and is what checkpoints save.
``train_step`` is a pure jitted function of ``(state, batch)``, and ``train`` is the
loop: for each step, a batch from the loader, one ``train_step``, the metrics to the
logger, and every ``eval_every`` steps an evaluation whose metrics are logged and
handed to the checkpointer.

The loop never reads a device value itself: the logger decides when to sync
(``loggers.DelayedLogger``), and evaluation syncs only every ``eval_every`` steps. The
logger, the checkpointer and the evaluation function are injected, so tests run with
``NullLogger`` and ``NullCheckpointer`` and the experiment layer wires wandb, Orbax
and the run directory.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Mapping
from copy import replace
from functools import cached_property
from typing import Any, cast, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from .checkpoint import Checkpointer, NullCheckpointer
from .loggers import Logger, NullLogger
from .losses import AbstractLoss, Batch
from .schedules import AbstractLossSchedule, Constant


FilterSpec = Callable[[Any], bool]
"""A predicate on pytree leaves."""
M = TypeVar("M", bound=eqx.Module)
"""The model type."""
Metrics = dict[str, Float[Array, ""]]
"""The scalar metrics of one step, as device arrays."""


class TrainerState(eqx.Module, Generic[M]):
    """Everything that changes during training; the array leaves are what a
    checkpoint saves.

    **Attributes:**

    - `step`: The step count.
    - `model`: The model.
    - `opt_state`: The optimizer state over the trainable leaves.
    - `training_key`: The PRNG key, folded in with the step at every update.
    - `schedule_state`: The loss-weight schedule's state.
    - `optimizer`, `is_trainable`: Part of the recipe, static.
    """

    step: Int[Array, ""]
    model: M
    opt_state: optax.OptState
    training_key: PRNGKeyArray
    schedule_state: PyTree

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    is_trainable: FilterSpec = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        model: M,
        optimizer: optax.GradientTransformation,
        is_trainable: FilterSpec = eqx.is_inexact_array,
        schedule_state: PyTree = None,
        *,
        key: PRNGKeyArray,
    ) -> TrainerState[M]:
        """The state at step 0, with the optimizer initialized over the trainable
        leaves of `model`."""
        return cls(
            step=jnp.asarray(0, dtype=int),
            model=model,
            opt_state=optimizer.init(eqx.filter(model, is_trainable)),
            training_key=key,
            schedule_state=schedule_state,
            optimizer=optimizer,
            is_trainable=is_trainable,
        )

    def take_step(self, grads: M, schedule_state: PyTree = None) -> TrainerState[M]:
        """The state after one optimizer update from `grads`, filtered to the
        trainable leaves, with `step` and the key advanced and `schedule_state`
        stored."""
        grads = self.filter_trainable(grads)
        updates, opt_state = self.optimizer.update(
            cast(optax.Params, grads),
            self.opt_state,
            cast(optax.Params, self.filter_trainable(self.model)),
        )
        return replace(
            self,
            step=self.step + 1,
            model=eqx.apply_updates(self.model, updates),
            opt_state=opt_state,
            training_key=jax.random.fold_in(self.training_key, self.step),
            schedule_state=schedule_state,
        )

    def filter_trainable(self, pytree: M) -> M:
        """`pytree` with every non-trainable leaf replaced by ``None``."""
        return eqx.filter(pytree, self.is_trainable)


def _expected_nan(keys: list[str]) -> bool:
    """Whether every key in `keys` is a metric that is ``nan`` by design when its
    bucket is empty, ``val/mse_early`` or ``val/mse_late`` with no window on that side
    of ``t_split``; such a ``nan`` must not stop a run."""
    return all(k in ("val/mse_early", "val/mse_late") for k in keys)


def check_batch_dtype(batch: Batch, *, x64_enabled: bool | None = None) -> None:
    """Refuse float64 data when ``jax_enable_x64`` is off, since ``jnp.asarray`` would
    silently round it to float32 and every tolerance in the project assumes float64.

    **Raises:** `TypeError` for a float64 array in `batch` with x64 off.
    """
    if x64_enabled is None:
        x64_enabled = bool(jax.config.jax_enable_x64)
    for name, value in batch.items():
        if np.asarray(value).dtype == np.float64 and not x64_enabled:
            raise TypeError(
                f"batch[{name!r}] is float64 but jax_enable_x64 is off; enable it "
                "(jax.config.update('jax_enable_x64', True)) or generate float32 data."
            )


class Trainer(Generic[M]):
    """What defines a training step: the optimizer, the loss, the loss-weight
    schedule and the trainable-leaf predicate.

    **Arguments:**

    - `optimizer`: The optax optimizer.
    - `loss`: The loss.
    - `schedule`: The loss-weight schedule; `Constant` at the loss's default weights
        when ``None``.
    - `is_trainable`: Which leaves the optimizer updates; the inexact arrays by
        default.

    **Raises:** `ValueError` when the schedule's weight count differs from the loss's.
    """

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        loss: AbstractLoss,
        schedule: AbstractLossSchedule | None = None,
        *,
        is_trainable: FilterSpec = eqx.is_inexact_array,
    ) -> None:
        self.optimizer = optimizer
        self.loss = loss
        self.schedule = Constant(loss.default_weights) if schedule is None else schedule
        self.is_trainable = is_trainable
        n = len(self.schedule.weights(self.schedule.init(), jnp.asarray(0)))
        if n != loss.num_weights:
            raise ValueError(
                f"schedule produces {n} weights, loss {type(loss).__name__} expects "
                f"{loss.num_weights} ({loss.weight_names})."
            )

    def init(self, model: M, *, key: PRNGKeyArray) -> TrainerState[M]:
        """The initial state for `model`."""
        return TrainerState.init(
            model, self.optimizer, self.is_trainable, self.schedule.init(), key=key
        )

    @cached_property
    def train_step(
        self,
    ) -> Callable[[TrainerState[M], Batch], tuple[TrainerState[M], Metrics]]:
        """The jitted step, ``(state, batch) -> (state, metrics)``; compiled once per
        trainer."""
        return eqx.filter_jit(self._train_step)

    def _train_step(
        self, state: TrainerState[M], batch: Batch
    ) -> tuple[TrainerState[M], Metrics]:
        """One step: the weights from the schedule, the loss and its gradient, the
        schedule update, the optimizer update. The metrics are the loss, every term
        and the weights as ``w/<name>``."""
        weights = self.schedule.weights(state.schedule_state, state.step)
        model = eqx.nn.inference_mode(state.model, False)
        (loss, terms), grads = eqx.filter_value_and_grad(self.loss, has_aux=True)(
            model, batch, weights
        )
        schedule_state = self.schedule.update(state.schedule_state, state.step, terms)
        metrics: Metrics = {"loss": loss, **terms}
        for name, w in zip(self.loss.weight_names, weights):
            metrics[f"w/{name}"] = w
        return state.take_step(grads, schedule_state), metrics

    def train(
        self,
        model_or_state: M | TrainerState[M],
        train_loader: Iterable[Batch],
        *,
        num_steps: int | None = None,
        key: PRNGKeyArray | None = None,
        logger: Logger | None = None,
        checkpointer: Checkpointer | None = None,
        evaluate: Callable[[M], Mapping[str, Any]] | None = None,
        eval_every: int = 1,
        stop_on_nonfinite: bool = True,
    ) -> TrainerState[M]:
        """Run the loop until the loader ends, or for `num_steps` steps.

        A ``data.window_batches`` loader is finite and epoch-structured, so the data
        define the run; `num_steps` is an upper bound, and a loader that ends first
        stops the run with a warning. Batches are used as the loader yields them, so
        the transfer belongs in the loader (``data.to_device``); NumPy batches still
        work, transferred synchronously by ``jit``. The logger and the checkpointer
        are entered as context managers, so they close, and the ``DelayedLogger``
        flushes, even if a step raises.

        **Arguments:**

        - `model_or_state`: A model to start from, with `key`, or a restored
            ``TrainerState`` to resume from.
        - `train_loader`: The batches.
        - `num_steps`: The bound on the number of steps; ``None`` runs the loader out.
        - `key`: The PRNG key, required when starting from a model.
        - `logger`: The logger; `NullLogger` when ``None``.
        - `checkpointer`: The checkpointer; `NullCheckpointer` when ``None``.
        - `evaluate`: ``model -> validation metrics`` (``training.evaluation``). They
            are logged under their own keys and passed to ``checkpointer.save`` every
            `eval_every` steps, and the final step is evaluated and checkpointed
            regardless of the save policy; without `evaluate` there are no
            intermediate checkpoints.
        - `eval_every`: The evaluation interval in steps.
        - `stop_on_nonfinite`: Whether a non-finite validation metric stops the run
            with `FloatingPointError` before that state is checkpointed, so that the
            last checkpoint on disk is a finite one, the starting point for
            ``scripts/find_nan.py``. The check lives at the evaluations because the
            loop itself never reads device values.

        **Returns:** The final state.

        **Raises:** `ValueError` for a missing `key`, a non-positive `eval_every` or
        a negative `num_steps`; `FloatingPointError` as above.
        """
        if isinstance(model_or_state, TrainerState):
            state = model_or_state
        else:
            if key is None:
                raise ValueError("key is required when starting from a model.")
            state = self.init(model_or_state, key=key)
        if eval_every < 1:
            raise ValueError("eval_every must be a positive integer.")
        log: Logger = NullLogger() if logger is None else logger
        ckpt: Checkpointer = (
            NullCheckpointer() if checkpointer is None else checkpointer
        )

        saved_step = int(state.step) if isinstance(model_or_state, TrainerState) else -1

        def evaluate_and_save(step: int, *, force: bool = False) -> None:
            nonlocal saved_step
            assert evaluate is not None
            val_metrics = {k: float(v) for k, v in evaluate(state.model).items()}
            log.log(val_metrics, step)
            bad = sorted(k for k, v in val_metrics.items() if not math.isfinite(v))
            if stop_on_nonfinite and bad and not _expected_nan(bad):
                raise FloatingPointError(
                    f"non-finite validation metrics at step {step}: {bad}; the state "
                    f"was not checkpointed (last checkpoint: step {saved_step}). "
                    "Hunt it with scripts/find_nan.py from that checkpoint."
                )
            if ckpt.save(step, state, val_metrics, force=force):
                saved_step = step

        if num_steps is not None and num_steps < 0:
            raise ValueError("num_steps must be non-negative or None.")
        batches = iter(train_loader)
        step = int(state.step)  # Python-side counter: the device value is never read
        with log, ckpt:
            for i in itertools.count():
                if num_steps is not None and i >= num_steps:
                    break
                try:
                    batch = next(batches)
                except StopIteration:
                    if num_steps is not None:
                        warnings.warn(
                            f"the loader ended after {i} of {num_steps} steps.",
                            stacklevel=2,
                        )
                    break
                if i == 0:
                    check_batch_dtype(batch)
                state, metrics = self.train_step(state, batch)
                step += 1
                log.log(metrics, step)
                if evaluate is not None and step % eval_every == 0:
                    evaluate_and_save(step)
            # the final state is always evaluated and checkpointed (``force`` overrides
            # the save policy), unless the loop's last step already did both
            if step != saved_step:
                if evaluate is not None:
                    evaluate_and_save(step, force=True)
                else:
                    ckpt.save(step, state, {}, force=True)
        return state
