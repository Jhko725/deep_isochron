"""The training loop (roadmap C1, ADR-0009).

``Trainer(optimizer, loss, schedule)`` holds what defines a step; ``TrainerState`` holds
everything that changes (model, optimizer state, step, key, schedule state) and is what
checkpoints save. ``train_step`` is a pure jitted function of ``(state, batch)``;
``train`` is the loop:

    for each step: batch = next(loader); state, metrics = train_step(state, batch)
                   log.log(metrics, step)
                   every eval_every steps: evaluate(state.model) -> val metrics,
                                           logged and handed to the checkpointer

The loop never reads a device value itself: the logger decides when to sync (see
``loggers.DelayedLogger``, the cookbook's one-step delay), and evaluation syncs only
every ``eval_every`` steps. Loggers, checkpointer and the evaluation function are
injected, so tests run with ``NullLogger``/``NullCheckpointer`` and the experiment
script (Phase D) wires wandb, Orbax and Hydra's run directory."""

from __future__ import annotations

import itertools
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
M = TypeVar("M", bound=eqx.Module)
Metrics = dict[str, Float[Array, ""]]


class TrainerState(eqx.Module, Generic[M]):
    """Everything that changes during training; array leaves are what a checkpoint
    saves. ``optimizer`` and ``is_trainable`` are static (part of the recipe)."""

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
        """Apply one optimizer update from ``grads`` (filtered to the trainable
        leaves); advance ``step`` and the key; store the schedule state."""
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
        return eqx.filter(pytree, self.is_trainable)


def check_batch_dtype(batch: Batch, *, x64_enabled: bool | None = None) -> None:
    """Refuse float64 data when ``jax_enable_x64`` is off: ``jnp.asarray`` would
    silently round it to float32 and every tolerance in the project assumes float64."""
    if x64_enabled is None:
        x64_enabled = bool(jax.config.jax_enable_x64)
    for name, value in batch.items():
        if np.asarray(value).dtype == np.float64 and not x64_enabled:
            raise TypeError(
                f"batch[{name!r}] is float64 but jax_enable_x64 is off; enable it "
                "(jax.config.update('jax_enable_x64', True)) or generate float32 data."
            )


class Trainer(Generic[M]):
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
        return TrainerState.init(
            model, self.optimizer, self.is_trainable, self.schedule.init(), key=key
        )

    @cached_property
    def train_step(
        self,
    ) -> Callable[[TrainerState[M], Batch], tuple[TrainerState[M], Metrics]]:
        return eqx.filter_jit(self._train_step)

    def _train_step(
        self, state: TrainerState[M], batch: Batch
    ) -> tuple[TrainerState[M], Metrics]:
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
    ) -> TrainerState[M]:
        """Run until the loader ends — a ``data.window_batches`` loader is finite and
        epoch-structured, so the data define the run — or for ``num_steps`` steps if
        given, which is an upper bound: a loader that ends first stops the run with a
        warning. ``evaluate(model)``
        returns validation metrics (``training.evaluation``); they are logged under
        their own keys and passed to ``checkpointer.save`` every ``eval_every`` steps
        and at the end (no intermediate checkpoints without ``evaluate``). Batches
        are used as the loader yields them: put the transfer in the loader
        (``data.to_device``, grain's two-stage prefetch); NumPy batches still work
        (``jit`` transfers them, synchronously). The logger and checkpointer are
        entered as context managers, so they close — and the ``DelayedLogger``
        flushes — even if a step raises. Resume by passing a restored
        ``TrainerState``."""
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

        def evaluate_and_save(step: int) -> None:
            assert evaluate is not None
            val_metrics = {k: float(v) for k, v in evaluate(state.model).items()}
            log.log(val_metrics, step)
            ckpt.save(step, state, val_metrics)

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
            if evaluate is not None and step % eval_every != 0:
                evaluate_and_save(step)
            elif evaluate is None:
                ckpt.save(step, state, {})
        return state
