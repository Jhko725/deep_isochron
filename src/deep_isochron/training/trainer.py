import datetime
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from orbax.checkpoint import v1 as ocp


FilterSpec = Callable


# Inspired by levanter
class TrainerState[M: eqx.Module](eqx.Module):
    step: Int[Array, ""]
    model: M
    opt_state: optax.OptState
    training_key: PRNGKeyArray

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    is_trainable: FilterSpec = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        model: M,
        optimizer: optax.GradientTransformation,
        is_trainable: FilterSpec = eqx.is_inexact_array,
        *,
        key: PRNGKeyArray,
    ):
        # Making this as the __init__ will clash with the use of replace()
        opt_state = optimizer.init(
            eqx.filter(model, is_trainable)  # ty: ignore[invalid-argument-type]
        )

        return cls(
            step=jnp.asarray(0, dtype=int),
            model=model,
            opt_state=opt_state,
            training_key=key,
            optimizer=optimizer,
            is_trainable=is_trainable,
        )

    def take_step(
        self,
        grads: M,
    ) -> Self:
        """Given a pytree of model gradients, update model parameters using the
        appropriate optimizer update function."""
        grads = self.filter_trainable(grads)
        updates, opt_state_next = self.optimizer.update(
            grads,  # ty: ignore[invalid-argument-type]
            self.opt_state,
            self.filter_trainable(self.model),  # ty: ignore[invalid-argument-type]
        )
        model_next = eqx.apply_updates(self.model, updates)
        return replace(
            self,
            step=self.step + 1,
            model=model_next,
            opt_state=opt_state_next,
            training_key=jax.random.fold_in(self.training_key, self.step),
        )

    def filter_trainable(self, pytree: M) -> M:
        """Filter the given pytree to retain the trainable model parameters, as
        determined by self.is_trainable. Note that this function works on all pytrees
        that are compatible with self.model."""
        return eqx.filter(pytree, self.is_trainable)


Batch = PyTree
M = TypeVar("M", bound=eqx.Module)


class Trainer:
    loss_fn: Callable
    optimizer: optax.GradientTransformation
    checkpoint_path: Path
    wandb_kwargs: dict[str, Any]

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        checkpoint_dir: str | Path,
        checkpoint_name: str | None,
        wandb_kwargs: dict[str, Any],
        config_dict: dict[str, Any],
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_path = self._make_checkpoint_path(
            checkpoint_dir, checkpoint_name
        )
        self.wandb_kwargs = wandb_kwargs
        self.config_dict = config_dict

    def _make_checkpoint_path(
        self, checkpoint_dir: str | Path, checkpoint_name: str | None
    ) -> Path:
        """Given checkpoint directory and name, return the absolute path to save
        checkpoints in."""
        now = datetime.datetime.now()
        now_str = now.strftime("%y-%m-%d-%H_%M_%S")
        if checkpoint_name is None:
            checkpoint_name = ""

        checkpoint_path = Path(checkpoint_dir) / checkpoint_name / now_str
        # orbax.checkpoint does not like relative paths
        checkpoint_path = checkpoint_path.resolve()
        return checkpoint_path

    def train(
        self,
        model: M,
        train_dataloader: Iterable[Batch],
        loss_args: Any = None,
        *,
        trainable_filterspec: FilterSpec = eqx.is_inexact_array,
        num_steps: int = 5000,
        seed: int = 0,
    ):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        state = TrainerState[M].init(
            model, self.optimizer, trainable_filterspec, key=jax.random.key(seed)
        )
        train_dataiter = iter(train_dataloader)

        save_metric = "train_loss"

        with ExitStack() as stack:
            logger = stack.enter_context(
                wandb.init(config=self.config_dict, **self.wandb_kwargs)
            )
            ckptr = stack.enter_context(
                ocp.training.Checkpointer(
                    self.checkpoint_path,
                    preservation_policy=ocp.training.preservation_policies.BestN(
                        get_metric_fn=lambda metrics: metrics[save_metric],
                        reverse=True,
                        n=1,
                    ),
                    custom_metadata=self.config_dict["model"],
                )
            )  # add preservation policy, custom_metadata

            state_prev, outputs_prev = None, None

            for _ in range(num_steps):
                try:
                    batch = next(train_dataiter)
                except StopIteration:
                    break

                state_next, loss, metrics = self.train_step(state, batch, loss_args)

                output = {"train_loss": loss} | metrics

                if (outputs_prev is not None) and (state_prev is not None):
                    step_log = int(state_prev.step)
                    outputs_prev = jax.tree.map(lambda x: float(x), outputs_prev)
                    logger.log(outputs_prev, step=step_log)
                    print(
                        f"""Step: {step_log} | Train loss: {outputs_prev["train_loss"]}"""
                    )
                    weights = eqx.filter(state_prev.model, eqx.is_array)
                    ckptr.save(step_log, weights, metrics=outputs_prev)

                outputs_prev = output
                state, state_prev = state_next, state

            step_log = int(state_prev.step)
            outputs_prev = jax.tree.map(lambda x: float(x), outputs_prev)
            logger.log(outputs_prev, step=step_log)
            print(f"""Step: {step_log} | Train loss: {outputs_prev["train_loss"]}""")
            weights = eqx.filter(state_prev.model, eqx.is_array)
            ckptr.save(step_log, weights, metrics=outputs_prev)
            return state.model

    @cached_property
    def train_step(
        self,
    ) -> Callable[
        [TrainerState[M], Batch, Batch | None, Any],
        tuple[M, Float[Array, ""], dict[str, Array]],
    ]:
        return eqx.filter_jit(self._train_step)

    def _train_step(
        self, state: TrainerState[M], batch: Batch, args
    ) -> tuple[M, Float[Array, ""], dict[str, Array]]:
        model = eqx.nn.inference_mode(state.model, False)

        loss_grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)
        (loss, metrics), grads = loss_grad_fn(model, batch, args)
        state_next = state.take_step(grads)
        return state_next, loss, metrics
