from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jaxtyping import Array
from orbax.checkpoint._src.checkpoint_managers.preservation_policy import BestN

from .loaders import SegmentLoader


# TODO: better type signatures?
AbstractDynamicsLoss = Callable
AbstractDynamicsModel = Any


class BaseTrainer(ABC):
    optimizer: optax.GradientTransformation
    max_epochs: int
    savedir: Path
    savename: str
    logger: wandb.sdk.wandb_run.Run

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        max_epochs: int,
        savedir: Path | str,
        savename: str,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        self.savedir = Path(savedir)
        self.savename = savename

        self.logger = wandb.init(
            entity=wandb_entity, project=wandb_project, mode=wandb_mode
        )

    def train(
        self,
        model: AbstractDynamicsModel,
        loss_fn: AbstractDynamicsLoss,
        train_loader: SegmentLoader,
        validation_loader: SegmentLoader | None = None,
        args: Any = None,
        metric_fns: dict[str, Callable] | None = None,
        *,
        config: dict | None = None,
        **kwargs,
    ):
        step_fn = self.make_step_fn(
            loss_fn, train_loader, validation_loader, metric_fns
        )

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        if validation_loader is None:
            loader_state = (train_loader.init(), None)
            save_metric_name = "loss_train"
        else:
            loader_state = (train_loader.init(), validation_loader.init())
            save_metric_name = "loss_validation"

        ckpt_manager = ocp.CheckpointManager(
            (self.savedir / self.savename).resolve(),
            options=ocp.CheckpointManagerOptions(
                preservation_policy=BestN(
                    get_metric_fn=lambda metrics: metrics[save_metric_name],
                    reverse=True,
                    n=1,
                )
            ),
            metadata=config["model"],
        )

        with self.logger as logger:
            if config is not None:
                self.logger.config.update(config)

            with ckpt_manager as mngr:
                loss_history = []
                for step in range(self.max_epochs):
                    loss, log_dict, model_next, loader_state, opt_state = step_fn(
                        model, args, loader_state, opt_state
                    )
                    logger.log(log_dict, step=step)

                    print(f"{step=}, {loss=}")
                    loss_history.append(loss.item())
                    mngr.save(
                        step,
                        args=ocp.args.StandardSave(eqx.filter(model, eqx.is_array)),
                        metrics=log_dict,
                    )
                    model = model_next
        return model, np.asarray(loss_history)

    @abstractmethod
    def make_step_fn(
        self,
        loss_fn: AbstractDynamicsLoss,
        train_loader: SegmentLoader,
        validation_loader: SegmentLoader | None,
        metric_fn_dict: dict[str, Callable] | None,
    ) -> Callable: ...

    @property
    def savedir(self) -> Path:
        return self.__savedir

    @savedir.setter
    def savedir(self, value: Path | str):
        self.__savedir = Path(value)
        self.__savedir.mkdir(parents=True, exist_ok=True)

    @property
    def savepath(self) -> Path:
        return self.savedir / self.savename


class VanillaTrainer(BaseTrainer):
    optimizer: optax.GradientTransformation
    max_epochs: int
    savedir: Path
    savename: str
    logger: wandb.sdk.wandb_run.Run

    def __init__(
        self,
        optimizer: optax.GradientTransformation = optax.adabelief(1e-3),
        max_epochs: int = 5000,
        savedir: Path | str = "./results",
        savename: str = "checkpoint.eqx",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    ):
        super().__init__(
            optimizer,
            max_epochs,
            savedir,
            savename,
            wandb_entity,
            wandb_project,
            wandb_mode,
        )

    def make_step_fn(
        self,
        loss_fn: AbstractDynamicsLoss,
        train_loader: SegmentLoader,
        validation_loader: SegmentLoader | None,
        metric_fn_dict: dict[str, Callable] | None,
    ) -> Callable:
        loss_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        ## TODO: find better place to place this function; probably will need to
        # refactor class hierarchy
        if metric_fn_dict is None:

            def metric_fn(model_, batch, args_) -> dict[str, Array]:
                return dict()
        else:

            def metric_fn(model_, batch, args_) -> dict[str, Array]:
                return {
                    name: fn(model_, batch, args_)
                    for name, fn in metric_fn_dict.items()
                }

        @eqx.filter_jit
        def _step_fn(model_, args_, loader_states, opt_state):
            train_state, valid_state = loader_states
            # Train dataset
            batch_train, train_state = train_loader.load_batch(train_state)

            (loss, log_dict), grads = loss_grad_fn(model_, batch_train, args_)
            updates, opt_state = self.optimizer.update(
                grads, opt_state, eqx.filter(model_, eqx.is_inexact_array)
            )

            metrics_train = {
                f"{k}_train": v
                for k, v in metric_fn(model_, batch_train, args_).items()
            }
            log_dict = {f"{k}_train": v for k, v in log_dict.items()}
            log_dict = log_dict | metrics_train | {"loss_train": loss}
            # Validation dataset
            if validation_loader is not None:
                batch_valid, valid_state = validation_loader.load_batch(valid_state)
                loss_valid, log_dict_valid = loss_fn(model_, batch_valid, args_)
                metrics_valid = {
                    f"{k}_validation": v
                    for k, v in metric_fn(model_, batch_valid, args_).items()
                }
                log_dict_valid = {
                    f"{k}_validation": v for k, v in log_dict_valid.items()
                }
                log_dict_valid = (
                    log_dict_valid | metrics_valid | {"loss_validation": loss_valid}
                )
                log_dict = log_dict | log_dict_valid
            model_ = eqx.apply_updates(model_, updates)
            return loss, log_dict, model_, (train_state, valid_state), opt_state

        return _step_fn
