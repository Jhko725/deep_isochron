"""Checkpointing of the whole ``TrainerState`` with Orbax (roadmap C4, ADR-0009).

``OrbaxCheckpointer`` saves the array leaves of a ``TrainerState`` (model, optimizer
state, step, key, schedule state) so a run can resume, and restores them into a
template state with ``eqx.combine``. Which steps are saved is Orbax's
``save_decision_policy``, every ``save_every`` steps by default, and which are kept is
its ``preservation_policy``, the best by ``metric`` plus the latest. The directory and
the ``custom_metadata`` are the caller's, since the experiment layer owns run
directories (ADR-0011); nothing here invents paths or metadata. ``NullCheckpointer``
is for tests and notebooks.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, TypeVar

import equinox as eqx
import jax
from orbax.checkpoint import v1 as ocp


S = TypeVar("S", bound=eqx.Module)


class Checkpointer(abc.ABC):
    """The base class: a subclass implements `save`.

    `save` is offered a state at every evaluation step; the checkpointer's save policy
    decides whether to write it, and the return value says whether it did. Every
    checkpointer is a context manager whose exit calls `close`, a no-op unless
    overridden, also when the loop raised.
    """

    @abc.abstractmethod
    def save(
        self,
        step: int,
        state: Any,
        metrics: Mapping[str, float],
        *,
        force: bool = False,
    ) -> bool:
        """Offer `state` at `step` with its validation `metrics`.

        `force` writes this call's checkpoint whatever the policy says. It is per
        call, not a mode: ``Trainer.train`` passes it exactly once, for the run's
        final step, so that a run always ends with its last state on disk, whatever
        ``save_every`` is.

        **Returns:**

        Whether a checkpoint was written."""

    def close(self) -> None:
        """Release whatever the checkpointer holds; a no-op by default."""
        pass

    def __enter__(self):  # beartype rejects typing.Self outside @beartype classes
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullCheckpointer(Checkpointer):
    """A checkpointer that never writes, for tests and notebooks."""

    def save(
        self,
        step: int,
        state: Any,
        metrics: Mapping[str, float],
        *,
        force: bool = False,
    ) -> bool:
        """Write nothing."""
        return False


class OrbaxCheckpointer(Checkpointer):
    """A checkpointer writing the array leaves of a state to `directory` with Orbax.

    **Arguments:**

    - `directory`: Where the checkpoints go; made absolute, as Orbax requires.
    - `save_every`: The save policy: a checkpoint every this many offered steps.
    - `metric`: The key of `metrics` the best checkpoint is chosen by, smaller being
        better unless `maximize`; ``None`` keeps only the latest.
    - `maximize`: Whether larger is better for `metric`.
    - `keep_best`: How many best checkpoints to keep, besides the latest.
    - `custom_metadata`: Recorded with the directory, such as the run's resolved
        config; read back by `custom_metadata`.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        save_every: int = 1,
        metric: str | None = "val/mse",
        maximize: bool = False,
        keep_best: int = 1,
        custom_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        policies: list[Any] = [ocp.training.preservation_policies.LatestN(1)]
        if metric is not None:
            policies.append(
                ocp.training.preservation_policies.BestN(
                    # BestN keeps the *last* n of the sorted metrics, so ascending
                    # order (reverse=False) keeps the largest; invert for minimization
                    get_metric_fn=lambda m: m[metric],
                    reverse=not maximize,
                    n=keep_best,
                )
            )
        self.directory = Path(directory).resolve()  # orbax wants absolute paths
        self.metric = metric
        self._ckptr = ocp.training.Checkpointer(
            str(self.directory),
            # orbax's own policies do not satisfy its protocols as typed (their
            # ``should_save``/``should_preserve`` signatures differ); type-level casts.
            save_decision_policy=cast(
                ocp.training.save_decision_policies.SaveDecisionPolicy,
                ocp.training.save_decision_policies.FixedIntervalPolicy(save_every),
            ),
            preservation_policy=cast(
                ocp.training.preservation_policies.PreservationPolicy,
                ocp.training.preservation_policies.AnyPreservationPolicy(policies),
            ),
            custom_metadata=dict(custom_metadata) if custom_metadata else None,
        )

    def save(
        self,
        step: int,
        state: Any,
        metrics: Mapping[str, float],
        *,
        force: bool = False,
    ) -> bool:
        """Offer `state` at `step`: written when `step` is a multiple of `save_every`,
        Orbax's ``FixedIntervalPolicy``, or unconditionally with `force`. Which
        checkpoints are kept afterwards is the preservation policy: the latest, plus
        the best `keep_best` by `metric`.

        **Returns:**

        Whether a checkpoint was written.

        **Raises:**

        `KeyError` when `metrics` lacks `metric`."""
        if self.metric is not None and self.metric not in metrics:
            raise KeyError(
                f"checkpoint metric {self.metric!r} missing from metrics "
                f"{sorted(metrics)}"
            )
        arrays = eqx.filter(state, eqx.is_array)
        json_metrics = cast(dict[str, Any], {k: float(v) for k, v in metrics.items()})
        response = self._ckptr.save_async(
            step, arrays, metrics=json_metrics, force=force
        )
        return response is not None

    def restore(self, template: S, step: int | None = None) -> S:
        """The state at `step`, the latest when ``None``, with the static parts of
        `template`."""
        arrays, static = eqx.partition(template, eqx.is_array)
        abstract = jax.tree.map(
            lambda a: jax.ShapeDtypeStruct(
                a.shape, a.dtype, sharding=getattr(a, "sharding", None)
            ),
            arrays,
        )
        loaded = self._ckptr.load(step, abstract)
        return eqx.combine(loaded, static)

    @property
    def steps(self) -> list[int]:
        """The saved steps, ascending, after pending writes complete."""
        self._ckptr.wait()
        return sorted(c.step for c in self._ckptr.checkpoints)

    @property
    def latest_step(self) -> int | None:
        """The most recent saved step, or ``None`` when the directory holds none."""
        self._ckptr.wait()
        latest = self._ckptr.latest
        return None if latest is None else int(latest.step)

    @property
    def custom_metadata(self) -> dict[str, Any]:
        """The ``custom_metadata`` the directory was created with, ``{}`` if none: the
        run's resolved config, for ``experiment.load_run``."""
        raw = self._ckptr.root_metadata().custom_metadata
        meta = cast(Mapping[str, Any] | None, raw)
        return dict(meta) if meta else {}

    def close(self) -> None:
        """Finish pending writes and close the Orbax checkpointer."""
        self._ckptr.close()
