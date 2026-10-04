"""Checkpointing of the whole ``TrainerState`` with Orbax (roadmap C4, ADR-0009).

``OrbaxCheckpointer(directory, ...)`` saves the **array leaves** of a ``TrainerState``
(model, optimizer state, step, key, schedule state) so a run can resume, and restores
them into a template state with ``eqx.combine``. Which steps are saved is Orbax's
``save_decision_policy`` (default: every ``save_every`` steps) and which are kept is its
``preservation_policy`` (default: the best by ``metric`` plus the latest). The directory
is given by the caller — the experiment script owns run directories (Phase D) — and the
``custom_metadata`` (resolved config, dataset ``config_hash``, ``x64`` flag) is whatever
the caller passes; nothing here invents paths or metadata.

``NullCheckpointer`` is for tests and notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, Protocol, runtime_checkable, TypeVar

import equinox as eqx
import jax
from orbax.checkpoint import v1 as ocp


S = TypeVar("S", bound=eqx.Module)


@runtime_checkable
class Checkpointer(Protocol):
    """``save(step, state, metrics)``; a context manager whose exit closes it."""

    def save(self, step: int, state: Any, metrics: Mapping[str, float]) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> "Checkpointer": ...

    def __exit__(self, *exc: object) -> None: ...


class _Closing:
    def close(self) -> None:
        pass

    def __enter__(self):  # beartype rejects typing.Self outside @beartype classes
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullCheckpointer(_Closing):
    def save(self, step: int, state: Any, metrics: Mapping[str, float]) -> None:
        pass


class OrbaxCheckpointer(_Closing):
    """See the module docstring. ``metric`` is the key of ``metrics`` the best
    checkpoint is chosen by (smaller is better unless ``maximize``)."""

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

    def save(self, step: int, state: Any, metrics: Mapping[str, float]) -> None:
        if self.metric is not None and self.metric not in metrics:
            raise KeyError(
                f"checkpoint metric {self.metric!r} missing from metrics "
                f"{sorted(metrics)}"
            )
        arrays = eqx.filter(state, eqx.is_array)
        json_metrics = cast(dict[str, Any], {k: float(v) for k, v in metrics.items()})
        self._ckptr.save_async(step, arrays, metrics=json_metrics)

    def restore(self, template: S, step: int | None = None) -> S:
        """The state at ``step`` (latest if ``None``) with ``template``'s static
        parts."""
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
        self._ckptr.wait()
        return sorted(c.step for c in self._ckptr.checkpoints)

    def close(self) -> None:
        self._ckptr.close()
