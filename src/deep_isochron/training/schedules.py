"""Loss-weight schedules, owned by the trainer (roadmap C2, ADR-0009).

A schedule turns ``(state, step)`` into the weight vector an ``AbstractLoss`` consumes
and updates its state from the step's metrics, all inside the jitted training step, so
the state lives in ``TrainerState`` and survives a checkpoint. Three kinds cover what
the project needs: constant weights; a step-dependent **curriculum** (any ``step ->
weights`` function, including optax schedules per weight); and Yawata et al.'s
**threshold switch** — one set of weights until every monitored term has dropped below
its threshold, then another, *once and for all* (their Sec. IV.A), which the per-batch
version in B12 approximated."""

import abc
from collections.abc import Callable, Mapping

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PyTree


class AbstractLossSchedule(eqx.Module):
    @abc.abstractmethod
    def init(self) -> PyTree:
        """The schedule's state at step 0 (array leaves only; it is checkpointed)."""

    @abc.abstractmethod
    def weights(self, state: PyTree, step: Int[Array, ""]) -> Float[Array, " n"]:
        """Loss weights to use at ``step``."""

    def update(
        self, state: PyTree, step: Int[Array, ""], metrics: Mapping[str, Array]
    ) -> PyTree:
        """State after seeing the step's loss terms; the identity by default."""
        return state


class Constant(AbstractLossSchedule):
    values: tuple[float, ...] = eqx.field(static=True)

    def init(self):
        return None

    def weights(self, state, step):
        return jnp.asarray(self.values)


class StepSchedule(AbstractLossSchedule):
    """Curriculum: ``weights = fn(step)`` with ``fn`` traceable (``jnp`` arithmetic or
    optax schedules applied per weight), e.g.
    ``StepSchedule(lambda s: jnp.stack([1.0, optax.linear_schedule(0, 1, 1000)(s)]))``.
    """

    fn: Callable[[Int[Array, ""]], Float[Array, " n"]] = eqx.field(static=True)

    def init(self):
        return None

    def weights(self, state, step):
        return jnp.asarray(self.fn(step))


class ThresholdSwitch(AbstractLossSchedule):
    """``before`` until every term named in ``thresholds`` is below its threshold at the
    end of a step, ``after`` from the next step on, permanently."""

    before: tuple[float, ...] = eqx.field(static=True)
    after: tuple[float, ...] = eqx.field(static=True)
    thresholds: tuple[tuple[str, float], ...] = eqx.field(static=True)

    def __init__(
        self,
        before: tuple[float, ...],
        after: tuple[float, ...],
        thresholds: Mapping[str, float],
    ):
        if len(before) != len(after):
            raise ValueError("before and after must have the same length.")
        self.before, self.after = tuple(before), tuple(after)
        self.thresholds = tuple(thresholds.items())

    def init(self) -> Bool[Array, ""]:
        return jnp.asarray(False)

    def weights(self, state, step):
        return jnp.where(state, jnp.asarray(self.after), jnp.asarray(self.before))

    def update(self, state, step, metrics):
        below = jnp.asarray(True)
        for name, threshold in self.thresholds:
            below = below & (metrics[name] < threshold)
        return state | below
