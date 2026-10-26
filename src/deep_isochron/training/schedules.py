"""Loss-weight schedules, owned by the trainer (roadmap C2, ADR-0009).

A schedule turns ``(state, step)`` into the weight vector an ``AbstractLoss`` consumes
and updates its state from the step's loss terms, all inside the jitted training step,
so the state lives in ``TrainerState.schedule_state`` and survives a checkpoint. The
state's type is the schedule's type parameter: ``None`` for the stateless schedules, a
boolean scalar for the threshold switch. Three kinds cover what the project needs:
``Constant`` weights; ``StepSchedule``, a curriculum given by any traceable
``step -> weights`` function; and ``ThresholdSwitch``, the switch of Yawata et al.
2024, §IV.A, from one set of weights to another once every monitored term has dropped
below its threshold, once and for all.
"""

import abc
from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


S = TypeVar("S")
"""The schedule's state: ``None`` when stateless, or an array pytree, which is
checkpointed."""


class AbstractLossSchedule(eqx.Module, Generic[S]):
    """A loss-weight schedule: `weights` at a step from a state that `update` advances
    with the step's loss terms."""

    @abc.abstractmethod
    def init(self) -> S:
        """The state at step 0: ``None``, or array leaves only, since it is
        checkpointed."""

    @abc.abstractmethod
    def weights(self, state: S, step: Int[Array, ""]) -> Float[Array, " n"]:
        """The loss weights at `step`, shape ``(n,)``, in the order of the loss's
        ``weight_names``."""

    def update(self, state: S, step: Int[Array, ""], terms: Mapping[str, Array]) -> S:
        """The state after the step's loss `terms`; the identity by default."""
        return state


class Constant(AbstractLossSchedule[None]):
    """Fixed weights `values`; no state."""

    values: tuple[float, ...] = eqx.field(static=True)

    def init(self) -> None:
        """No state."""
        return None

    def weights(self, state: None, step):
        """`values`, as an array."""
        return jnp.asarray(self.values)


class StepSchedule(AbstractLossSchedule[None]):
    """A curriculum, ``weights = fn(step)`` with `fn` traceable: ``jnp`` arithmetic
    or optax schedules applied per weight, such as
    ``StepSchedule(lambda s: jnp.stack([1.0, optax.linear_schedule(0, 1, 1000)(s)]))``.
    No state."""

    fn: Callable[[Int[Array, ""]], Float[Array, " n"]] = eqx.field(static=True)

    def init(self) -> None:
        """No state."""
        return None

    def weights(self, state: None, step):
        """``fn(step)``, as an array."""
        return jnp.asarray(self.fn(step))


class ThresholdSwitch(AbstractLossSchedule[Bool[Array, ""]]):
    """The weights `before` until every term named in `thresholds` is below its
    threshold at the end of a step, and `after` from the next step on, permanently.

    **Arguments:**

    - `before`, `after`: The two weight tuples, of one length.
    - `thresholds`: Term name to threshold.

    **Raises:** `ValueError` when the two tuples differ in length.
    """

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
        """``False``: not yet switched."""
        return jnp.asarray(False)

    def weights(self, state: Bool[Array, ""], step):
        """`after` once switched, `before` until then."""
        return jnp.where(state, jnp.asarray(self.after), jnp.asarray(self.before))

    def update(self, state: Bool[Array, ""], step, terms) -> Bool[Array, ""]:
        """Switch when every monitored term is below its threshold; never unswitch."""
        below = jnp.asarray(True)
        for name, threshold in self.thresholds:
            below = below & (terms[name] < threshold)
        return state | below
