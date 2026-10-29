"""Training: the loop, its losses and loss-weight schedules, loggers, checkpointing and
held-out evaluation (ADR-0009).

``Trainer`` holds what defines a step and ``TrainerState`` everything that changes; a
loss is an ``AbstractLoss`` of weighted terms, an ``AbstractLossSchedule`` supplies the
weights, a ``Logger`` receives the metrics, a ``Checkpointer`` is offered the state at
every evaluation, and an ``Evaluator`` computes the validation metrics.
"""

from .checkpoint import (
    Checkpointer as Checkpointer,
    NullCheckpointer as NullCheckpointer,
    OrbaxCheckpointer as OrbaxCheckpointer,
)
from .evaluation import collect_batches as collect_batches, Evaluator as Evaluator
from .loggers import (
    DelayedLogger as DelayedLogger,
    EpochLogger as EpochLogger,
    ListLogger as ListLogger,
    Logger as Logger,
    MultiLogger as MultiLogger,
    NullLogger as NullLogger,
    PrintLogger as PrintLogger,
    WandbLogger as WandbLogger,
)
from .losses import (
    AbstractLoss as AbstractLoss,
    ConjugacyTrajectoryLoss as ConjugacyTrajectoryLoss,
    PhaseAutoencoderLoss as PhaseAutoencoderLoss,
)
from .optimizers import decay_mask as decay_mask
from .schedules import (
    AbstractLossSchedule as AbstractLossSchedule,
    Constant as Constant,
    StepSchedule as StepSchedule,
    ThresholdSwitch as ThresholdSwitch,
)
from .trainer import Trainer as Trainer, TrainerState as TrainerState
