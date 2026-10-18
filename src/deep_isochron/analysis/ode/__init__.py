"""ODE-side analysis: reference objects of a limit-cycle oscillator computed from its
vector field (roadmap Phase E, O1–O6). Host-side results (NumPy dataclasses), JAX
inside.

O1: ``Hyperplane`` (a cross-section), ``PoincareMap`` (the first-return map on it, with
``fixed_point`` and ``jacobian``), ``multiple_shooting`` (the polish on the time-T flow)
and ``limit_cycle`` (settle → return map → shooting → ``LimitCycle``).
"""

from .limit_cycle import limit_cycle as limit_cycle, LimitCycle as LimitCycle
from .poincare import (
    Direction as Direction,
    FixedPoint as FixedPoint,
    PoincareMap as PoincareMap,
)
from .sections import Hyperplane as Hyperplane
from .shooting import (
    multiple_shooting as multiple_shooting,
    segment_flow as segment_flow,
    shooting_residual as shooting_residual,
    SHOOTING_SOLVER_CONFIG as SHOOTING_SOLVER_CONFIG,
    ShootingResult as ShootingResult,
)
