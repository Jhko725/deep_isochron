"""ODE-side analysis: reference objects of a limit-cycle oscillator computed from its
vector field (roadmap Phase E, O1–O6). Host-side results (NumPy dataclasses), JAX
inside; the stages are jitted, the drivers are not.

O1: ``CrossSection`` (a hyperplane), ``PoincareMap`` (the first-return map on it,
differentiable in the point, with ``iterate`` and ``fixed_point``),
``multiple_shooting`` (the polish on the time-T flow) and ``find_limit_cycle`` (settle
→ return map → shooting → ``LimitCycle``), with its stages exported for reuse.
"""

from .limit_cycle import (
    find_limit_cycle as find_limit_cycle,
    initial_guess as initial_guess,
    is_equilibrium as is_equilibrium,
    LimitCycle as LimitCycle,
    orientation as orientation,
    period_consistent as period_consistent,
    polish as polish,
    sample_orbit as sample_orbit,
    settle as settle,
)
from .poincare import (
    Direction as Direction,
    FixedPoint as FixedPoint,
    PoincareMap as PoincareMap,
)
from .sections import CrossSection as CrossSection
from .shooting import (
    multiple_shooting as multiple_shooting,
    shooting_residual as shooting_residual,
    SHOOTING_SOLVER_CONFIG as SHOOTING_SOLVER_CONFIG,
    ShootingArgs as ShootingArgs,
    ShootingResult as ShootingResult,
)
