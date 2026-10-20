"""ODE-side analysis: reference objects of a limit-cycle oscillator computed from its
vector field (roadmap Phase E, O1–O6). Results are fixed-shape ``eqx.Module`` pytrees
and the functions of them (``f(cycle, ...)``) are jit- and vmap-compatible and left
unjitted — the caller chooses the level; the host-side drivers (``find_limit_cycle``,
``fixed_point``) jit the stages they call.

O1: ``CrossSection`` (a hyperplane), ``PoincareMap`` (the first-return map on it,
differentiable in the point, with ``iterate`` and ``fixed_point``),
``multiple_shooting`` (the polish on the time-T flow) and ``find_limit_cycle`` (settle
→ return map → shooting → ``LimitCycle``, which samples itself), with its stages
exported for reuse. O2: ``floquet`` (multipliers and exponents of a ``LimitCycle`` by
the seeded periodic Schur chain, with the monodromy and Liouville routes as checks).
"""

from .floquet import (
    divergence_integral as divergence_integral,
    Floquet as Floquet,
    floquet as floquet,
    floquet_liouville as floquet_liouville,
    floquet_monodromy as floquet_monodromy,
    floquet_periodic_schur as floquet_periodic_schur,
    FloquetMethod as FloquetMethod,
    monodromy as monodromy,
    multipliers_from_monodromy as multipliers_from_monodromy,
    periodic_schur as periodic_schur,
    PeriodicSchur as PeriodicSchur,
    return_map_multiplier as return_map_multiplier,
    segment_jacobians as segment_jacobians,
)
from .limit_cycle import (
    find_limit_cycle as find_limit_cycle,
    initial_guess as initial_guess,
    is_equilibrium as is_equilibrium,
    LimitCycle as LimitCycle,
    orientation as orientation,
    period_consistent as period_consistent,
    polish as polish,
    settle as settle,
)
from .poincare import (
    Direction as Direction,
    FixedPoint as FixedPoint,
    FixedPointIteration as FixedPointIteration,
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
