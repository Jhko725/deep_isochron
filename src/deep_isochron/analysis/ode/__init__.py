"""ODE-side analysis: the reference objects of a limit-cycle oscillator, computed from
its vector field (roadmap Phase E, O1–O7).

O1 is the orbit: ``CrossSection``, ``PoincareMap`` (the first-return map on a section,
differentiable in the point, with ``iterate`` and ``fixed_point``),
``multiple_shooting``, and ``find_limit_cycle``, whose stages (settle, return map,
shooting) are exported for reuse and whose result ``LimitCycle`` samples itself;
``reanchor`` moves a cycle's phase origin to the orbit point that
``nearest_cycle_time`` finds nearest a given point. O2 is ``floquet``, the multipliers
and exponents of a ``LimitCycle`` by the seeded periodic Schur chain, with the
monodromy and Liouville routes as checks. O3 is ``asymptotic_phase``, the asymptotic
phase of a point by forward integration and a read-off on the cycle, and
``phase_grid``, its masked field on a grid. Results are fixed-shape ``eqx.Module``
pytrees; the functions of them, ``f(cycle, ...)``, are jit- and vmap-compatible and
left unjitted, and the host-side drivers ``find_limit_cycle``, ``reanchor``,
``fixed_point`` and ``phase_grid`` jit the stages they call. O7 is
``isochron_foliation``, the isochrons of a planar cycle by AUTO-07p continuation
(``auto``, ``docs/design/isochrons.md``), whose results are host-side dataclasses.
"""

from .auto.isochron import (
    Isochron as Isochron,
    isochron_foliation as isochron_foliation,
    IsochronFailure as IsochronFailure,
    IsochronFoliation as IsochronFoliation,
    IsochronSettings as IsochronSettings,
    wrap_phases as wrap_phases,
)
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
    nearest_cycle_time as nearest_cycle_time,
    orientation as orientation,
    period_consistent as period_consistent,
    polish as polish,
    reanchor as reanchor,
    settle as settle,
)
from .phase import (
    asymptotic_phase as asymptotic_phase,
    AsymptoticPhase as AsymptoticPhase,
    phase_grid as phase_grid,
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
