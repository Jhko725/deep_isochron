"""Isochrons of a planar system by AUTO-07p continuation (roadmap O7).

``codegen`` writes the AUTO problem files from a system's ``symbolic()`` view and needs
nothing but SymPy (ADR-0016). ``_env`` finds the built AUTO tree and imports AUTO's
Python layer on demand; it is the only module that imports ``auto``, so nothing in this
subpackage needs AUTO at import time. The pipeline (ADR-0017,
``docs/design/isochrons.md``) is four modules: ``results`` (the settings and the result
types), ``runner`` (an AUTO session and its runs), ``stages`` (the four numerical
stages) and ``foliation`` (the driver, ``isochron_foliation``).
"""

from .codegen import (
    auto_pars as auto_pars,
    AutoProblem as AutoProblem,
    constants_file as constants_file,
    generic_source as generic_source,
    Idx as Idx,
    indices_block as indices_block,
    internal_indices as internal_indices,
    parameter_map as parameter_map,
    rhs_block as rhs_block,
    write_problem as write_problem,
)
from .foliation import (
    isochron_foliation as isochron_foliation,
    provenance as provenance,
    trace_isochron as trace_isochron,
    wrap_phases as wrap_phases,
)
from .results import (
    Isochron as Isochron,
    IsochronFailure as IsochronFailure,
    IsochronFoliation as IsochronFoliation,
    IsochronSettings as IsochronSettings,
    STOP_REASONS as STOP_REASONS,
)
from .runner import (
    auto_session as auto_session,
    AutoSolution as AutoSolution,
    bvp as bvp,
    LOG_TAIL as LOG_TAIL,
    Runner as Runner,
)
from .stages import (
    adjoint_bundle as adjoint_bundle,
    bounding_box as bounding_box,
    isochron as isochron,
    periodic_orbit as periodic_orbit,
    rotate_phase as rotate_phase,
)
