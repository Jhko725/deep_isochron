"""Isochrons of a planar system by AUTO-07p continuation (roadmap O7).

``codegen`` writes the AUTO problem files from a system's ``symbolic()`` view and needs
nothing but SymPy (ADR-0016). The continuation pipeline itself is the next branch
(ADR-0017, pending) and will load AUTO lazily through its runner, so nothing in this
subpackage imports AUTO at module level.
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
