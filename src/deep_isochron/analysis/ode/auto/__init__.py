"""Isochrons by AUTO-07p continuation (roadmap; ADR-0016 for the code generation,
ADR-0017 — pending — for the pipeline). This subpackage never imports AUTO at module
level: ``codegen`` writes the problem files from a system's ``symbolic()`` view and
needs nothing but SymPy; the continuation pipeline (``isochron``, next branch) loads
AUTO lazily through its runner.
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
