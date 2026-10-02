"""Trajectory data generation (placeholder — next phase of the roadmap).

Intended entry point::

    def generate(
        system: AbstractODE,
        ic_sampler: Callable[[PRNGKeyArray, int], Float[Array, "N dim"]],
        ts: Float[Array, " time"],
        *,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        rtol: float = 1e-8,
        atol: float = 1e-10,
        key: PRNGKeyArray,
    ) -> TimeSeriesDataSource: ...

returning a ``TimeSeriesDataSource`` carrying ``DatasetMetadata`` (system class and
parameters, solver and tolerances, time grid, IC sampler and seed, dtype, git SHA) so
that a saved dataset is self-describing. See the data-generation step of the project
plan.
"""
