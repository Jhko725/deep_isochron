# 2026-10-02 — `data-generation`

Phase B of `docs/roadmap.md`: the ODE layer and the data layer, as decided on 2026-10-02
(ADR-0007, ADR-0008).

## Summary

The systems package gets its intended two-layer hierarchy — `AbstractODE` (any ODE; `rhs`,
vmappable `flow` with a `SolverConfig`) and `AbstractNormalForm` (planar oscillators with
closed-form phase, isostable amplitude, isochrons, Floquet exponent, polar chart) — and the
Bautin/Hopf forms are integrable by pluggable strategies (cartesian, polar, `r²`), all in
cartesian coordinates. The data package becomes xarray-backed: one netCDF4 file per dataset
carrying arrays and `DatasetMetadata`, `generate()` vmapped over initial conditions with loud
failure reporting, time/trajectory splits, a weighted-window sampler for transient
oversampling alongside the existing `mix` design, and a Hydra entry point with two configs.
Two new test files pin the closed forms exactly (autodiff of the defining identities) and
the data layer end to end.

## Files

Systems:

- `src/deep_isochron/systems/base.py` — `SolverConfig` (static, leafless), `diffeqsolve`
  helper, `AbstractODE.flow`/`flow_result` (replaces `solve(**kwargs)`); module docstring
  states the two-layer rule.
- `src/deep_isochron/systems/normal_form.py` — new: `AbstractNormalForm` (`radial_rate`,
  `angular_rate`, `floquet_exponent` by autodiff, `phase_shift`, `isostable`, `phase`,
  `amplitude`, `limit_cycle`, `isochron`, `to_chart`/`from_chart`, cartesian `rhs` and
  `rhs_polar`, `flow(..., strategy=)`).
- `src/deep_isochron/systems/strategies.py` — new: `AbstractFlowStrategy`,
  `CartesianIntegration`, `PolarIntegration`, `RadiusSquaredIntegration`, `STRATEGIES`,
  `resolve_strategy`.
- `src/deep_isochron/systems/normal_forms.py` — `HopfNormalForm`/`BautinNormalForm`
  rewritten on the two rates with closed forms; parameters as unconstrained leaves
  (`Positive`); `w0` defaults to `w`; `b > −1`; Bautin's `solve` removed (now a strategy).
- `src/deep_isochron/systems/__init__.py` — new: public exports.
- `src/deep_isochron/model/conjugacy.py` — `ConjugateLatentDynamics(latent, bijection,
  solver_config)`: integrates via `flow` in cartesian coordinates; no polar conversion; no
  hard-coded solver settings (Phase C's C2, done early).
- `src/deep_isochron/model/latent_dynamics.py` — `HopfLatentDynamics` deleted;
  `AbstractLatentDynamics` docstring scopes it to the autoencoder baseline.

Data:

- `src/deep_isochron/data/dataset.py` — rewritten: `DatasetMetadata` (frozen dataclass,
  `to_attrs`/`from_attrs` with JSON-encoded nested fields), xarray-backed
  `TimeSeriesDataSource` (windows, `window_start_times`, `split_time`,
  `split_trajectories`, `save`/`load` with dtype check, `from_xarray`).
- `src/deep_isochron/data/generate.py` — implemented: `AbstractICSampler`, `UniformBox`,
  `UniformAnnulus`, `generate(system, ic_sampler, ts, n, *, seed, config, strategy,
  window_size, extra)`, `config_hash`, `dataset_path`, git/package provenance.
- `src/deep_isochron/data/sampling.py` — new: `transient_weights`, `weighted_windows`,
  `mixed_split`.
- `src/deep_isochron/data/__init__.py` — exports.
- `scripts/generate_data.py` — new Hydra entry point; `configs/data/{fhn,bautin}.yaml`.
- `pyproject.toml` — `xarray`, `h5netcdf` runtime dependencies.
- `prototype.ipynb` — normal-form cells use `flow`/`to_chart`; model cell passes a
  `SolverConfig` preserving the former Kvaerno5 / 1e-4 / 1e-6 / 8192 settings.

Tests and docs:

- `tests/test_systems.py` — new (see Tests).
- `tests/test_data.py` — new (see Tests).
- `tests/helpers.py` — `TOL["flow"] = 1e-6`.
- Review round 2: `src/deep_isochron/data/dataset.py` (rewritten), `data/windows.py`
  (new; replaces `data/sampling.py`), `data/generate.py` (grouped metadata),
  `data/__init__.py`, `training/losses.py` (dict batch), `training/__init__.py`,
  `tests/test_data.py` (rewritten), `configs/data/*.yaml`, `scripts/generate_data.py`
  (no `window_size`), `prototype.ipynb`, ADR-0008 amended, `docs/architecture.md`.
- Review round 1: `src/deep_isochron/model/invertible/constraints.py` (`GreaterThan`,
  `at_zero` documentation, `BoundedPositive.lower`), `analytic.py`,
  `splines/rational_quadratic.py` (call sites), `tests/test_constraints.py`;
  `systems/normal_forms/{__init__,base,hopf,bautin,integration}.py` replace
  `normal_form.py`, `normal_forms.py`, `strategies.py`; `systems/base.py` (`params()`,
  single `flow`); `data/generate.py` (`params()`, `integration`); `configs/data/*.yaml`,
  `scripts/generate_data.py` (`integration:`); `tests/test_normal_forms.py` (new),
  `tests/test_systems.py` (facts only); `docs/decisions/0007` amended.
- `docs/decisions/0007-systems-hierarchy-and-flow-strategies.md`,
  `docs/decisions/0008-dataset-format-and-sampling.md` — new ADRs.
- `docs/architecture.md` — systems and data sections rewritten; module map updated.
- `docs/roadmap.md` — Phase B rewritten around the decisions; B1–B7 to Done; B8
  (`analysis/`) added; parked items.

## Review round 1 (2026-10-03) — what changed in response

Every row of the first *Review notes* table was answered as follows (the design points
deferred to discussion were settled on 2026-10-03 and are tracked as roadmap B9–B12):

- **`flow`/`flow_result` → one `flow -> diffrax.Solution`.** Integrations that change
  chart return the same `Solution` with `.ys` replaced via `eqx.tree_at`, so `.result`,
  `.stats`, `.ts` are diffrax's and `.ys` is always cartesian. Callers read `.ys`.
- **Names.** `strategy` → `integration` everywhere (`systems/normal_forms/integration.py`,
  `AbstractFlowIntegration`, `INTEGRATIONS`, `resolve_integration`, the `flow` keyword,
  the Hydra key); the first argument is `normal_form: AbstractNormalForm`. Reason beyond
  taste: `tests/strategies.py` already means Hypothesis strategies.
- **Subpackage.** `systems/normal_form.py` + `normal_forms.py` → `systems/normal_forms/
  {base, hopf, bautin, integration}.py`; the integrations live there because they use
  `radial_rate`/`angular_rate`/`rhs_polar`, which only normal forms have.
- **`AbstractODE.params()`** is part of the contract: constrained parameter values keyed
  by mathematical name (default: every non-static field; the normal forms override it).
  `generate` records `system.params()`; the former `_system_params` and its `k[4:]`
  attribute-name inspection are gone. `SolverConfig.params()` likewise.
- **`GreaterThan(lower, at_zero=None)`** replaces `Positive(eps, at_zero)`; the default
  `at_zero = lower + 1` makes `GreaterThan(0.0)` map `0 ↦ 1` and `GreaterThan(−1.0)` map
  `0 ↦ 0`, which is exactly what the two normal-form parameters need, so the module-level
  `A_CONSTRAINT`/`B_CONSTRAINT` read without a comment. `Positive(at_zero=1.0)` stays as
  `GreaterThan(0.0, at_zero)`; `BoundedPositive` gets the same default and `eps` → `lower`.
  The constraints module docstring now has a paragraph on the role of `at_zero`.
- **Tests split.** `test_normal_forms.py` (generic laws, Hypothesis over both forms) and
  `test_systems.py` (flow machinery; facts about the observed systems). The FHN facts cite
  Langfield, Krauskopf & Osinga (2014), Sec. III, whose parameters `a = 0.7, b = 0.8, c =
  3, z = −0.4` are this class's defaults: equilibrium `(0.9066, −0.2582)`, eigenvalues
  `0.1339 ± 0.9163i`, and — new — the period `T_Γ ≈ 11.2279`, measured from zero
  crossings after the transient and reproduced to `2e-3`.
- **Deferred to B9** (same branch, next commit): whole-trajectory source with grain
  `RandomMap` windowing (swirl-dynamics pattern), `copy.replace` on a frozen dataclass,
  grouped metadata. **B10/B11**: the normal-form design note, then the `r`-based API.
  **B12**: the Yawata et al. (2024) baseline.

## Review round 2 (2026-10-03) — data layer as grain transforms (B9)

- **Whole-trajectory source.** `TimeSeriesDataSource` is a frozen dataclass `(ts, ys,
  metadata)`; `len` = trajectories, `source[i] = {"t", "u"}`. All window bookkeeping
  (`window_size`, `windows_per_trajectory`, `window_index`, `window_start_times`, the
  per-half window handling in `split_time`) is gone; splits are `copy.replace`.
- **Windows are `grain.transforms.RandomMap`s** (`data/windows.py`): `RandomWindow(length,
  start_range)` and `WeightedWindow(length, weight)`; `windows()` builds
  `source → shuffle → repeat → random_map`, `mixed_windows()` is `mix` of two start-range
  restricted `RandomWindow` pipelines. Verified that grain keys the per-element generator
  by the global index, so a repeated trajectory gets a fresh window every epoch
  (`test_weighted_window_frequencies_and_determinism`). The DySLIM pattern
  (`swirl_dynamics/projects/ergodic/utils.py`: `ArrayDataSource` + `RandomSection`).
- **Grouped metadata**: `SystemSpec`, `SamplingSpec`, `GridSpec`, `SolveSpec`,
  `Provenance`, `extra`; one JSON attribute per group; `config_hash` a property over the
  first four. `SolveSpec` is built from `SolverConfig.params()` + the integration name.
- **Dict batches** `{"t", "u"}`; `ConjugacyTrajectoryLoss` reads by key.
  `TimeSeriesDataSource` is no longer re-exported from `training`.
- Notebook data cells rewritten on the new API (`windows` with `start_range` for the two
  halves, `transient_weight`, `mixed_windows`; `batch["u"]`).

## Design

The decisions and their rejected alternatives are in ADR-0007 (hierarchy, `SolverConfig`,
strategies as objects, parameter constraints) and ADR-0008 (netCDF via xarray, no DVC,
weighted windows alongside `mix`). Points specific to this branch:

- **Closed forms were derived, then verified two ways before being written down.** For
  `ṙ = rρ(r²)`, `θ̇ = ω(r²)`: the asymptotic phase is `φ = θ + h(r)` with `h' = (ω(1) −
  ω(r²))/(rρ(r²))`, and the isostable `ψ` satisfies `ψ'/ψ = κ/(rρ(r²))`, `κ = 2ρ'(1)`. For
  Bautin, `h(r) = c[ln r − ½ ln((1+br²)/(1+b))]`, `ψ(r) = (1−r²)(1+br²)^b / r^{2(1+b)}`,
  `c = (w−w₀)/a`; Hopf is the `b = 0` case. The identities hold to 1e-15 under autodiff
  (`test_phase_and_isostable_identities`) and the integrated versions to 1e-6
  (`test_phase_and_amplitude_along_the_flow`). The `(1+b)` normalisation of `h` is what
  makes `h(1) = 0`, i.e. the phase on the cycle equals the polar angle.
- **`floquet_exponent` is concrete on the base class** (`2·grad(radial_rate)(1)`), and the
  subclasses' closed forms are checked against it — one less thing a new normal form can
  get wrong.
- **`b ∈ (−1, 0)` is allowed but documented.** The first Hypothesis run found it: with
  `b = −0.5` and `r₀ = 2` the radius blows up, because `ρ(s) = a(1−s)(1+bs)` has a second
  zero at `s = −1/b` — an unstable cycle, the genuine Bautin two-cycle regime. The
  stability constraint `b > −1` is kept (it is the mathematically right one); the flow
  laws are tested for `b ≥ 0`, and `test_bautin_negative_b_has_unstable_outer_cycle` pins
  the outer cycle's location and instability.
- **`generate` raises on any failed integration** rather than dropping trajectories: a
  dataset with silently missing initial conditions would not match its metadata's
  `n_trajectories` and sampler, defeating the one-file guarantee. The message lists the
  failing indices.
- **`config_hash` excludes provenance.** Regenerating an unchanged config must land on the
  same filename; `git_sha`, `created`, `package_version` are recorded but not hashed.
- **`__getitem__(self, index)`** — the parameter is named to match grain's
  `RandomAccessDataSource` protocol; `ty` checks protocol conformance by parameter name.
- **`typing.Self` avoided** in `dataset.py` (string forward references instead): the
  beartype import hook rejects PEP 673 `Self` outside `@beartype`-decorated classes, as
  found on the first branch.

ADRs: [0007](../decisions/0007-systems-hierarchy-and-flow-strategies.md),
[0008](../decisions/0008-dataset-format-and-sampling.md); builds on
[0001](../decisions/0001-unconstrained-leaves.md), [0004](../decisions/0004-abstractvar-fields.md).

## Bugs fixed

- `HopfNormalForm(w0=None)` reported eigenvalues at the origin with zero imaginary part
  while integrating with `θ̇ = w`; `w0` now defaults to `w` and `eigenvalues_origin` uses
  it.
- `BautinNormalForm` accepted `b = 0` at construction and then stored `log(0) = −inf`;
  parameters are now constrained leaves with `b = 0` a regular value.
- `HopfLatentDynamics` called `HopfNormalForm` as a function (would raise `TypeError`);
  deleted rather than fixed, as decided.
- `AbstractODE.solve` silently swallowed unknown keyword arguments through `**kwargs`
  (a misspelt `rtol` was ignored); `flow` has no `**kwargs`.
- `TimeSeriesDataSource.split` returned a `tuple` of arbitrary length for its
  `tuple[Self, Self]` annotation and silently produced empty sources for out-of-range
  indices; `split_time` validates and the window size is handled per half.

## Tests

`uv run pytest -n 4`; the two new files take ~1 min at the default profile.

`tests/test_systems.py` (19 tests): the phase/isostable identities exactly by autodiff and
again by integration; `h(1) = ψ(1) = ρ(1) = 0`; `floquet_exponent` closed form vs autodiff;
cartesian `rhs` equals `rhs_polar` pushed through the chart; `isochron`/`limit_cycle`
consistency; Hopf equals Bautin(b=0); the outer cycle for `b < 0`; the three strategies
agree to `TOL["flow"]` and start at `u0`; cartesian is finite (value and Jacobian) at the
origin; unknown strategy name rejected; strategies are static under `filter_jit` (one trace
each, no retrace); `filter_vmap` over initial conditions equals a loop; `SolverConfig` is
honoured, hashable and leafless; `flow_result` reports failure with `throw=False`;
FitzHugh–Nagumo fixed point `(0.9066, −0.2582)` with eigenvalues `0.1339 ± 0.9163i`
(independently computed with scipy); Hodgkin–Huxley gating variables stay in `[0, 1]` and
the neuron spikes at `I = 30`.

`tests/test_data.py` (14 tests): window enumeration is trajectory-major and every window is
the right slice (Hypothesis over `n`, `T`, window); invalid shapes/windows rejected;
`split_time` partitions and handles a half shorter than the window; `split_trajectories`
partitions, is seeded, rejects `frac ∉ (0, 1)`; save → load round-trips arrays and every
metadata field (nested dicts included) through one file; dtype mismatch raises; save without
metadata raises; `generate` on Bautin: shapes, `u0 == ys[:, 0]`, metadata fields, trajectories
equal the system's flow, `config_hash` stable across runs and changed by the seed, filename
scheme; `generate` on FHN, `strategy` rejected for a non-normal-form, failures raise with
indices; `transient_weights` shape/decay/validation; `weighted_windows` empirical frequencies
match the weights within 3 % over 4000 draws and the stream is seed-deterministic; weight
validation; `mixed_split` interleaves in ratio and rejects splits that do not fit a window.

Also run by hand: `scripts/generate_data.py --config-name {bautin,fhn}` writes
`<name>-<hash>.nc`; reloading gives the metadata including the resolved Hydra config, and
`weighted_windows(...).batch(64)` yields `(64, W)`, `(64, W, 2)` batches.

## Open issues

- `deep_isochron.analysis` (numerical limit cycle, monodromy, asymptotic phase for any
  `AbstractODE`) is the next item (roadmap B8); until then the observed systems have no
  ground-truth isochrons in code.
- `t_settle` per trajectory (a physical basis for sampling weights) parked with B8.
- `scripts/training/` (the earlier autoencoder training script and its configs) untouched;
  Phase D decides whether it is folded into `scripts/train.py`.
- `uv.lock` not regenerated here (same reason as the previous branch); `uv sync` will add
  `xarray`/`h5netcdf`.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/deep_isochron/systems/base.py`: `SolverConfig`; `flow`/`flow_result` replace `solve` | Similar concerns raised in `normal_form.py`: no need to keep both `flow` and `flow_result`. | Deferred to the next discussion with Claude. |
| `src/deep_isochron/systems/normal_form.py`: `AbstractNormalForm` with closed-form phase/amplitude/isochrons and the chart | `flow` is too thin a wrapper around `flow_result` and should be removed. If we need `diffrax.diffeqsolve`'s metadata in `sol`, then may be better to return the entire `sol` instead of cherrypicking `sol.ys` and `sol.result`. Additional relevant concerns are raised below in the entry for `normal_forms.py`| Deferred to the next discussion with Claude. |
| `src/deep_isochron/systems/strategies.py`: cartesian / polar / `r²` integration strategies | Like the strategies. But couple things regarding the naming. <ul><li>Argument name `nf` is not clear. Either type annotate to `AbstractNormalForm` (or `AbstractODE`, in which case the current variable name is even more misleading) and/or change the variable name to `normal_form`. <li>Don't like the word "strategy" here. In fact, `tests/` already contains `strategies.py`, referring to the strategies used by `hypothesis`. The child classes are named as {...}Integration, and I think both `AbstractFlowStrategy` and `strategies.py` should be renamed similarly. For example, `integration.py` and `AbstractFlowIntegration`.<ul> | Deferred to the next discussion with Claude for refinement. |
| `src/deep_isochron/systems/normal_forms.py`: Hopf/Bautin on the two rates; constrained leaves; `b > −1` | <ul><li>Having both `normal_forms.py` and `normal_form.py` is confusing. I think it is better to create a submodule `normal_forms`, then `normal_form.py` → `normal_forms/base.py`, and Hopf and Bautin into `normal_forms/hopf.py` and `normal_forms/bautin.py` respectively. Depending on whether the integration strategies in `strategies.py` (to be renamed; see above) pertain only to the normal forms or to all `AbstractODE`s, this could also be moved into the submodule. <li> Some properties are written in terms of `r`, others in terms of `s`(=r**2). This is inconsistent and needs to be unified to `r`. Also, need proper design document on the derivation of the different quantities. This part will have to be separated into another step in phase B, where the design document with the proper conventions and mathematics is laid down first, and the implementaion following next. <li>`_positive_a = Positive(0.0, at_zero=1.0)` is difficult to understand before checking out the implementation. I think it is worth reworking the `Positive` constraint class so that this becomes `GreaterThan(ref=1.0)`, and `Positive` be an alias for `GreaterThan(ref=0.0)`. `_shifted_b = Positive(-1.0, at_zero=0.0)` is similarly not immediately clear and needs refinement. <ul> | Deferred to the discussion with Claude for refinement. Other points raised in the review should come first, then discussion to finalize the math and the design, then the methods for the analytical properties reimplemented consistent with the design document. |
| `src/deep_isochron/systems/__init__.py`: exports | Trivial changes. | None |
| `src/deep_isochron/model/conjugacy.py`: `flow` in cartesian; `solver_config` field | Looks good. | None |
| `src/deep_isochron/model/latent_dynamics.py`: `HopfLatentDynamics` deleted | Good. On a related note regarding this file, a baseline I need to setup is that of Yawata et al. *Chaos* **34**, 063111 (2024), which uses a two variable latent dynamics with the phase evolving with a constant angular velocity and the amplitude exhibiting exponential decay. Once this is in place with the other normal forms, `LinearLatentDynamics` could also be removed. | None |
| `src/deep_isochron/data/dataset.py`: `DatasetMetadata`; xarray-backed source; splits; save/load | <ul><li>The information contained in the metadata look good. <li>However, storing them in a flat manner makes it feel a bit less organized. Feel it may be better to group them into relevant metadata groups: for example, `rtol`, `atol`, `solver`, `max_steps`, `t0`, `t1`, `strategy` would be grouped together in a diffeqsolve-configuration group. Need design refinement here, to flesh out the right groupings and the relevant group names. <li>`_with` method of the `TimeSeriesDataSource` seems unneeded. It is only used in `dataset.py`, and its functionally can be replaced by `copy.replace`. Better to use that instead. <ul> | Deferred to future discussions with Claude |
| `src/deep_isochron/data/generate.py`: IC samplers; `generate`; `config_hash`; provenance | The `_system_params` function logic is unclear - the `k[4:]` used in the `hasattr` and `getattr` is not clear to understand. Needs to be changed - for example, `AbstractODE` contract could be expanded to expose the necessary attributes. | None |
| `src/deep_isochron/data/sampling.py`: `weighted_windows`, `transient_weights`, `mixed_split` | Had a look at the implementation, but not sure if having functions that consume data sources to emit different `MapDataset`s is the idiomatic `grain` design. I think a cleaner alternative is letting the data source contain full, unwindowed trajectories, and implementing the windowing / weighted windowing strategies as a `grain.Map`/`grain.RandomMap` transforms. The source code of the DySLIM paper (Schiff et al. ICML 2024) [does this](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/ergodic/utils.py). Such refactor will then simplify all the helper methods of `TineSeriesDataSource` as well. Alternatively, one could create a custom `IndexSampler`, but I think the transform approach is cleaner. | None. Design refinement deferred to next discussion with Claude. |
| `src/deep_isochron/data/__init__.py`: exports | Trivial changes. | None. Will change in the future as the data contract becomes further refined. |
| `scripts/generate_data.py`, `configs/data/*.yaml`: Hydra entry point | Looks okay. However, these will change as the abstractions regarding data generation changes, so not paying too much attention for now. | None |
| `pyproject.toml`: `xarray`, `h5netcdf` | Good. | Ran `uv sync --all-groups` in the local dev environment. |
| `prototype.ipynb`: `flow`/`to_chart`; `SolverConfig` on the model | Had a cursory look. Will need to change as the abstractions are further refined. | Deferred to Claude to keep matching the refined abstractions. |
| `tests/test_systems.py`: normal-form identities, strategies, flow, observed systems | Had a cursory look. Like what is being tested, but feel like the tests for the normal forms and the tests for properties of a specific parameter values of a concrete systen (ex. testing for the eigenvalues of the Fitzhugh Nagumo) should be separated. Also the eigenvalue test should contain a reference to the Winfree puzzle (Langfield et al.) paper, which is where these values were taken from. | None |
| `tests/test_data.py`: windows, splits, disk, generate, sampling | Again what is being tested look okay; not delving into the specifics for now since with the refinements to the data abstraction, the tests will also change / be refined. | None |
| `tests/helpers.py`: `TOL["flow"]` | Simple change. Good. | None |
| `docs/decisions/0007-…`, `0008-…`: new ADRs | Looks good. Perhaps 0008 may need additional updates as the data abstractions are refined. | None |
| `docs/architecture.md`: systems/data sections | Read through, but the new architectural elements will need to be refined as per the comments above. | None |
| `docs/roadmap.md`: Phase B rewritten; ledger | Overall okay. But phase B will need to be changed - performing the design refinements flagged in this review is the next thing to do. `deep_isochron.analysis` is good, but this pertains to the science I want to do, and want a more careful consideration of what I need / what algorithms I will implement. So keep the namespace, but defer the actual implementation to phase E (and this will also be expanded as research progresses). | Deferred to Claude for updates. |

Review round 1 (commit `d6e560b`):

| Change | Thoughts | Modifications |
|---|---|---|
| `src/.../invertible/constraints.py`: `GreaterThan(lower, at_zero=lower+1)`; `Positive` alias; `at_zero` documented; `BoundedPositive.lower` | | |
| `src/.../invertible/{analytic,splines/rational_quadratic}.py`, `tests/test_constraints.py`: call sites | | |
| `src/deep_isochron/systems/base.py`: `params()` contract; single `flow -> Solution`; `SolverConfig.params()` | | |
| `src/deep_isochron/systems/normal_forms/{__init__,base,hopf,bautin}.py`: subpackage; `A_CONSTRAINT`/`B_CONSTRAINT`; `params()` | | |
| `src/deep_isochron/systems/normal_forms/integration.py`: `AbstractFlowIntegration` and the three integrations; `Solution` with cartesian `.ys` | | |
| `src/deep_isochron/model/conjugacy.py`, `data/generate.py`, `configs/`, `scripts/`: `.ys`, `integration`, `system.params()` | | |
| `tests/test_normal_forms.py`: generic normal-form laws (split out) | | |
| `tests/test_systems.py`: flow machinery; FHN equilibrium/eigenvalues/period and HH facts, citing Langfield et al. (2014) | | |
| `docs/decisions/0007-…`: amended for the review | | |
| `docs/architecture.md`, `docs/roadmap.md`: B8–B12, analysis to Phase E | | |

Review round 2 (B9) and the B10 draft:

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/design/normal-forms.md`: agreed revision (Joon, 2026-10-03) adopted; then amended — §5.1 corollary sign (confirmed), §5.3 Yawata paragraph rewritten from Eqs. (11)–(28) of the paper with the B12 implementation mapping, §5.4 Kvalheim & Revzen global existence/uniqueness (*proposed*) | | |
| `src/deep_isochron/data/dataset.py`: frozen-dataclass whole-trajectory source; grouped `DatasetMetadata`; `copy.replace` splits | | |
| `src/deep_isochron/data/windows.py`: `RandomWindow`, `WeightedWindow`, `transient_weight`, `windows`, `mixed_windows` (replaces `sampling.py`) | | |
| `src/deep_isochron/data/generate.py`: grouped metadata from `params()` | | |
| `src/deep_isochron/training/losses.py`, `training/__init__.py`: dict batches; no data re-export | | |
| `tests/test_data.py`: rewritten for the transform design | | |
| `configs/data/*.yaml`, `scripts/generate_data.py`, `prototype.ipynb`: follow the API | | |
| `docs/decisions/0008-…`, `docs/architecture.md`, `docs/roadmap.md`: amended | | |
