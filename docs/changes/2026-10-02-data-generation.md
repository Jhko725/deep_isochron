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
- `docs/decisions/0007-systems-hierarchy-and-flow-strategies.md`,
  `docs/decisions/0008-dataset-format-and-sampling.md` — new ADRs.
- `docs/architecture.md` — systems and data sections rewritten; module map updated.
- `docs/roadmap.md` — Phase B rewritten around the decisions; B1–B7 to Done; B8
  (`analysis/`) added; parked items.

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
| `src/deep_isochron/systems/base.py`: `SolverConfig`; `flow`/`flow_result` replace `solve` | | |
| `src/deep_isochron/systems/normal_form.py`: `AbstractNormalForm` with closed-form phase/amplitude/isochrons and the chart | | |
| `src/deep_isochron/systems/strategies.py`: cartesian / polar / `r²` integration strategies | | |
| `src/deep_isochron/systems/normal_forms.py`: Hopf/Bautin on the two rates; constrained leaves; `b > −1` | | |
| `src/deep_isochron/systems/__init__.py`: exports | | |
| `src/deep_isochron/model/conjugacy.py`: `flow` in cartesian; `solver_config` field | | |
| `src/deep_isochron/model/latent_dynamics.py`: `HopfLatentDynamics` deleted | | |
| `src/deep_isochron/data/dataset.py`: `DatasetMetadata`; xarray-backed source; splits; save/load | | |
| `src/deep_isochron/data/generate.py`: IC samplers; `generate`; `config_hash`; provenance | | |
| `src/deep_isochron/data/sampling.py`: `weighted_windows`, `transient_weights`, `mixed_split` | | |
| `src/deep_isochron/data/__init__.py`: exports | | |
| `scripts/generate_data.py`, `configs/data/*.yaml`: Hydra entry point | | |
| `pyproject.toml`: `xarray`, `h5netcdf` | | |
| `prototype.ipynb`: `flow`/`to_chart`; `SolverConfig` on the model | | |
| `tests/test_systems.py`: normal-form identities, strategies, flow, observed systems | | |
| `tests/test_data.py`: windows, splits, disk, generate, sampling | | |
| `tests/helpers.py`: `TOL["flow"]` | | |
| `docs/decisions/0007-…`, `0008-…`: new ADRs | | |
| `docs/architecture.md`: systems/data sections | | |
| `docs/roadmap.md`: Phase B rewritten; ledger | | |
