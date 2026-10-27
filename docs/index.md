---
type: index
status: current
updated: 2026-10-27
---

# `deep_isochron` documentation — index

Progressive disclosure: start here, then read only what the task needs. Every document
carries a frontmatter block (`type`, `status`, `updated`, and where relevant
`verified_by`, `sources`, `id`) — the useful part of Google's Open Knowledge Format,
adopted 2026-10-04 so that a document's standing is machine-readable without reading it.
`status` values: `current` (kept true at all times), `agreed` (design settled with Joon),
`accepted` (a decision in force; `amended` when later changed in place), `tentative`
(written by Claude, not yet checked by Joon).

## Read first

| Document | Type | What it answers |
|---|---|---|
| [`architecture.md`](architecture.md) | architecture | module map, data flow of one training step, where each decision is recorded |
| [`roadmap.md`](roadmap.md) | roadmap | the single current plan, phase by phase, with the Done ledger |
| [`../CLAUDE.md`](../CLAUDE.md) | conventions | how Claude works in this repository (change documents, review notes, tooling) |
| [`coding-standards.md`](coding-standards.md) | reference | where each convention is enforced here: tiers, named tolerances, the contracts the ADRs fix, the Markdown rules, the docstring and citation forms, the checks that replace prose |
| [`references.md`](references.md) | reference | the one bibliography: every work the documents, docstrings and tests cite, by key, with DOI and the locators used |

## Design documents (`design/`) — the mathematics behind the code

| Document | Status | Covers |
|---|---|---|
| [`normal-forms.md`](design/normal-forms.md) | agreed; §5.3–5.4 and the 2026-10-08 amendment (§3.3 Winfree, basins) tentative | Θ/Ψ conventions, Hopf and Bautin closed forms, the (Θ, Ψ) chart, the implementation contract (§9) and the tests it implies (§10); relation to Yawata et al. and Kvalheim & Revzen |
| [`cubic-bspline.md`](design/cubic-bspline.md) | agreed | the `CubicBSpline` boundary treatment and inverse error bound |
| [`validation-split.md`](design/validation-split.md) | tentative | why `val/mse` is a near-cycle metric and the early/late split by window start (`t_split`); how to choose the threshold; what it does not do |
| [`analysis.md`](design/analysis.md) | tentative (§1 E1, §2 O1) | the `Cycle` representation; per analysis function (E1, O1 so far): literature survey with cited/deduced claims, method, oracles, limits |
| [`isochrons.md`](design/isochrons.md) | draft for review (§3 and §11 open) | isochrons by AUTO-07p continuation: the method of Langfield et al. 2014, the coworker's run-by-run algorithm, ours; AUTO from the ground up; the stable direction (variational, adjoint, and the coworker's $+A^{\mathsf T}$ defect); accuracy, oracles, limits, tests (roadmap O7) |
| [`training-step-performance.md`](design/training-step-performance.md) | tentative | where a JAX training step spends its time: asynchronous dispatch, committed/uncommitted placement, the GIL and grain's threads, what `bench_dataloader.py` measures, the bottleneck taxonomy, and the Phase C case history |

## Decision records (`decisions/`)

Numbers are allocated when a branch is cut, so merge order can leave a temporary gap (0012 lived on `analysis` while 0013 landed on master).

| ADR | Status | Decision |
|---|---|---|
| [0001](decisions/0001-unconstrained-leaves.md) | accepted | unconstrained `raw` leaves; `constrain(raw)` |
| [0002](decisions/0002-identity-at-zero.md) | accepted | `constrain(0)` is the identity |
| [0003](decisions/0003-declared-smoothness.md) | accepted | regularity is declared, not inferred |
| [0004](decisions/0004-abstractvar-fields.md) | accepted | `AbstractVar` as `eqx.field(static=True, init=False)` |
| [0005](decisions/0005-constraint-primitives.md) | accepted; amended | constraint primitives created inside `constrain`; shift computed lazily (no JAX at import) |
| [0006](decisions/0006-cubic-bspline-boundary-and-inverse.md) | accepted | `CubicBSpline` boundary and inverse |
| [0007](decisions/0007-systems-hierarchy-and-flow-strategies.md) | accepted; amended (ADR-0013: basins, even subfamily; ADR-0016: symbolic defining data) | `AbstractODE`/`AbstractNormalForm`, `SolverConfig`, integrations as objects |
| [0008](decisions/0008-dataset-format-and-sampling.md) | accepted; amended (Decision 3 review pending) | netCDF4 datasets via xarray; weighted windows alongside `mix`; batched `WindowBatchSource` for training runs |
| [0009](decisions/0009-trainer-losses-schedules-checkpoints.md) | accepted; amended (review round 2 pending) | trainer: injected logging/checkpointing, losses as weighted terms, schedules in the state, whole-state checkpoints |
| [0010](decisions/0010-rotations-by-cayley-transform.md) | accepted | `BiLipschitzLinear` rotations by Cayley transform, not `expm` (GPU conditionals) |
| [0011](decisions/0011-experiment-layer.md) | accepted; amended 2026-10-07 (§6 `Run`/`setup`, explicit resume) | experiment layer: YAML `_target_` configs, builders, Hydra-owned runs, explicit data generation, forced final checkpoint |
| [0012](decisions/0012-analysis-layout.md) | accepted; amended 2026-10-15 (ODE half first; `Cycle` not the organizing object; survey delivered before implementation) | `analysis`: a data-driven half and an ODE half, shared objects chosen from what the functions share; `estimate_cycle` self-contained, `settling_time` takes a cycle; the data half host-side NumPy, the ODE half's results `eqx.Module`s; one commit per function after a literature survey |
| [0013](decisions/0013-normal-form-basins.md) | accepted (review pending) | normal forms take their data in `r` on a declared basin; evenness is the `AbstractEvenNormalForm` subfamily (Hopf, Bautin); Winfree's model with a hole; `nan` outside the basin; `generate(outside_basin=error\|resample)` |
| [0014](decisions/0014-limit-cycle-by-shooting.md) | accepted (review pending) | `analysis.ode.limit_cycle`: shooting on the time-T flow (not collocation), Poincaré section normal to the flow, `PoincareMap` as the one abstraction with `direction`/`reverse_time`, `eqx.Module` results, `analysis` through the `flow` contract alone; Floquet multipliers in ADR-0015 |
| [0015](decisions/0015-floquet-by-seeded-periodic-schur.md) | accepted (review pending) | `analysis.ode.floquet`: the seeded periodic Schur chain on the shooting-segment Jacobians (forward, in the attracting direction) as the method, explicit monodromy and Liouville as checks, `Literal` method dispatch; `LimitCycle`/`FixedPoint` as `eqx.Module`s carrying the ODE and configuration (amends ADR-0012 §3, ADR-0014 §3) |
| [0016](decisions/0016-symbolic-systems.md) | accepted (review pending) | systems declared once in SymPy (`AbstractSymbolicODE._symbolic_template`; normal forms' polar `_symbolic_template_polar`, with even-family membership checked), JAX `rhs`/hooks derived and compiled by `_compile` once a class has no abstract methods; AUTO-07p Fortran by `INCLUDE` of generated `rhs.inc`/`indices.inc` (amends ADR-0007) |
| 0017 | reserved | the AUTO-07p isochron pipeline (branch `auto-isochrons`, roadmap O7 step 2) |
| [0018](decisions/0018-docstrings-map-and-contract.md) | accepted | docstrings: the module docstring is the map, a function docstring the contract, reasoning in the ADRs; `**Attributes:**` lists; one bibliography `references.md` with `(Key, locator)` citations and `check_references.py`; ruff `D100`–`D103` switched on round by round |
| [0019](decisions/0019-adrs-describe-the-current-design.md) | accepted | ADRs: Context with Motivation and Forces; Decision and Consequences always the current design; superseded designs in Alternatives rejected; a dated History; relative links checked by `check_references.py` |

## Change documents (`changes/`) — one per branch, the reviewer's map

| Branch | Document |
|---|---|
| `splines-refactor` | [`2026-09-30-splines-refactor.md`](changes/2026-09-30-splines-refactor.md) |
| `scalar-param-refactor` | [`2026-10-01-scalar-param-refactor.md`](changes/2026-10-01-scalar-param-refactor.md) |
| `invertible-cleanup` | [`2026-10-02-invertible-cleanup.md`](changes/2026-10-02-invertible-cleanup.md) |
| `data-generation` | [`2026-10-02-data-generation.md`](changes/2026-10-02-data-generation.md) |
| `trainer` | [`2026-10-04-trainer.md`](changes/2026-10-04-trainer.md) |
| `experiment-config` | [`2026-10-05-experiment-config.md`](changes/2026-10-05-experiment-config.md) |
| `analysis` | [`2026-10-06-analysis.md`](changes/2026-10-06-analysis.md) |
| `run-layout` | [`2026-10-07-run-layout.md`](changes/2026-10-07-run-layout.md) |
| `normal-forms-basin` | [`2026-10-08-normal-forms-basin.md`](changes/2026-10-08-normal-forms-basin.md) |
| `conventions-docs` | [`2026-10-25-conventions-docs.md`](changes/2026-10-25-conventions-docs.md) |
| `prose` | [`2026-10-26-prose.md`](changes/2026-10-26-prose.md) |
