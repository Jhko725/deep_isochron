---
type: index
status: current
updated: 2026-10-05
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

## Design documents (`design/`) — the mathematics behind the code

| Document | Status | Covers |
|---|---|---|
| [`normal-forms.md`](design/normal-forms.md) | agreed; §5.3–5.4 tentative | Θ/Ψ conventions, Hopf and Bautin closed forms, the (Θ, Ψ) chart, the implementation contract (§9) and the tests it implies (§10); relation to Yawata et al. and Kvalheim & Revzen |
| [`cubic-bspline.md`](design/cubic-bspline.md) | agreed | the `CubicBSpline` boundary treatment and inverse error bound |

## Decision records (`decisions/`)

| ADR | Status | Decision |
|---|---|---|
| [0001](decisions/0001-unconstrained-leaves.md) | accepted | unconstrained `raw` leaves; `constrain(raw)` |
| [0002](decisions/0002-identity-at-zero.md) | accepted | `constrain(0)` is the identity |
| [0003](decisions/0003-declared-smoothness.md) | accepted | regularity is declared, not inferred |
| [0004](decisions/0004-abstractvar-fields.md) | accepted | `AbstractVar` as `eqx.field(static=True, init=False)` |
| [0005](decisions/0005-constraint-primitives.md) | accepted; amended | constraint primitives created inside `constrain`; shift computed lazily (no JAX at import) |
| [0006](decisions/0006-cubic-bspline-boundary-and-inverse.md) | accepted | `CubicBSpline` boundary and inverse |
| [0007](decisions/0007-systems-hierarchy-and-flow-strategies.md) | accepted; amended | `AbstractODE`/`AbstractNormalForm`, `SolverConfig`, integrations as objects |
| [0008](decisions/0008-dataset-format-and-sampling.md) | accepted; amended | netCDF4 datasets via xarray; weighted windows alongside `mix` |
| [0009](decisions/0009-trainer-losses-schedules-checkpoints.md) | accepted; amended (review round 2 pending) | trainer: injected logging/checkpointing, losses as weighted terms, schedules in the state, whole-state checkpoints |
| [0010](decisions/0010-rotations-by-cayley-transform.md) | accepted (review pending) | `BiLipschitzLinear` rotations by Cayley transform, not `expm` (GPU conditionals) |

## Change documents (`changes/`) — one per branch, the reviewer's map

| Branch | Document |
|---|---|
| `splines-refactor` | [`2026-09-30-splines-refactor.md`](changes/2026-09-30-splines-refactor.md) |
| `scalar-param-refactor` | [`2026-10-01-scalar-param-refactor.md`](changes/2026-10-01-scalar-param-refactor.md) |
| `invertible-cleanup` | [`2026-10-02-invertible-cleanup.md`](changes/2026-10-02-invertible-cleanup.md) |
| `data-generation` | [`2026-10-02-data-generation.md`](changes/2026-10-02-data-generation.md) |
| `trainer` | [`2026-10-04-trainer.md`](changes/2026-10-04-trainer.md) |
