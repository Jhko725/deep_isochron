# 2026-10-02 — `invertible-cleanup`

Phase A of `docs/roadmap.md`: structural follow-ups from the two reviews of the
invertible package, closing its design before the data/trainer phases build on it.

## Summary

Collapses the two hand-written coupling classes onto `CouplingFlow` via `Shift`/`Affine`
scalar templates (A1); makes identity-at-init a universal law by initialising the linear
layers to `I` and bringing `BiLipschitzLinear` into the `raw`/`constrain` shape (A2, A3);
settles `InvertibleLinear`'s orientation (A4); documents the scalar-bijection
implementation checklist, the `CubicBSpline` design and the package architecture (A5–A7);
removes the remaining `ty: ignore`s and moves `matplotlib` to the dev group (A8, A9).

## Files

- `src/deep_isochron/model/invertible/affine.py` — rewritten: `Shift`, `Affine` scalar
  bijections (`ShiftParams`, `AffineParams`); `ResidualCoupling`/`AffineCoupling` become
  factory functions returning `CouplingFlow(Shift())` / `CouplingFlow(Affine(clamp))`.
- `src/deep_isochron/model/invertible/__init__.py` — exports `Shift`, `Affine`.
- `tests/registry.py` — `shift`, `affine`, `affine (clamp=0.5)`, `chain (affine, shift)`
  scalar templates; `affine_coupling (flip, 1 of 3)` builder; `residual_coupling` now in
  `IDENTITY_AT_INIT`; keyword `width_hidden` → `mlp_width`.
- `prototype.ipynb` — `AffineCoupling(width_hidden=..., affine_clamping=...)` →
  `(mlp_width=..., clamp=...)`.
- `docs/roadmap.md` — Phase A items moved to the Done ledger as they land.
- `docs/changes/2026-10-02-invertible-cleanup.md` — this document.

## Design

- **A1 — factory functions, not subclasses.** `AffineCoupling`/`ResidualCoupling` keep
  their names (the notebook and the registry use them) but are plain functions returning a
  `CouplingFlow`. A subclass `class AffineCoupling(CouplingFlow)` was rejected: it would
  add a type with no behaviour of its own, and `tests/test_registry.py` would then demand
  a registry entry for a class that is just a template choice. The `# noqa: N802` marks
  the deliberate CapWords on a function.
- **A1 — `Affine` scale via `BoundedPositive(0, at_zero=1, a=clamp)`.** This is exactly
  the former `exp(affine_clamping * tanh(s))` (RealNVP soft clamp), now with the bound
  as a static field of the template rather than of the coupling layer. The former
  `affine_clamping=None` (unbounded `exp`) was dropped: nothing used it, and an unbounded
  scale is the overflow mechanism `BoundedPositive` exists to remove (ADR-0005).
- **A1 — `flip` semantics.** The former classes swapped which half was conditioned on
  while keeping the array order; `CouplingFlow.flip` reverses the array before splitting.
  For `split_idx = dim // 2` on even `dim` these coincide up to the (irrelevant) internal
  ordering of the conditioned half; for other splits the conditioned block is the *last*
  `split_idx` entries rather than the last `dim - split_idx`. No caller relied on the old
  convention.
- **A1 — `ResidualCoupling` is now identity at init.** The former class never zeroed its
  MLP's final layer, so a fresh `ResidualCoupling` was a random translation. Through
  `CouplingFlow` it inherits the zero-initialised final layer, and `constrain(0)` of
  `Shift` is `loc = 0` (ADR-0002).

ADRs: [0001](../decisions/0001-unconstrained-leaves.md),
[0002](../decisions/0002-identity-at-zero.md),
[0005](../decisions/0005-constraint-primitives.md).

## Bugs fixed

- `ResidualCoupling` was not the identity at init (no zeroed final layer); see Design.

## Tests

`uv run pytest -n 4` (`--hypothesis-profile=dev` for a quick pass). No new test files:
`Shift`/`Affine` are scalar templates in the registry and so run through every law in
`test_bijections.py` (round trip, identity at `raw = 0`, Jacobian vs. finite differences,
orientation, composition, template contract); the two coupling factories run through the
vector laws including identity-at-init.

## Open issues

- Periodic / free-boundary endpoint handling on `AbstractSpline` stays in Phase E
  (roadmap).

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/.../invertible/affine.py`: `Shift`/`Affine` templates; coupling classes → factories over `CouplingFlow` | | |
| `src/.../invertible/__init__.py`: export `Shift`, `Affine` | | |
| `tests/registry.py`: new scalar templates and builders; `residual_coupling` identity at init | | |
| `prototype.ipynb`: `AffineCoupling` keyword rename | | |
| `docs/roadmap.md`: Phase A ledger updates | | |
