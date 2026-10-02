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
  scalar templates; `affine_coupling (flip, 1 of 3)` and `bilipschitz_linear (dim 3,
  L=5)` builders; `invertible_linear` entry, `IDENTITY_AT_INIT` and
  `ORIENTATION_NOT_GUARANTEED` deleted; keyword `width_hidden` → `mlp_width`.
- `src/deep_isochron/model/invertible/linear.py` — `InvertibleLinear` removed;
  `BiLipschitzLinear` on leaves `raw_U`/`raw_V`/`raw_s` with `constrain` →
  `LinearParams(U, V, s)` (`Interval(1/L, L, at_zero=1)`), `params`/`weight`
  properties, `init="identity" | "rotation"` (default identity), `L > 1` required.
- `tests/test_linear.py` — new: parameter-space guarantees at arbitrary raw leaves
  (`U, V ∈ SO(dim)`, `s ∈ (1/L, L)`, `σ(W)` bounds, `det W > 0`), `weight` built from
  `params`, both inits, validation.
- `tests/test_bijections.py` — `IDENTITY_AT_INIT`/`ORIENTATION_NOT_GUARANTEED` skips
  removed; identity-at-init and orientation are universal laws.
- `prototype.ipynb` — `AffineCoupling(width_hidden=..., affine_clamping=...)` →
  `(mlp_width=..., clamp=...)`; `InvertibleLinear` cells → `BiLipschitzLinear`.
- `src/deep_isochron/model/invertible/base.py` — `AbstractScalarBijection` docstring is
  now the six-step implementation checklist (A5).
- `src/deep_isochron/model/invertible/splines/base.py` — module docstring points at the
  checklist.
- `src/deep_isochron/model/invertible/splines/cubic.py` — docstring: corrected inverse
  error-bound claim; links to ADR-0006 and the design note.
- `docs/decisions/0006-cubic-bspline-boundary-and-inverse.md` — new ADR (A6).
- `docs/design/cubic-bspline.md` — new math note: index conventions, parametrisation,
  `C²` boundary argument, pieces, inverse guarantees (A6).
- `docs/architecture.md` — new: module map, invertible-package overview, systems/chart,
  training-step data flow, tests, ADR index (A7).
- `CLAUDE.md` — points at `docs/architecture.md`.
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

- **A4 — `InvertibleLinear` removed rather than reparametrised.** The alternative was
  `W = expm(skew)` (a pure rotation) or `expm(skew) · diag(exp(·)) · expm(skew)ᵀ`
  (an SVD without the Lipschitz bound). The second is `BiLipschitzLinear` with
  `L → ∞`, so a separate class would duplicate it for the sole benefit of an unbounded
  condition number — which is the failure mode the bi-Lipschitz layer exists to
  prevent in a conjugacy. The first is a strict subset (`s ≡ 1`). An unconstrained
  matrix cannot stay in `GL⁺(dim)` under gradient steps (`det W` is continuous and can
  cross 0), so it can never satisfy the orientation law the rest of the vocabulary
  satisfies; with it gone the law is universal and `ORIENTATION_NOT_GUARANTEED` is
  deleted. Reinstating it is one `git revert` away if an experiment needs it. The
  Mezzadri Haar-rotation init went with it: `BiLipschitzLinear`'s `init="rotation"`
  draws Gaussian skew generators (the former default), which gives a random — not
  Haar-distributed — rotation; Haar would need a matrix logarithm JAX does not ship
  and the distribution does not matter for an init.
- **A2 — identity is the default init; rotation opt-in.** With `raw_U = raw_V = 0` and
  `raw_s = 0` (`Interval(1/L, L, at_zero=1)` puts the shift in the map, ADR-0002) the
  layer is `I`, so an INN of alternating linear and coupling layers starts as the
  identity. The roadmap's "Parked" initialisation-strategy enum is not introduced; the
  one `Literal["identity", "rotation"]` argument on the one class that has a choice is
  enough.
- **A3 — three raw leaves, not one `raw` vector.** `BiLipschitzLinear` is a vector
  bijection, so it is not bound by the scalar contract's single-leaf rule; what ADR-0001
  asks for is that leaves be unconstrained and constrained values be computed on read
  in one place (`constrain`). Flattening `(U, V, s)` into one vector would only add
  reshapes. `L = 1` is now rejected (`Interval(1, 1)` is empty); the former code
  accepted it and produced a constant `s = 1`, a degenerate rotation-only layer nobody
  used.

- **A6 — the `2^-newton_iters` claim was wrong and is corrected, not implemented.**
  The `CubicBSpline` docstring said the inverse is exact to `2^-newton_iters` of a bin in
  the worst case. That bound holds for pure bisection; an accepted Newton step may shrink
  the bracket by less than half, so the mixed scheme has no uniform bound (it does have
  the invariants that matter — bracketed root, in-bin iterate, hence always finite — and
  it converges). Forcing a bisection every other iteration would restore a `2^-n/2`
  bound at the cost of typical speed; not done, since the suite shows round-off accuracy
  at 20 iterations. Recorded in ADR-0006 and §5 of the design note.
- **A7 — one page, prose plus one tree and one flow diagram.** A fuller document (per-class
  API tables) would duplicate docstrings and go stale; the page says what each module is
  *for* and where the decisions are, which is what a new session needs first.

ADRs: [0001](../decisions/0001-unconstrained-leaves.md),
[0002](../decisions/0002-identity-at-zero.md),
[0005](../decisions/0005-constraint-primitives.md),
[0006](../decisions/0006-cubic-bspline-boundary-and-inverse.md) (new);
design note [`docs/design/cubic-bspline.md`](../design/cubic-bspline.md).

## Bugs fixed

- `ResidualCoupling` was not the identity at init (no zeroed final layer); see Design.
- `BiLipschitzLinear(max_lipschitz=1.0)` was accepted and silently produced a constant
  `s = 1`; now rejected.

## Tests

`uv run pytest -n 4` (`--hypothesis-profile=dev` for a quick pass). No new test files:
`Shift`/`Affine` are scalar templates in the registry and so run through every law in
`test_bijections.py` (round trip, identity at `raw = 0`, Jacobian vs. finite differences,
orientation, composition, template contract); the two coupling factories run through the
vector laws including identity-at-init.

`tests/test_linear.py` sets `BiLipschitzLinear`'s raw leaves directly to arbitrary values
(singular-value leaves up to `|raw| = 30`) and checks the guarantees that hold at every
point of parameter space — `U, V ∈ SO(dim)`, `s ∈ (1/L, L)`, `1/L ≤ σ(W) ≤ L`, `det W >
0` — plus that `weight` is assembled from `params`, both inits, and the validation
errors. With `IDENTITY_AT_INIT`/`ORIENTATION_NOT_GUARANTEED` gone,
`test_vector_identity_at_init` and `test_vector_orientation_preserving` run on every
registry entry with no skips.

## Open issues

- Periodic / free-boundary endpoint handling on `AbstractSpline` stays in Phase E
  (roadmap).

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/.../invertible/affine.py`: `Shift`/`Affine` templates; coupling classes → factories over `CouplingFlow` | | |
| `src/.../invertible/__init__.py`: export `Shift`, `Affine` | | |
| `tests/registry.py`: new scalar templates and builders; `invertible_linear`, `IDENTITY_AT_INIT`, `ORIENTATION_NOT_GUARANTEED` deleted | | |
| `src/.../invertible/linear.py`: `InvertibleLinear` removed; `BiLipschitzLinear` raw leaves + `LinearParams`, identity init, `init="rotation"` opt-in | | |
| `tests/test_linear.py`: parameter-space guarantees, params/weight, inits, validation | | |
| `tests/test_bijections.py`: identity-at-init and orientation laws without skips | | |
| `prototype.ipynb`: `AffineCoupling` keyword rename; `InvertibleLinear` → `BiLipschitzLinear` | | |
| `src/.../invertible/base.py`: `AbstractScalarBijection` implementation checklist docstring | | |
| `src/.../invertible/splines/base.py`: cross-link to the checklist | | |
| `src/.../invertible/splines/cubic.py`: corrected inverse-bound claim; doc links | | |
| `docs/decisions/0006-cubic-bspline-boundary-and-inverse.md`: Greville pinning; bracketed Newton inverse | | |
| `docs/design/cubic-bspline.md`: indices, parametrisation, `C²` argument, inverse guarantees | | |
| `docs/architecture.md`: module map and data flow | | |
| `CLAUDE.md`: architecture pointer | | |
| `docs/roadmap.md`: Phase A ledger updates | | |
