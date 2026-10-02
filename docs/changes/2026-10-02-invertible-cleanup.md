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
- `src/deep_isochron/systems/{base,fitzhugh_nagumo,hodgekin_huxley,normal_forms}.py` —
  `dim: ClassVar[int] = … # ty: ignore` → `eqx.field(static=True, default=…, init=False)`
  (ADR-0004); `rhs(t: Float[ArrayLike, ""], …)` to match diffrax's `ODETerm` (no
  `ty: ignore` on `ODETerm(self.rhs)`); unused ignore removed (A8).
- `src/deep_isochron/training/trainer.py` — all `ty: ignore`s removed: `cast(optax.Params,
  …)` at the optax boundary, `cast(PreservationPolicy, BestN(…))` and `str(path)` at the
  orbax boundary, correct `train_step`/`_train_step` return types, `state_prev` typed
  and the post-loop flush guarded; E501 fixed (A8).
- `src/deep_isochron/training/__init__.py` — re-exports `TimeSeriesDataSource` from
  `..data` (was `from .dataset`, a module that does not exist in `training/`).
- `src/deep_isochron/data/__init__.py` — new: exports `TimeSeriesDataSource`.
- `src/deep_isochron/data/dataset.py` — `split` returns an explicit `(before, after)` pair
  built with `type(self)` (matches its `tuple[Self, Self]` annotation).
- `src/deep_isochron/model/conjugacy.py` — `latent_dynamics: AbstractODE` (only `.solve`
  is used; the `AbstractLatentDynamics` half of the union had no `solve`).
- `src/deep_isochron/model/latent_dynamics.py` — `HopfLatentDynamics` calls
  `self.hopf.solve(…)` (`HopfNormalForm` is not callable; legacy path, see Open issues).
- `pyproject.toml` — `matplotlib` moved from `dependencies` to the `dev` group (A9).
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
- **A8 — the two remaining type mismatches are upstream, so they are `cast`s, not
  ignores.** optax types `update`'s arguments as `Params = ArrayTree`, a recursive union
  of arrays and `Iterable`/`Mapping` containers; an Equinox module is a pytree but not an
  `Iterable`, so the type checker cannot know it is valid. orbax's `BestN` does not
  satisfy orbax's own `PreservationPolicy` protocol as typed (its `should_preserve`
  parameter type differs). A `typing.cast` names the type being asserted and is checked
  for plausibility, where `# ty: ignore` silences everything on the line; both casts
  carry a comment saying they are type-level only. `rhs`'s `t` becomes
  `Float[ArrayLike, ""]` because diffrax's `ODETerm` may pass a Python float — the
  annotation was narrower than the call site, which is a real (if harmless) mismatch
  rather than a checker limitation.
- **A9 — `uv.lock` deliberately not regenerated here.** Running `uv lock` with the
  container's `uv 0.8` rewrote the lockfile's revision (1 → 3, `upload-time` fields on
  every entry; ~2900 lines) with only five actual version changes. That noise belongs to
  the reviewer's own `uv`: `uv sync --group dev` will refresh the lock for the group move.
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
- `from deep_isochron.training import TimeSeriesDataSource, Trainer` (the notebook's
  import) raised `ModuleNotFoundError`: `training/__init__.py` imported `.dataset`, which
  lives in `data/`. Fixed by re-exporting from `..data` (and giving `data/` an
  `__init__.py`).
- `Trainer.train` crashed with `AttributeError` on `None` when the dataloader was empty or
  `num_steps == 0` (post-loop flush of `state_prev`); now guarded.
- `HopfLatentDynamics.__call__` called `HopfNormalForm` as a function (it has `solve`, no
  `__call__`); would have raised `TypeError`. Now `.solve`.

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

Final state on the branch: `pytest -n 4` (default profile) 360 passed, 1 xfailed, 0
skipped; `ty check src` clean (`--python` pointing at a venv with `wandb` installed);
`grep -rn "ty: ignore" src` empty; `ruff check src tests` clean.

## Open issues

- Periodic / free-boundary endpoint handling on `AbstractSpline` stays in Phase E
  (roadmap).
- `latent_dynamics.py` / `autoencoder.py` (the non-invertible autoencoder path) are
  legacy: `HopfLatentDynamics` is only made type-correct here, not re-validated. Phase B
  decides what of them survives (`docs/architecture.md`, *Systems and the latent chart*).
- `ty check src` is clean only with `wandb` and `orbax-checkpoint` importable; in a bare
  checkout without `uv sync` the two imports are reported as unresolved.
- `uv.lock` to be refreshed by `uv sync` on the reviewer's machine (see Design, A9).

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/.../invertible/affine.py`: `Shift`/`Affine` templates; coupling classes → factories over `CouplingFlow` | Looks good. Maybe later, it might be useful to group the scalar bijections together instead of having them scattered about. But not a concern for now. | None. |
| `src/.../invertible/__init__.py`: export `Shift`, `Affine` | Trivial change. | None |
| `tests/registry.py`: new scalar templates and builders; `invertible_linear`, `IDENTITY_AT_INIT`, `ORIENTATION_NOT_GUARANTEED` deleted | Looks good. | None |
| `src/.../invertible/linear.py`: `InvertibleLinear` removed; `BiLipschitzLinear` raw leaves + `LinearParams`, identity init, `init="rotation"` opt-in | Will leave as is for now. `InvertibleLinear` may be brought back in the future - mathematically the determinant sign can cross zero, but in practice, did not find that to happen. Furthermore, `InvertibleLinear` was much cheaper to evaluate than `BiLipschitzLinear`. In the future, will try different parametrization (than the matrix exponential) for the class. If the result proves to still be too slow, `InvertibleLinear` will be brought back, as having one test exception group is worth the price. | None |
| `tests/test_linear.py`: parameter-space guarantees, params/weight, inits, validation | The properties being tested look okay, but seeing tolerance values in the test functions that should be named and included in the `TOL` dictionary of `helpers.py`. Speaking of tolerances, is the `IDENTITY_TOL` dict in `registry.py` still relevant? | Deferred to Claude for further changes. |
| `tests/test_bijections.py`: identity-at-init and orientation laws without skips | Looks good. | None |
| `prototype.ipynb`: `AffineCoupling` keyword rename; `InvertibleLinear` → `BiLipschitzLinear` | Confirmed. | None |
| `src/.../invertible/base.py`: `AbstractScalarBijection` implementation checklist docstring | Docstring does make this design element harder to miss. Good. | None |
| `src/.../invertible/splines/base.py`: cross-link to the checklist | Confirmed. | None |
| `src/.../invertible/splines/cubic.py`: corrected inverse-bound claim; doc links | Doc links are  good. No need to mention bisection bounds in the doc string (its not the actual bound, so is superfluous). | Doc string changed to refer to the design docs for discussions on convergence. |
| `docs/decisions/0006-cubic-bspline-boundary-and-inverse.md`: Greville pinning; bracketed Newton inverse | Read through. Looks good. | None |
| `docs/design/cubic-bspline.md`: indices, parametrisation, `C²` argument, inverse guarantees | Had a cursory look for now. Will revisit in the future for in-depth scrutiny when I start heavily using `CubicBSpline` for the learned bijections. | None |
| `docs/architecture.md`: module map and data flow | On a high level, good. The specifics  will change with time, but so will the document. | None |
| `CLAUDE.md`: architecture pointer | Read through the file. Good. | None |
| `src/deep_isochron/systems/*.py`: `dim` as static `init=False` field; `t: ArrayLike` in `rhs` | Looks good. | None |
| `src/deep_isochron/training/trainer.py`: casts at optax/orbax boundaries; return types; guarded flush | Trivial change. | None |
| `src/deep_isochron/training/__init__.py`, `data/__init__.py`: fixed `TimeSeriesDataSource` re-export | Trivial change. | None |
| `src/deep_isochron/data/dataset.py`: `split` returns a typed pair | Simple fix. Good. | None |
| `src/deep_isochron/model/conjugacy.py`: `latent_dynamics: AbstractODE` | Fair. The original issue stemmed from the fact the `AbstractODE` and `AbstractLatentDynamics` were designed at different times + Proper design refinement was not performed (`AbstractLatentDynamics` should have been a subclass of `AbstractODE`; The intention was `AbstractODE` are all ODE systems used in the study - normal forms, data generation; not necessarily need to carry helper methods required to analytically compute phase/amplitude response curves, etc., whereas `AbstractLatentDynamics` are the subsets carrying that info). For now, this revised type hint suffices, and the design refinement will be done in Phase B. | None |
| `src/deep_isochron/model/latent_dynamics.py`: `HopfLatentDynamics` uses `.solve` | Good. The latent_dynamics code here is legacy, and not planned to be used in experiments for the immediate future. So keeping them type correct is sufficient. | None |
| `pyproject.toml`: `matplotlib` → dev group | Trivial changes. | Ran uv sync on the local repo. |
| `docs/roadmap.md`: Phase A ledger updates | Read through. Looks good. | None |
