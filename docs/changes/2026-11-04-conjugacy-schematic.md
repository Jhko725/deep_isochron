---
type: change
status: in progress; commits 1–2 landed, commit 3 waits for the notebook
updated: 2026-11-04
---

# `conjugacy-schematic` — the training pipeline drawn as a commutative square

## Summary

The branch adds `plotting/schematic.py`, the method figure of the paper and the talk:
four phase-plane panels in a square, drawn from real trajectories, joined by the
blocks and arrows of the training pipeline. The top row is the data space: windows of
FitzHugh–Nagumo data with their initial conditions marked, and the model's predictions
dashed over the same data. The bottom row is the latent space: the initial conditions
mapped through the bijection, open-marked, and the normal form's trajectories from
them. The blocks `h` and `h⁻¹` sit between the rows, the normal form between the
latent panels, the unknown true flow across the top, with the loss under it. One
call, `plot_conjugacy_schematic(model, ts, data, cycle)`, draws the figure from a
checkpoint and a validation batch. The design was settled in the grilling round of
2026-11-04; the decisions that are not obvious from the code are below.

## Files

- `src/deep_isochron/plotting/trajectories.py` — `cycle_curve`, moved from `isochrons.py` as a public helper; the module docstring names it.
- `src/deep_isochron/plotting/isochrons.py` — `_cycle_curve` removed; the four call sites import `cycle_curve` from `.trajectories`.
- `src/deep_isochron/plotting/style.py` — `curve_colors`, moved from `deformation/style.py` as a public helper; the module docstring names it.
- `src/deep_isochron/plotting/deformation/style.py` — `_curve_colors` removed; the module docstring points to `plotting.style.curve_colors`.
- `src/deep_isochron/plotting/deformation/figures.py` — imports `curve_colors` from `..style`.
- `src/deep_isochron/plotting/__init__.py` — exports `cycle_curve` and `curve_colors`.
- `tests/test_plotting.py` — `test_curve_colors_forms` imports `curve_colors` from `deep_isochron.plotting.style`.
- `docs/index.md` — a row for this change document.
- `src/deep_isochron/plotting/schematic.py` — `SchematicLabels`, `SchematicStyle` and the four panel drawers `draw_data`, `draw_latent_initial`, `draw_latent_trajectories` and `draw_prediction`.
- `src/deep_isochron/plotting/__init__.py` — exports `SchematicLabels`, `SchematicStyle` and the four drawers; the package docstring names the conjugacy schematic.
- `tests/test_plotting.py` — the `schematic` section: start markers, line styles, the shape annotation and margin check, and `copy.replace` on the labels.

## Design

The grilling round settled twenty-one questions; the ones a reader of the code would
ask are these.

- **One module, not a subpackage.** `deformation/` and `phase_portrait/` are
  subpackages because each holds several figures or animations; this holds one
  figure. The module splits if it passes the 600-line check, not before.
- **The contract is `ConjugateLatentDynamics`, not `AbstractPhaseAmplitudeModel`.**
  The latent circle comes from `latent_dynamics.limit_cycle`, and the blocks are
  labeled `h` and `h⁻¹`; the autoencoder has neither. Widening the abstract contract
  for one adopter would be a hypothetical seam.
- **The model is evaluated inside the figure function**, through
  `eqx.filter_jit(eqx.filter_vmap(model, in_axes=(None, 0)))` at the batch's own
  `ts`, so predictions are sampled where the data is. Rejected: a precomputed
  `(xt, yt)` argument, a second route to the same arrays; a finer grid for the dashed
  curves, until a render shows one is needed.
- **Segments are `ts` and `data[q, time, 2]`; `x0` is `data[:, 0]`.** Rejected: a
  separate `x0s` argument, which would be a second route to the same value. Which
  windows to draw is the caller's slice; `plotting` does not import `data/windows`.
- **The drawers are pure matplotlib.** Each takes an `Axes`, NumPy points and the
  style; the figure function samples the cycle (`cycle_curve`), the latent circle
  (`limit_cycle` over turns) and the model. A notebook can feed the drawers anything,
  which is how the assembly was prototyped.
- **Every string is data.** `SchematicLabels`, a frozen dataclass of raw mathtext
  (panel titles, the block names and formulas, the flow label, the loss, the legend),
  held by `SchematicStyle.labels`; an equation is changed with `copy.replace` in the
  notebook, not in the module. The default notation is `h`, the paper's.
- **Font sizes are five explicit fields** (`panel_title_size`, `block_title_size`,
  `block_formula_size`, `annotation_size`, `legend_size`), as `PortraitStyle` has
  `font_size` and `annotation_font_size`. Rejected: one size with fixed ratios, which
  hides the ratios.
- **Geometry in inches.** Figure size, panel edge, margins, block sizes and arrow gaps
  are `SchematicStyle` fields in inches, drawn on an overlay axes whose data units are
  inches, as `phase_portrait/drawing.py` places its axes by rect.
- **Shared boxes.** The two data panels share one box pooled from the cycle, the data
  and the predictions; the two latent panels share one pooled from the circle and the
  latent trajectories; each with a `margin` fraction. No `box=` override until a
  render asks for one.
- **Two helpers move because they gained a second adopter.** `_cycle_curve` of
  `isochrons.py` becomes `cycle_curve` in `trajectories.py`; `_curve_colors` of
  `deformation/style.py` becomes `curve_colors` in `plotting/style.py`.
  `phase_portrait/drawing.py`'s `_draw_cycle` stays, since it draws a normal form's
  closed-form cycle, not a `LimitCycle`.
- **No ADR.** Nothing here is hard to reverse or surprising without context.
- The change is reversible: reverting the branch removes the module and restores the
  two private helpers.

Ledger (one commit each):

1. [x] `cycle_curve` moved to `trajectories.py` and `curve_colors` promoted to
   `plotting/style.py`, call sites updated.
2. [x] `SchematicStyle`, `SchematicLabels` and the four drawers, with their tests.
3. [ ] `plot_conjugacy_schematic` with the geometry settled in the notebook, its test,
   the exports, the architecture line and the roadmap ledger. Waits for Joon's
   notebook prototype of the assembly.

## Bugs fixed

None.

## Tests

Per Joon's rule for plotting work (2026-11-04), only `tests/test_plotting.py` runs on
this branch, at the `dev` profile, before and after each commit; the full tier is not
run. Baseline on `a88fb03`: 26 passed in 62 s on two workers. After commit 1 (`eff7ae6`):
26 passed. After commit 2 (`90435c0`): 30 passed in 20 s, the four new tests being
the start markers' fill per drawer, solid versus dashed curves, the shape and style
errors, and `copy.replace` on `SchematicLabels`. The count after commit 3 is pending.

## Open issues

- Deferred, by the grilling round: icons marking `h` and the normal form's parameters
  as trainable; a rubber-sheet thumbnail of `h` beside its block; a slide build-up
  (panels appearing in pipeline order); a `box=` override of the panel limits.
- Ledger item 3 waits for the notebook prototype, which fixes the geometry defaults.

## Review notes

One row per *Files* bullet; *Thoughts* and *Modifications* are Joon's.

| Change | Thoughts | Modifications |
|---|---|---|
| `src/deep_isochron/plotting/trajectories.py` — `cycle_curve`, moved from `isochrons.py` | | |
| `src/deep_isochron/plotting/isochrons.py` — `_cycle_curve` removed, the four call sites | | |
| `src/deep_isochron/plotting/style.py` — `curve_colors`, moved from `deformation/style.py` | | |
| `src/deep_isochron/plotting/deformation/style.py` — `_curve_colors` removed, the module docstring | | |
| `src/deep_isochron/plotting/deformation/figures.py` — imports `curve_colors` from `..style` | | |
| `src/deep_isochron/plotting/__init__.py` — the exports of `cycle_curve` and `curve_colors` | | |
| `tests/test_plotting.py` — `test_curve_colors_forms` imports `curve_colors` | | |
| `docs/index.md` — the `conjugacy-schematic` row | | |
| `src/deep_isochron/plotting/schematic.py` — `SchematicLabels`, `SchematicStyle` and the four drawers | | |
| `src/deep_isochron/plotting/__init__.py` — the exports of the schematic and the package docstring | | |
| `tests/test_plotting.py` — the `schematic` section | | |
