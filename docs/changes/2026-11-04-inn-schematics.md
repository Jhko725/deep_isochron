---
type: change
status: landed on the branch; review pending
updated: 2026-11-04
---

# `inn-schematics` — the two INN schematics of KSIAM slides 9a and 9b

## Summary

The branch adds two figures to `plotting` for the KSIAM 2026 talk. `inn_stack.py`
draws the invertible network as a chain of six blocks `h₁ … h_N` alternating linear
and spline coupling layers, with the forward map `H` above, the inverse `H⁻¹` below,
and under each block type a before and an after panel of a square region (a grid and
four Bautin trajectories) mapped through that layer (slide 9a). `coupling_warp.py`
draws the spline coupling layer beside an affine foil as a 2 × 3: the warp of one
column above, the image of the square region below, for no coupling, affine coupling
and spline coupling (slide 9b). `schematic_layers.py` holds the hand-set layers both
figures draw from, and `ksiam_schematics.ipynb` renders them to `results/ksiam/`. The
design was settled in the grilling rounds of 2026-11-04 (Project document
`claude/inn-schematics-handoff-2026-11-04.md`) and revised in four review rounds on
the renders the same day; the decisions a reader of the code would ask about are in
*Design*.

## Files

- `docs/changes/2026-11-04-inn-schematics.md` — this document.
- `docs/index.md` — a row for this change document.
- `src/deep_isochron/plotting/coupling_warp.py` — new: the slide-9b figure, with the
  hand-set affine and spline coupling layers, the warp-curve and warped-grid drawers,
  `plot_coupling_warps`, `WarpLabels` and `WarpStyle`.
- `src/deep_isochron/plotting/__init__.py` — exports the seven names of
  `coupling_warp` and names the figure in the package docstring.
- `tests/test_plotting.py` — a `coupling_warp` section: the layers round-trip and
  move points, the spline band is steep, knots are marked only for splines, fixed
  grid lines stay vertical, the style and labels, and the four panels.
- `src/deep_isochron/plotting/inn_stack.py` — new: the slide-9a figure, with the
  hand-set linear layer `schematic_linear`, `plot_inn_stack`, `StackLabels` and
  `StackStyle`.
- `src/deep_isochron/plotting/__init__.py` — exports the four names of `inn_stack`
  and names the figure in the package docstring.
- `tests/test_plotting.py` — an `inn_stack` section: the linear layer rotates and
  stretches with a positive determinant, the labels and style, and the figure's
  blocks, arrows, labels and inset axes.
- `ksiam_schematics.ipynb` — the notebook that renders both schematics to `results/ksiam/` (new; unexecuted, since the cloud session has no notebook executor; the orchestrator ran the same code as a script).
- `src/deep_isochron/plotting/deformation/figures.py` — `draw_trajectories`, the
  trajectory block of `draw_sheet` factored out for the grid panels of
  `coupling_warp`; `draw_sheet` calls it.
- `src/deep_isochron/plotting/deformation/__init__.py` — exports `draw_trajectories`.
- `src/deep_isochron/plotting/coupling_warp.py` — the figure as 2 × 3: the identity
  warp in the first column, a lower row of one `schematic_sheet` with Bautin
  trajectories (`schematic_trajectories`) before and after each layer, drawn by
  `draw_grid_sheet` in place of `draw_warped_grid`; one set of limits on all six
  panels, rcParams spines on the warp panels, Okabe–Ito green as the moved color.
- `src/deep_isochron/plotting/inn_stack.py` — the coupling inset draws one
  `schematic_sheet` and its image with `draw_grid_sheet`; `_identity_coupling` goes.
- `src/deep_isochron/plotting/__init__.py` — exports `draw_grid_sheet`,
  `draw_identity_warp`, `schematic_sheet`, `schematic_trajectories` and
  `draw_trajectories`, and drops `draw_warped_grid`.
- `tests/test_plotting.py` — `draw_trajectories`' start triangles, the trajectories
  fitting the box, the warp panels' spines and ticks, the grid sheet's vertical fixed
  lines and triangles, and the six panels with equal limits.
- `ksiam_schematics.ipynb` — the 9b cell passes `schematic_trajectories((-1.0, 1.0))`
  and its heading names the three columns.
- `src/deep_isochron/plotting/coupling_warp.py` — the trajectories of `WarpStyle.sheet`
  take four Okabe–Ito colors, blue, reddish purple, orange and sky blue, so none is
  the green or vermilion accent.
- `src/deep_isochron/plotting/inn_stack.py` — tall blocks, the chiral glyph removed,
  and under each block type a before and an after panel of one `schematic_sheet`
  with Bautin trajectories mapped through that layer; `plot_inn_stack` takes the
  trajectories; Okabe–Ito vermilion for the linear block and `WarpStyle.moved_color`
  for the coupling block.
- `src/deep_isochron/plotting/coupling_warp.py` — the grid panels jit the bound
  `layer.__call__`, as `stage_sheets` does; the `draw_grid_sheet` docstring says so.
- `tests/test_plotting.py` — the insets' start triangles and the after-panels' block
  colors, the coupling color equal to the warp figure's, and `inset_cells` checked.
- `ksiam_schematics.ipynb` — the 9a cell passes `schematic_trajectories((-1.0, 1.0))`.
- `src/deep_isochron/plotting/coupling_warp.py` — `WarpStyle` draws the grids and
  warp curves in two grays, `moved_color = "0.3"` and `fixed_color = "0.75"`, so the
  colored trajectories stand out.
- `src/deep_isochron/plotting/inn_stack.py` — the insets take the gray `WarpStyle`;
  a frame of `StackStyle.frame_width` in the block's color surrounds each
  after-panel; `coupling_color` is the literal Okabe–Ito bluish green.
- `tests/test_plotting.py` — the after-panels carry a frame in the block color and
  the before-panels none, the moved lines are gray, and the accent test is deleted.
- `src/deep_isochron/plotting/coupling_warp.py` — `schematic_spline_coupling` builds
  its raw vector from a knot layout: a steep band whose center moves with $x_1$ and a
  pinch fixed in $x_2$; `draw_grid_sheet` clips its grid lines and trajectories to the
  box, adds its line collections without autoscaling and sets the limits last.
- `tests/test_plotting.py` — the six panels' limits after drawing, the limits and clip
  box of a map that leaves the box, the spline's band and pinch, the band moving with $x_1$, and the
  layout's range checks.
- `src/deep_isochron/plotting/schematic_layers.py` — new: `schematic_linear`,
  `schematic_affine_coupling`, `schematic_spline_coupling`, `schematic_trajectories`
  and `schematic_sheet`, moved unchanged from the two figure modules.
- `src/deep_isochron/plotting/coupling_warp.py` — the four schematic builders move
  out; the module docstring says where the layers come from.
- `src/deep_isochron/plotting/inn_stack.py` — `schematic_linear` moves out; the module
  docstring says where the layers come from.
- `src/deep_isochron/plotting/__init__.py` — the five builders are exported from
  `schematic_layers`.
- `tests/test_plotting.py` — the tests of the builders move into a `schematic_layers`
  section, unchanged.
- `src/deep_isochron/plotting/coupling_warp.py` — `WarpStyle.clip_pad`, 0.04 of the
  box width on each side of the clip rectangle, so the edge lines and the outermost
  start triangle draw whole.
- `tests/test_plotting.py` — the clip box is the box widened by `clip_pad`, and a
  negative `clip_pad` raises.
- `docs/architecture.md` — the three new plotting modules in the tree.
- `docs/roadmap.md` — the Done row of `inn-schematics`.
- `.claude/agents/implementer.md` — new files are staged before `just check`, which skips untracked files.
- `src/deep_isochron/plotting/schematic_layers.py` — `schematic_affine_coupling` takes
  a raw log-scale `log_scale_offset + scale_amplitude · sin(π x₁)` and a shift
  `shift_curvature · x₁²`, so its image lines curve and fan while every column keeps
  equal spacing; the module docstring names spacing within a column as the contrast.
- `tests/test_plotting.py` — the affine image keeps equal spacing within each column,
  differs between columns and bends a line of constant $x_2$; a negative amplitude
  raises; the map that leaves the box uses `log_scale_offset=2.0`.

## Design

- **Hand-set layers through the real classes, not sketches.** The schematic layers
  are `CouplingFlow`s over `Affine` and `MonotonicRQSpline` whose conditioners are
  plain functions of `x₁`, and a `BiLipschitzLinear` built with `eqx.tree_at`. A
  sketch would not have the knots where the model has them, and the figures must
  show what the model does. The spline is built from a desired knot layout (a steep
  band whose center moves with `x₁`, a pinch fixed in `x₂`), converted to raws with
  the inverses of the template's own constraints; the raw scalars of the first
  version could not make a band that reads from a projector.
- **The affine example is nonlinear in `x₁`.** An affine coupling layer bends grid
  lines as much as its conditioner does; what it cannot do is vary the spacing
  within a column. With a linear conditioner the affine column was straight and the
  contrast read as "straight versus curved" (Joon's review); the sinusoidal
  log-scale and quadratic shift make the lines curve and fan while every column
  keeps equal spacing.
- **The grid figures show the image of the regular grid** (lines of constant `x₂`
  in the domain, mapped), not the preimage: a steep warp stretches, so the picture
  matches the warp curve drawn above it. Figure A4 on slide 11 keeps the same
  convention.
- **One sheet drawer, mapping is the caller's.** `draw_grid_sheet(ax, sheet, box,
  style)` draws any `Sheet`, domain or image, so the before panel, the after panels
  and the `inn_stack` insets are one function; the caller maps with
  `sheet.map(eqx.filter_jit(layer.__call__))`, the bound method, as `stage_sheets`
  does, since `eqx.filter_jit` reads a function's `__name__`. `draw_trajectories` is
  factored out of `draw_sheet` for its second adopter. `Sheet` does not store its
  box and a mapped sheet's corners no longer give it, so the drawer takes `box`.
- **Every grid panel clips to the box plus `clip_pad`.** The affine image leaves the
  box; clipping to the box keeps the six panels the same visual size, and the pad
  keeps the edge lines and the outermost start triangle whole. The collections are
  added without autoscaling and the shared limits are set last.
- **Neutral grids; the type color on the blocks, the frames and the type names.**
  Colored grid lines clashed with the colored trajectories (Joon's second review).
  Moved lines are dark gray, fixed lines light gray, warp curves dark gray with the
  knots marked; in `inn_stack` each after-panel carries a frame in its block's color
  (Okabe–Ito vermilion for linear, bluish green for coupling) and the trajectories
  take the rest of the Okabe–Ito set. The hex values are style fields, placeholders
  for the deck palette.
- **No sentences in the figures.** Only element-bound labels (column headers, row
  labels, block names, `H`, `H⁻¹`, endpoint words, type names), every one a field of
  `WarpLabels` or `StackLabels`, every font size a field of the style; captions are
  PowerPoint text boxes. Labels must survive a re-export.
- **Six blocks with symbolic last indices.** The depth varies across experiments
  (16 layers in `configs/model/conjugacy.yaml`); Joon says the number.
- **Two modules and a layers module, not a subpackage.** Each figure is one module;
  the hand-set layers moved to `schematic_layers.py` when `coupling_warp.py` reached
  the 600-line limit.
- The `implementer` agent type was not registered in the session until its second
  turn, so units 1 and 2 ran as `general-purpose` subagents on Opus with the agent
  file as their first read; units 4 to 8 ran as `implementer`.
- **No ADR.** Joon: nothing here is hard to reverse; a single ADR on the
  algorithm-visualization strategy may come later.

Ledger (one commit each, fix-ups folded into their unit):

1. [x] `plotting/coupling_warp.py`: the hand-set coupling constructors, the warp-curve
   and warped-grid drawers, `plot_coupling_warps`, labels and style, tests.
2. [x] `plotting/inn_stack.py`: `schematic_linear`, the chain with its insets,
   `plot_inn_stack`, labels and style, tests.
3. [x] `ksiam_schematics.ipynb` rendering both figures to `results/ksiam/`.
4. [x] Review 1: 2 × 3 with a before panel and trajectories; `draw_trajectories`
   factored out; tall blocks and before/after insets in `inn_stack`.
5. [x] Review 2: neutral grids, the type color as a frame on the after-panels.
6. [x] Review 3: grid panels clipped to the box; a moving band and a pinch in the
   spline; the layers moved to `schematic_layers.py`.
7. [x] Review 4: the affine example nonlinear in `x₁`.
8. [x] The documents (this commit).

## Bugs fixed

None.

## Tests

Per Joon's rule for plotting work (2026-11-04), only `tests/test_plotting.py` runs on
this branch, at the `dev` profile, before and after each commit; the full tier is not
run. Baseline on `cd8c439`: 30 passed in 67 s on two workers. After unit 1
(`d00d604`): 37 passed. After unit 2 (`abb2a4c`): 41. After review 1 (`a73b44f`): 46.
After the frames (`5b972bb`): 47 (one test deleted with the accent it pinned). After
the clipping and the knot layout (`dd2179c`): 50. After the affine change
(`ac36289`): 52 passed in 25 s. Each render was viewed at 100 dpi and at 25 percent of
the slide width before Joon's review; the band, the pinch and the frames read at that
scale.

## Open issues

- The captions of both slides are PowerPoint text boxes, delivered in chat, not in
  the figures.
- Deferred by the grilling round: the conditioner MLP in the picture (Joon adds it by
  hand); a cubic B-spline rendering of 9b (the knot-layout builder is written for
  `MonotonicRQSpline`'s raw layout).
- The `inn_stack` insets are 1.15 in because the 12 in figure has to hold the chain;
  `StackStyle.figsize` and `inset_size` are the knobs if the slide wants them larger.
- `conjugacy-schematic`'s commit 3 will add its own `docs/architecture.md` and
  roadmap lines; whichever of the two branches lands second resolves one conflict in
  the `plotting/` tree block.
- Equinox 0.13.8: a Module with an `init=False` static field (`smoothness` on every
  bijection) loses `__name__` after any pytree rebuild, so `eqx.filter_jit(module)`
  raises for a layer from `tree_at` or a checkpoint; `eqx.filter_jit(module.__call__)`
  is the project's way, now also in `code-conventions`.

## Review notes

One row per *Files* bullet; *Thoughts* and *Modifications* are Joon's.

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/changes/2026-11-04-inn-schematics.md` — this document | | |
| `docs/index.md` — the `inn-schematics` row | | |
| `src/deep_isochron/plotting/coupling_warp.py` — the slide-9b figure | | |
| `src/deep_isochron/plotting/__init__.py` — the `coupling_warp` exports | | |
| `tests/test_plotting.py` — the `coupling_warp` section | | |
| `src/deep_isochron/plotting/inn_stack.py` — the slide-9a figure | | |
| `src/deep_isochron/plotting/__init__.py` — the `inn_stack` exports | | |
| `tests/test_plotting.py` — the `inn_stack` section | | |
| `ksiam_schematics.ipynb` — renders both schematics | | |
| `src/deep_isochron/plotting/deformation/figures.py` — `draw_trajectories` factored out | | |
| `src/deep_isochron/plotting/deformation/__init__.py` — the `draw_trajectories` export | | |
| `src/deep_isochron/plotting/coupling_warp.py` — the 2 × 3 figure with a before panel and trajectories | | |
| `src/deep_isochron/plotting/inn_stack.py` — the coupling inset through `draw_grid_sheet` | | |
| `src/deep_isochron/plotting/__init__.py` — the new `coupling_warp` and `draw_trajectories` exports | | |
| `tests/test_plotting.py` — the reworked `coupling_warp` tests and `draw_trajectories` | | |
| `ksiam_schematics.ipynb` — the 9b cell passes the trajectories | | |
| `src/deep_isochron/plotting/coupling_warp.py` — trajectory colors avoid the accents | | |
| `src/deep_isochron/plotting/inn_stack.py` — tall blocks and before/after insets of a sheet | | |
| `src/deep_isochron/plotting/coupling_warp.py` — layers jitted through the bound `__call__` | | |
| `tests/test_plotting.py` — the inset and accent tests of `inn_stack` | | |
| `ksiam_schematics.ipynb` — the 9a cell passes the trajectories | | |
| `src/deep_isochron/plotting/coupling_warp.py` — gray grids and warp curves | | |
| `src/deep_isochron/plotting/inn_stack.py` — the after-panel frames in the block colors | | |
| `tests/test_plotting.py` — the frame and gray-grid tests of `inn_stack` | | |
| `src/deep_isochron/plotting/coupling_warp.py` — grid panels clipped to the box; a moving band and a pinch in the spline | | |
| `tests/test_plotting.py` — limits and clipping after drawing, band and pinch | | |
| `src/deep_isochron/plotting/schematic_layers.py` — the hand-set layers and the sheet | | |
| `src/deep_isochron/plotting/coupling_warp.py` — the builders move out | | |
| `src/deep_isochron/plotting/inn_stack.py` — `schematic_linear` moves out | | |
| `src/deep_isochron/plotting/__init__.py` — the builders from `schematic_layers` | | |
| `tests/test_plotting.py` — the `schematic_layers` section | | |
| `src/deep_isochron/plotting/coupling_warp.py` — the clip rectangle padded by `clip_pad` | | |
| `tests/test_plotting.py` — the padded clip box and the `clip_pad` check | | |
| `src/deep_isochron/plotting/schematic_layers.py` — the affine conditioner nonlinear in $x_1$ | | |
| `tests/test_plotting.py` — uniform spacing per affine column | | |
| `docs/architecture.md` — the plotting tree | | |
| `docs/roadmap.md` — the Done row | | |
| `.claude/agents/implementer.md` — staging new files before the check | | |
