---
type: change
status: in progress
updated: 2026-11-04
---

# `inn-schematics` — the two INN schematics of KSIAM slides 9a and 9b

## Summary

The branch adds two figures to `plotting` for the KSIAM 2026 talk: `inn_stack.py`
draws the invertible network as a chain of alternating linear and spline coupling
layers with its exact inverse (slide 9a), and `coupling_warp.py` draws the spline
coupling layer as a one-dimensional monotone warp beside an affine foil (slide 9b).
The design was settled in the grilling rounds of 2026-11-04 and is in the Project
document `claude/inn-schematics-handoff-2026-11-04.md`; the decisions a reader of the
code would ask about are in *Design*, written when the branch lands.

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

## Design

Written in the documents commit, from the handoff's decisions table.

Ledger (one commit each):

1. [x] `plotting/coupling_warp.py`: the hand-set coupling constructors, the warp-curve
   and warped-grid drawers, `plot_coupling_warps`, labels and style, tests.
2. [x] `plotting/inn_stack.py`: `schematic_linear`, the chain with its insets,
   `plot_inn_stack`, labels and style, tests.
3. [x] `ksiam_schematics.ipynb` rendering both figures to `results/ksiam/`.

## Bugs fixed

None.

## Tests

Per Joon's rule for plotting work (2026-11-04), only `tests/test_plotting.py` runs on
this branch, at the `dev` profile, before and after each commit; the full tier is not
run. Baseline on `cd8c439`: 30 passed in 67 s on two workers.

## Open issues

- The captions of both slides are PowerPoint text boxes, delivered in chat, not in
  the figures.

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
