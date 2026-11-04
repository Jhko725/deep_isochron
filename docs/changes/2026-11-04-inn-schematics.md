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

## Design

Written in the documents commit, from the handoff's decisions table.

Ledger (one commit each):

1. [x] `plotting/coupling_warp.py`: the hand-set coupling constructors, the warp-curve
   and warped-grid drawers, `plot_coupling_warps`, labels and style, tests.
2. [x] `plotting/inn_stack.py`: `schematic_linear`, the chain with its insets,
   `plot_inn_stack`, labels and style, tests.
3. [ ] `ksiam_schematics.ipynb` rendering both figures to `results/ksiam/`.

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
