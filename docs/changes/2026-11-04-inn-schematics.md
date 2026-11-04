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

## Design

Written in the documents commit, from the handoff's decisions table.

Ledger (one commit each):

1. [ ] `plotting/coupling_warp.py`: the hand-set coupling constructors, the warp-curve
   and warped-grid drawers, `plot_coupling_warps`, labels and style, tests.
2. [ ] `plotting/inn_stack.py`: `schematic_linear`, the chain with its insets,
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
