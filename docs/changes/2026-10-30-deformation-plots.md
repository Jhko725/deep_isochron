---
type: change
status: in progress (unit 1 of 3 next)
updated: 2026-10-30
---

# `deformation-plots` — rubber-sheet figures of a planar bijection

## Summary

Pending: written in the documents commit. The branch adds `plotting/deformation.py`,
which draws how any map of the plane stretches a mesh and a batch of trajectories, as a
domain | image pair, a strip of stages and a flipbook animation, with the per-cell area
factor or anisotropy as the fill. The design is the Project document
`claude/deformation-plots-handoff-2026-10-30.md`.

## Files

## Design

The ledger of the change, in the order the commits will follow. Each item becomes a
*Files* bullet and a review row when its commit lands.

1. `plotting/deformation.py`: `Sheet` with `map`, `build_sheet`, `CellValues` and
   `cell_values`, `stage_sheets`, `stage_labels`, `sheet_limits`; their law tests.
2. `plotting/deformation.py`: `FillScale`, `area_scale`, `anisotropy_scale`,
   `SheetStyle`, `plot_deformation`; the smoke test.
3. `plotting/deformation.py`: `plot_deformation_strip`, `animate_deformation`; the
   exports in `plotting/__init__.py`; the smoke tests.

The decisions and their rejected alternatives are written in the documents commit.

## Bugs fixed

None.

## Tests

Pending: the fast tier of `tests/test_plotting.py` runs before each commit; the full
tier runs once before the documents commit. At `913c52e` (the branch point)
`uv run just test dev tests/test_plotting.py` is `5 passed`.

## Open issues

Pending.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
