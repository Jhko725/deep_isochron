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

- `src/deep_isochron/plotting/deformation.py` (**new**) — `Sheet` with `map`,
  `build_sheet`, `CellValues` and `cell_values`, `stage_sheets`, `stage_labels` and
  `sheet_limits`: the geometry of the deformation figures.
- `tests/test_plotting.py` — seven tests of the geometry: the identity, a linear map
  against NumPy's SVD of the matrix, the folded-cell error, the stages in both
  directions, the labels and the padded limits.
- `tests/helpers.py` — the `TOL` entry `cell_values_affine`.
- `tests/strategies.py` — `planar_matrices`, orientation-preserving `2×2` matrices
  with bounded entries and determinant.
- `src/deep_isochron/plotting/deformation.py` — `FillScale`, `area_scale` and
  `anisotropy_scale`, `SheetStyle`, the private `_draw_sheet` and `plot_deformation`:
  the fill scale and the domain | image pair.
- `tests/test_plotting.py` — four tests of the pair: the symmetric area scale and its
  ticks, the zeros left out of the anisotropy pool, the percentile check, and the
  pair figure's panels.
- `src/deep_isochron/plotting/sheet.py` (**new**) — the geometry of the first bullet
  moved out of `deformation.py` unchanged: `Sheet` with `map`, `build_sheet`,
  `CellValues` and `cell_values`, `stage_sheets`, `stage_labels` and `sheet_limits`.
- `src/deep_isochron/plotting/deformation.py` — the geometry removed, and its module
  docstring and imports reduced to the scales, the style and the renderers.
- `tests/test_plotting.py` — the geometry names import from `plotting.sheet`; no test
  body changes.
- `src/deep_isochron/plotting/deformation.py` — `plot_deformation_strip` and
  `animate_deformation` on `_draw_sheet`, the panel format and colorbar factored into
  private helpers that all three renderers call, and the pair's panel titles and axis
  labels.
- `src/deep_isochron/plotting/__init__.py` — the names of `plotting.sheet` and
  `plotting.deformation` exported.
- `tests/test_plotting.py` — three tests: the strip's panels and limits, the
  animation's frame count through a GIF, and the length check.

## Design

The ledger of the change, in the order the commits will follow. Each item becomes a
*Files* bullet and a review row when its commit lands.

1. `plotting/deformation.py`: `Sheet` with `map`, `build_sheet`, `CellValues` and
   `cell_values`, `stage_sheets`, `stage_labels`, `sheet_limits`; their law tests.
2. `plotting/deformation.py`: `FillScale`, `area_scale`, `anisotropy_scale`,
   `SheetStyle`, `plot_deformation`; the smoke test — done.
3. `plotting/deformation.py`: `plot_deformation_strip`, `animate_deformation`; the
   exports in `plotting/__init__.py`; the smoke tests — done.
4. `plotting/sheet.py`: the geometry of item 1 moved out of `deformation.py`, which had reached the 600-line module limit at item 2 (decided 2026-10-30). — done.

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
| `src/deep_isochron/plotting/deformation.py` — the geometry: `Sheet`, `build_sheet`, `cell_values`, `stage_sheets`, `stage_labels`, `sheet_limits` | Not happy with the abstraction. <ul>The biggest problem is the api of the plotting functions. For example, `plot_deformation_strip` requires the user to pass sheets, values, scale, and labels, which must all be constructed from `build_sheet`, `stage_sheets`, `cell_values`, etc. This is too much complexity for the user. The function should have taken `transforms: Sqeuence[AbstractBijection]`, `trajectories`, some config object for the sheets (how many cells, circles, etc), some argument that specifies the metric (area change or log anisotropy), and plotting configuration data. The users' focus is on the abstractions regarding the numerics of the project. Placing a bunch of plotting-specific abstractions on top makes the plotting functions very hard to use. If any abstractions are required for plotting, they should be opt-into, for a greateer flexibility of plotting customization (ex. for using different metric for cell coloring). <li> `build_sheet` is essentially the constructor logic of `Sheet` since nobody is going to create the point coordinates for the circles (the right abstraction level is at circles; not the points used to render them). Why not attach it there? <li>`Sheet.map` is essentially just multiple applications of vmapped bijection operations. Why not just make `Sheet` into a equinox Module (it already interacts with JAX part of the code anyways) and use `jax.tree.map` with `eqx.filter_vmap`? If the number of axes to vmap differ for different components, this is easily addressed using `einops.wrap` and `unwrap`. <li> As discussed with Claude, shoelace formula to compute area change gives negative area change for cells at high distortion areas. Should switch to Jacobian determinant. <ul> | To be discussed with Claude. |
| `tests/test_plotting.py` — the seven geometry tests | Skippig for now since the plotting functions will need a overhaul. | None for now. |
| `tests/helpers.py` — `TOL["cell_values_affine"]` | Skippig for now since the plotting functions will need a overhaul. | None for now. |
| `tests/strategies.py` — `planar_matrices` | Skippig for now since the plotting functions will need a overhaul. | None for now. |
| `src/deep_isochron/plotting/deformation.py` — the fill scale and the pair: `FillScale`, `area_scale`, `anisotropy_scale`, `SheetStyle`, `_draw_sheet`, `plot_deformation` | See complaints above. Also, `area_scale` and `fill_scale` are mostly doing bad value check and percentile-based normalization. This is not related to the actual metric (no metric is being computed), so does not make sense to have as two functions. Further housekeeping like tick and ticklabel customization can just be done on the user side, or as a convenicence function on `FillScale`. `_set_limits` and `_add_colorbar` look like general utility functions for the plotting submodule. Also, it's annoying for the user to pass `np.arrays` for lo and hi. Instead just accept `Sequence[float]`s and verify inside `_set_limits`. | To be discussed. |
| `tests/test_plotting.py` — the four tests of the scales and the pair | Skippig for now since the plotting functions will need a overhaul. | None for now. |
| `src/deep_isochron/plotting/sheet.py` — the geometry moved from `deformation.py` | See complaints above. | To be discussed with Claude. |
| `src/deep_isochron/plotting/deformation.py` — the geometry removed | Agree that long files should be split. How to split should be rediscussed after fixing the problems above. | To be discussed. |
| `tests/test_plotting.py` — the geometry imports from `plotting.sheet` | Skippig for now since the plotting functions will need a overhaul. | None for now. |
| `src/deep_isochron/plotting/deformation.py` — the strip, the animation, and the pair's titles and axis labels | See complaints above. | To be discussed. |
| `src/deep_isochron/plotting/__init__.py` — the deformation exports | Simple change. | None |
| `tests/test_plotting.py` — the three tests of the strip and the animation | Skippig for now since the plotting functions will need a overhaul. | None for now. |
