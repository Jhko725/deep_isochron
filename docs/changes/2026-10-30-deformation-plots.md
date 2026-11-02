---
type: change
status: landed 2026-11-02 (units 1–3, the review of 2026-10-31, units 5–8, the documents); review of units 5–8 and the documents pending
updated: 2026-11-02
---

# `deformation-plots` — rubber-sheet figures of a planar bijection

## Summary

The branch adds the deformation figures: how a bijection of the plane — the
conjugacy's INN, layer by layer — stretches a mesh and a batch of trajectories, as a
domain | image pair (`plot_deformation`), a strip of stages
(`plot_deformation_strip`) and a flipbook animation (`animate_deformation`), with the
area factor or the anisotropy of the map at each cell as the fill. The three figure
functions take the bijection's layers and the trajectories and default everything
else; the sheet, the metric and the drawing function are public for a custom figure.
The package is `plotting/deformation/` (`sheet`, `metrics`, `style`, `figures`) with
`clipped_norm` in `plotting/scales.py` and two axes helpers in `plotting/style.py`.

The work ran in two rounds. The first (units 1–4, Project document
`claude/deformation-plots-handoff-2026-10-30.md`) landed a lower-level API; Joon's
review of 2026-10-31 (the first twelve review rows) rejected its shape, and the second
round (units 5–8, `claude/deformation-plots-handoff-2-2026-10-31.md`) rewrote it on the
same branch, then folded in a coworker's review of the FHN flipbook and a style
comparison. The first round's commits stay in the history; nothing of their code
survives.

Reversible: one `git revert` per commit; no data, hash or checkpoint changes. The
`video` extra adds an optional dependency only.

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
- `src/deep_isochron/plotting/deformation/sheet.py` (**new**) — `SheetConfig`,
  `Sheet` as an `eqx.Module` built by its constructor with `map` as a tree map,
  `stage_sheets`, `stage_maps`, `stage_labels` with the `⁻¹` marker, `cell_jacobians`
  and `sheet_limits` as four floats.
- `src/deep_isochron/plotting/deformation/metrics.py` (**new**) — `CellMetric` and its
  instances `AREA` and `ANISOTROPY`, from the Jacobian at the cell centers.
- `src/deep_isochron/plotting/deformation/__init__.py` (**new**) — the subpackage's
  docstring and the re-exports of its ten names.
- `src/deep_isochron/plotting/scales.py` (**new**) — `clipped_norm`, the quantile
  clipping of a color range.
- `src/deep_isochron/plotting/sheet.py` (**deleted**) — replaced by
  `plotting/deformation/sheet.py`.
- `src/deep_isochron/plotting/deformation.py` (**deleted**) — shadowed by the
  subpackage; the figure functions return on the new API in the next commit.
- `src/deep_isochron/plotting/__init__.py` — the two old import blocks replaced by the
  subpackage's names and `clipped_norm`.
- `tests/test_plotting.py` — the geometry, scale and figure tests replaced by twelve
  tests of the config and box checks, the identity map, the two metrics on the
  identity, on a linear map and on a reflection, the stages and stage maps, the labels,
  the limits and the clipped norm.
- `tests/helpers.py` — `TOL["cell_values_affine"]` renamed `TOL["cell_metric_affine"]`,
  with its reason reworded.
- `src/deep_isochron/plotting/deformation/figures.py` (**new**) — `SheetStyle` with
  `color_limit_quantile` and the figure sizes, the public `draw_sheet`, and
  `plot_deformation`, `plot_deformation_strip` and `animate_deformation` on the
  bijection's layers and the trajectories.
- `src/deep_isochron/plotting/style.py` — `set_limits` and `add_colorbar`, the panel's
  box and format and the ticked colorbar, as shared helpers.
- `src/deep_isochron/plotting/deformation/__init__.py` — the re-exports of the figure
  names.
- `src/deep_isochron/plotting/__init__.py` — the figure names, `set_limits` and
  `add_colorbar` exported.
- `tests/test_plotting.py` — five tests: the pair's panels, the strip's stage selection
  and range check, the animation's frame count through a GIF, a custom metric through
  `draw_sheet`, and the limits check.
- `prototype_analysis.ipynb` — the three cells that called the old API call
  `plot_deformation` and `plot_deformation_strip`; the five cells of the shoelace
  investigation deleted.
- `src/deep_isochron/plotting/deformation/metrics.py` — `AREA` ticks at 0.5 and 2 as
  well, so that a narrow color range still carries ticks besides ×1.
- `src/deep_isochron/plotting/deformation/figures.py` — the private `_factor_ticks`,
  which ticks the colorbar from the norm's range by the 1–3–10 series with the 1–2–5
  series and `MaxNLocator` as fallbacks; the colorbar layout fields of `SheetStyle`;
  `inverse=True` plays the forward stages backward from the deformed sheet in all
  three figures; the MP4 save call at 24 fps in `animate_deformation`'s docstring.
- `src/deep_isochron/plotting/deformation/sheet.py` — the private `_stage_sequence`,
  the sheets, maps and labels of the stages that a figure draws: the forward stages,
  or the forward stages reversed with the inverse maps and labels; in this module
  rather than `figures.py`, which it would push over the 600-line limit.
- `src/deep_isochron/plotting/deformation/metrics.py` — `CellMetric.tick_values`
  removed, with its values in `AREA` and `ANISOTROPY`; `CellMetric.log_scale`, which
  decides whether the colorbar is ticked by the factor rule or by matplotlib's default
  ticks.
- `src/deep_isochron/plotting/style.py` — `add_colorbar` takes the shrink, aspect,
  pad, tick font size and label placement as keyword arguments, aligns the title to
  the bar's right edge, keeps matplotlib's ticks when given none, and draws no tick
  marks.
- `tests/test_plotting.py` — three tests: the tick rule on three ranges, the inverse
  sequence against the reversed forward sheets with the negated area fill, and the
  colorbar label placement; the pair test checks the colorbar title.
- `pyproject.toml` — the optional `video` dependency group with `imageio-ffmpeg`.
- `uv.lock` — changed by `uv add` for the new optional `video` group only.
- `CLAUDE.md` — the repository fact on MP4 export and `ffmpeg`.
- `src/deep_isochron/plotting/deformation/style.py` (**new**) — `SheetStyle`, moved from
  `figures.py` for the module limit, with `curve_colors` as one color, a sequence or a
  `ListedColormap` and the defaults `fill_alpha=0.5`, `curve_colors="0.1"`; the private
  `_curve_colors`, which gives one color per trajectory.
- `src/deep_isochron/plotting/deformation/figures.py` — `SheetStyle` removed and
  imported from `.style`; `draw_sheet` takes the trajectories' colors from
  `_curve_colors`.
- `src/deep_isochron/plotting/deformation/__init__.py` — `SheetStyle` re-exported from
  `.style`.
- `tests/test_plotting.py` — `test_curve_colors_forms` on the three forms and a rejected
  value; the custom-metric test passes `SheetStyle(fill_alpha=0.25)` explicitly.

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
5. `plotting/deformation/sheet.py`, `metrics.py`, `plotting/scales.py`: the rewrite of items 1–2 after the review of 2026-10-31 — `Sheet` an `eqx.Module` built by its constructor, the fill from the Jacobian at the cell centers through a `CellMetric`, `clipped_norm` as a utility; both old modules deleted, the figures return in item 6 — done.
6. `plotting/deformation/figures.py`, `plotting/style.py`: the figures on the new API — `plot_deformation`, `plot_deformation_strip`, `animate_deformation` on `transforms` and `trajectories`, public `draw_sheet`, `SheetStyle` with `color_limit_quantile`, `set_limits` and `add_colorbar` as shared helpers; the notebook cells follow — done.
7. Coworker's review of the FHN flipbook (2026-11-02): colorbar ticks by a 1–3–10 / 1–2–5 rule with a tunable layout; `inverse=True` plays the forward sequence backward from the deformed sheet; MP4 export at 24 fps with `imageio-ffmpeg` as the `video` extra — done.
8. `curve_colors` as one color, a sequence or a `ListedColormap`; defaults `fill_alpha=0.5`, `curve_colors="0.1"` (Joon's pick of the style experiment, 2026-11-02); `SheetStyle` moved to `deformation/style.py` for the module limit — done.

The decisions, each with the alternative that was rejected:

- **The figure functions take the numerics' objects.** `plot_deformation(transforms,
  trajectories)` and its two siblings build the sheet, the stages, the fill and the
  color scale themselves. Rejected: the first round's API, where the caller composed
  `build_sheet`, `stage_sheets`, `cell_values`, `area_scale` and the renderer by hand
  (review row 1: "too much complexity for the user"). The composition is one place, and
  the pieces stay public as the opt-in layer.
- **`Sheet` is an `eqx.Module` whose constructor builds its point sets.** It passes
  through `vmap`, which the container rule (`coding-standards.md` § Containers) makes a
  pytree, and `Sheet.map` is a `jax.tree.map` over its leaves. Rejected: a frozen
  NumPy dataclass with a free `build_sheet` (round 1), which misread both the
  container rule and the `build_*` convention — the latter exists so that `model/` and
  `training/` never see configs, not for constructors.
- **The fill is the metric of the map at the cell center.** `cell_jacobians` evaluates
  `jax.jacfwd` there; `AREA` is log|det J| and `ANISOTROPY` log σ_max/σ_min. Rejected:
  the shoelace area of the four corner images and the affine fit to them (round 1).
  On the FHN model that area went negative in one cell at 24² and two at 48², because
  a straight-sided quad cannot represent a cell whose image bends inside it, while a
  diffeomorphism's Jacobian is defined and non-singular everywhere. The sub-quad fill,
  the mesh polylines and the circles are still mapped points, so the drawn geometry is
  exact; only the color comes from the center.
- **The metric is a value, `CellMetric(fn, cmap, symmetric, label, log_scale)`.** A
  notebook defines a third metric as an instance. Rejected: a `Literal["area",
  "anisotropy"]` selector with an `if` per metric in the renderer, and a `CellValues`
  record carrying both scalars.
- **`clipped_norm` is a utility in `plotting/scales.py`; the quantile is
  `SheetStyle.color_limit_quantile`.** Rejected: `area_scale`/`anisotropy_scale`, two
  functions that shared the clipping and differed only in the metric's constants
  (review row 5), and a required `scale` argument on every renderer.
- **Colorbar ticks come from the norm's range** by the 1–3–10 series, then 1–2–5, then
  a rounded locator; `log_scale=False` keeps matplotlib's ticks. Rejected: a tick list
  on the metric (unit 7's `tick_values`), which crowded at log spacing once the list
  mixed the two series.
- **`inverse=True` plays the forward sequence backward**: the sheet starts deformed
  and the inverse layers peel off last-first until the mesh is regular, with the fill
  relative to the deformed start (a coworker's request, 2026-11-02). Rejected: a
  regular mesh laid in the base plane and pulled back, which is the isochron picture
  and stays available through `Sheet` on a base-plane box with `stage_sheets(...,
  inverse=True)`.
- **A tinted mesh is drawn opaque with the colormap pre-blended over white**, never
  with `alpha=`: under a real alpha the pixels on shared sub-quad edges are painted
  twice and show as seams.
- **Style defaults are the bold fill with neutral curves** (`fill_alpha=0.5`,
  `curve_colors="0.1"`), Joon's pick from three variants rendered on the prototype
  scene on 2026-11-02; `curve_colors` also takes a sequence or a `ListedColormap`, so
  the earlier look is `SheetStyle(fill_alpha=0.25, curve_colors=cc.cm.glasbey_dark)`.
- **A flipbook, not a tween.** The animation shows the stages themselves, one per
  second, and the MP4 is written at 24 fps by duplicating frames (`-r 24`) because
  players handle a 1 fps stream badly. Rejected for now: a convex interpolation
  between stages, which is not a diffeomorphism and can fold.
- **Subpackage `plotting/deformation/`** of four modules, each under the 600-line
  limit, split by what serves what; `clipped_norm` in `plotting/scales.py` rather than
  a `utils` module.

None of these meets the ADR test: every one is a revert away.

## Bugs fixed

None.

## Tests

- `uv run just check`: clean on every commit of the branch.
- The full tier, `uv run just test default` at `4107760` (two cores, 7 GB, JAX's
  persistent compilation cache on): `684 passed, 10 skipped, 1 xfailed in 1396 s`. The
  skips are the `auto` tests without an AUTO-07p tree; the `xfail` is the pinned
  `sinh` second-derivative case of `tests/test_analytic.py`, unrelated. The branch
  point `913c52e` was last recorded at `673 passed, 1 xfailed`
  (`2026-10-30-phase-turns.md`); the eleven new passes are this branch's.
- `tests/test_plotting.py` at the `dev` profile: `26 passed` (5 isochron tests from
  before the branch, 21 of this branch). Per unit, before its commit: unit 1 `12`, unit
  2 `16`, units 3a–3b `16`/`19`, unit 5 `17`, unit 6 `22`, unit 7 `25`, unit 8 `26`.
- What the branch's tests pin: `SheetConfig` and `Sheet` reject bad values and boxes;
  `Sheet.map` of the identity is the sheet, extra curves included; `AREA` and
  `ANISOTROPY` are zero on the identity and on a reflection, and equal NumPy's
  determinant and singular-value ratio on a linear map (Hypothesis over
  `planar_matrices`, to `TOL["cell_metric_affine"]`); `stage_sheets` has
  `len(layers) + 1` entries ending at `sheet.map(model)` in both directions, and
  `stage_maps` agrees with it stage by stage; the labels; the padded limits;
  `clipped_norm`'s symmetric range, zero exclusion and quantile check; the colorbar
  tick rule on three ranges; the inverse sequence equals the forward sheets reversed
  with the negated area fill; `curve_colors` in its three forms; and Agg smoke tests of
  the pair, the strip (stage selection and range check), the animation (frame count
  through a GIF), `draw_sheet` with a custom metric, `set_limits` and the colorbar
  label placement.
- Not covered by a test: the rendered pictures. Each unit's figures were rendered on
  the prototype scene (an untracked eight-layer INN at random initialization, seed 7,
  with twelve FHN trajectories) and viewed by the orchestrator against the prototype
  images before the unit was accepted; the MP4 was probed (`h264`, 1280×720,
  `yuv420p`, 24 fps).

## Open issues

- A tween between stages (the convex interpolation of the drawn points) is deferred:
  it is not a diffeomorphism and can fold; the flipbook is the honest sequence. Open
  until Joon has used the flipbook in a talk.
- The dense Jacobian views (a log|det J| heatmap on a fine grid, Tissot ellipses by
  linearization, the stretch-direction field) are a separate figure family, not
  started; `cell_jacobians` is the building block.
- The isochron pullback picture (a regular mesh in the base plane pulled back through
  the inverse layers) is reachable only through `Sheet` and `stage_sheets` by hand; a
  figure function for it waits for the isochron work to ask.
- The quantile clipping saturates on a trained map: the FHN flipbook's shrinking
  corners all read as the color limit from stage 6 on. `color_limit_quantile` is the
  knob; whether a per-stage range reads better is untried.
- Two gaps of the first handoff, recorded for the next one: a test fixture of freshly
  built layers is the identity (`BiLipschitzLinear` defaults to `init="identity"`,
  `CouplingFlow` zeroes its conditioner's final layer), so an endpoint test on it
  cannot fail — the fixture is perturbed with `tests/strategies.perturb`; and a
  container that passes through `vmap` is an `eqx.Module` by the rule even when
  matplotlib consumes it afterwards. The `build_*` convention of the `code-conventions`
  skill reads as if it covered every constructor; it covers construction from a
  config, and a proposal to narrow its wording goes to Joon with this branch.
- `eqx.filter_jit(layer)` on a module instance raises `TypeError: __name__ must be set`
  in the pinned equinox; `stage_sheets` jits the bound `__call__`. Worth a line in
  `coding-standards.md` if it recurs.

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
| `src/deep_isochron/plotting/deformation/sheet.py` — `SheetConfig`, `Sheet`, the stages, `cell_jacobians`, `sheet_limits` | | |
| `src/deep_isochron/plotting/deformation/metrics.py` — `CellMetric`, `AREA`, `ANISOTROPY` | | |
| `src/deep_isochron/plotting/deformation/__init__.py` — the subpackage docstring and re-exports | | |
| `src/deep_isochron/plotting/scales.py` — `clipped_norm` | | |
| `src/deep_isochron/plotting/sheet.py` — deleted | | |
| `src/deep_isochron/plotting/deformation.py` — deleted | | |
| `src/deep_isochron/plotting/__init__.py` — the subpackage exports and `clipped_norm` | | |
| `tests/test_plotting.py` — the twelve tests of the sheet, metrics, stages and clipped norm | | |
| `tests/helpers.py` — `TOL["cell_metric_affine"]` | | |
| `src/deep_isochron/plotting/deformation/figures.py` — `SheetStyle`, `draw_sheet` and the three figures on `transforms` and `trajectories` | | |
| `src/deep_isochron/plotting/style.py` — `set_limits` and `add_colorbar` | | |
| `src/deep_isochron/plotting/deformation/__init__.py` — the figure re-exports | | |
| `src/deep_isochron/plotting/__init__.py` — the figure names and the two helpers exported | | |
| `tests/test_plotting.py` — the five figure and helper tests | | |
| `prototype_analysis.ipynb` — the three cells on the new API | | |
| `src/deep_isochron/plotting/deformation/metrics.py` — the `AREA` ticks | | |
| `src/deep_isochron/plotting/deformation/figures.py` — the tick rule, the colorbar layout fields, the inverse played backward, the MP4 save call | | |
| `src/deep_isochron/plotting/deformation/sheet.py` — `_stage_sequence` | | |
| `src/deep_isochron/plotting/deformation/metrics.py` — `tick_values` removed, `log_scale` | | |
| `src/deep_isochron/plotting/style.py` — the `add_colorbar` layout arguments | | |
| `tests/test_plotting.py` — the tick, inverse and colorbar label tests | | |
| `pyproject.toml` — the `video` extra | | |
| `uv.lock` — the `video` extra | | |
| `CLAUDE.md` — the MP4 export fact | | |
| `src/deep_isochron/plotting/deformation/style.py` — `SheetStyle` moved, `curve_colors` forms and defaults, `_curve_colors` | | |
| `src/deep_isochron/plotting/deformation/figures.py` — `SheetStyle` imported, `draw_sheet` on `_curve_colors` | | |
| `src/deep_isochron/plotting/deformation/__init__.py` — `SheetStyle` from `.style` | | |
| `tests/test_plotting.py` — `test_curve_colors_forms`, the explicit `fill_alpha` | | |
