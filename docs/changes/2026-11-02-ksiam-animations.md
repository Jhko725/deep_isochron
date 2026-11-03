---
type: change
status: landed on the branch; review pending
updated: 2026-11-03
---

# `ksiam-animations` — animations of a known oscillator for the KSIAM talk

## Summary

The branch adds `plotting/phase_portrait/`, two animations of a known planar oscillator
for slides 3 and 4 of the KSIAM 2026 talk. Each shows the phase plane beside the
horizontal coordinate against time, and draws only what the normal form computes in
closed form. `animate_kick` shows that a kick shifts the timing of an oscillator for
good: one point on the limit cycle and its kicked copies, each copy's asymptotic phase
shift labeled in the final hold. `animate_isochron_sync` shows what an isochron is:
points seeded on one isochron ride it onto the cycle and fall onto one waveform. The
notebook `ksiam_animations.ipynb` writes the talk's GIFs, MP4s and static strips. The
design and its 57 decisions are the Project document
`claude/ksiam-animations-handoff-2026-11-02.md`; the ones that are not obvious from the
code are below.

## Files

- `src/deep_isochron/plotting/phase_portrait.py` (**new**) — `PortraitStyle`, the
  shared drawing helpers, `_KickScene` and `_kick_scene`, and `animate_kick`, which
  animates a point on the cycle and its kicked copies and labels each copy's Δφ.
- `src/deep_isochron/plotting/__init__.py` — exports `PortraitStyle` and
  `animate_kick`; the module docstring names the animations.
- `tests/test_phase_portrait.py` (**new**) — the Δφ oracle against the polar angles
  twelve periods after the kick, the GIF's frames counted by duration, and the four
  errors of `animate_kick`.
- `tests/helpers.py` — the `TOL` entry `relaxed_phase`.
- `src/deep_isochron/plotting/phase_portrait.py` — after the review of the first A1
  render: 40 frames per period, a kick pause of `kick_pause_seconds` with both kick
  arrows, a 3 s final hold, Δφ labels at `annotation_font_size` with thicker arrows
  and more space between them, and traces and tails drawn from `_SAMPLES_PER_FRAME`
  samples per frame; `_frame_sequence` takes the paused frames and their seconds.
- `tests/test_phase_portrait.py` — the GIF's summed frame durations include the kick
  pause and the longer hold.
- `src/deep_isochron/plotting/phase_portrait.py` — after the second review of A1: the
  figures draw under the style that the caller activates, the phase plane hides its
  tick marks and tick labels but keeps its tick positions for the grid, the grid lies
  below every drawn artist, both panels share their top and bottom edges with a square
  plane placed in inches from `_MARGINS` and `_PANEL_GAP`, `font_size` 18, `box` 1.8,
  and the docstring's save example passes `dpi=style.dpi`.
- `tests/test_phase_portrait.py` — the frame count runs under `configure_rcparams` and
  saves at `dpi=style.dpi`; `test_panels_share_top_and_bottom` checks the panel edges.
- `src/deep_isochron/plotting/phase_portrait.py` (**deleted**) — the module becomes a
  subpackage, because A2 would take it past the 600-line limit; nothing changes in
  behavior.
- `src/deep_isochron/plotting/phase_portrait/__init__.py` (**new**) — the subpackage
  docstring and the exports of `PortraitStyle` and `animate_kick`.
- `src/deep_isochron/plotting/phase_portrait/style.py` (**new**) — `PortraitStyle`,
  moved unchanged.
- `src/deep_isochron/plotting/phase_portrait/drawing.py` (**new**) — the shared
  private helpers and constants, moved unchanged.
- `src/deep_isochron/plotting/phase_portrait/kick.py` (**new**) — `MINUS`,
  `_KickScene`, `_kick_scene`, `_shift_label` and `animate_kick`, moved unchanged.
- `tests/test_phase_portrait.py` — the import of `_kick_scene` comes from
  `phase_portrait.kick`.
- `src/deep_isochron/plotting/phase_portrait/sync.py` (**new**) — `_SyncScene`,
  `_sync_scene` and `animate_isochron_sync`, which animates points seeded on chosen
  isochrons, redraws each group's isochron at the phase `(θ₀ + t/T) mod 1` at every
  frame, pauses on the start and labels each group's phase in the final hold.
- `src/deep_isochron/plotting/phase_portrait/style.py` — `kick_pause_seconds` becomes
  `pause_seconds`, the pause at the key instant of either animation; `group_colors`
  becomes blue and reddish purple.
- `src/deep_isochron/plotting/phase_portrait/kick.py` — `animate_kick` reads
  `pause_seconds`.
- `src/deep_isochron/plotting/phase_portrait/__init__.py` — exports
  `animate_isochron_sync`; the docstring names `sync.py`.
- `src/deep_isochron/plotting/__init__.py` — exports `animate_isochron_sync`.
- `tests/test_phase_portrait.py` — the seeded phases, the points on their moving
  isochron at every frame, the GIF's frames counted by duration, and the four errors
  of `animate_isochron_sync`.
- `src/deep_isochron/plotting/phase_portrait/drawing.py` — after the review of the A2
  variants: the clock sits in the white box `_TEXT_BOX`, which A1 shares, and
  `_draw_cycle` returns its line.
- `src/deep_isochron/plotting/phase_portrait/sync.py` — the φ labels sit in the same
  white box, at the position that `_place_labels` searches on the final frame: every
  45° around the landing point at offsets up to 1.2, inside the plane, off the clock and
  the other label, with the largest clearance from the cycle, the highlighted isochrons
  and the points, and at least `_MIN_LABEL_CLEARANCE_PX` when such a position exists.
- `tests/test_phase_portrait.py` — `test_sync_labels_clear_the_foreground` checks each
  label's box against the plane, the clock, the other label and the foreground curves.
- `ksiam_animations.ipynb` (**new**) — the notebook that writes `results/ksiam/`: the A1
  and A2 animations as GIF and MP4, and the strips `a1_strip.png` and `a2_strip.png`,
  whose frames are read from the saved GIFs by cumulative display time. The outputs are
  not committed.
- `docs/changes/2026-11-02-ksiam-animations.md` — the summary, the design decisions
  with their rejected alternatives, the tests and the open issues.
- `docs/architecture.md` — the `phase_portrait/` subtree under `plotting/`.
- `docs/roadmap.md` — the branch appended to the *Done* ledger.
- `src/deep_isochron/plotting/phase_portrait/labels.py` (**new**) — `_place_labels`,
  `_box_extent`, `_foreground_samples`, `_anchor` and their four constants, moved
  unchanged from `sync.py` so that a further animation can place labels with the same
  search.
- `src/deep_isochron/plotting/phase_portrait/sync.py` — the label search is removed and
  imported from `labels.py`.
- `src/deep_isochron/plotting/phase_portrait/__init__.py` — the docstring names
  `labels.py`.
- `tests/test_phase_portrait.py` — `_MIN_LABEL_CLEARANCE_PX` is imported from
  `phase_portrait.labels`.


## Design

The ledger of the change, in the order the commits followed.

1. `plotting/phase_portrait.py`: `PortraitStyle`, the shared drawing helpers,
   `_kick_scene` and `animate_kick`; the exports; `tests/test_phase_portrait.py` with
   the Δφ oracle, the frame count and the errors; `TOL["relaxed_phase"]` — done
   (`400bbfa`, revised after Joon's reviews in `7acdf2b` and `f477165`).
2. `plotting/phase_portrait/`: the module split into a subpackage (`db9fdcd`), then
   `_sync_scene` and `animate_isochron_sync` with their tests (`b82e25d`), then the
   boxed clock and the placed φ labels (`ba61047`) — done.
3. `ksiam_animations.ipynb`: the A1 and A2 cells, the GIF and MP4 exports, and the
   static strips — done (`ece9e3b`, import order fixed in `80303b6`).

The decisions, each with the alternative it rejected:

- **The functions take the system, not arrays.** `animate_kick` and
  `animate_isochron_sync` take an `AbstractNormalForm` and get every drawn object from
  it: `flow` in closed form, `phase`, `isochron`, `limit_cycle`, `period` and
  `in_basin`. They explain a known ODE, so the frames are exact and nothing is
  recomputed in the plotting layer. The `deformation/` figures take arrays instead,
  because they show measured data meeting a learned map. Rejected: array inputs by
  analogy with `deformation/`.
- **The parameters are fixed by a contact sheet.** The scenes use
  `BautinNormalForm(a=1/3, b=0.5, w=2π, w0=2π − 2/3)`: T = 1, κ = −1, μ = e^{κT} = 0.37,
  and the isochrons spiral visibly. The repository's `configs/data/bautin.yaml` has
  κT = −30, so a relaxation there is over within a few percent of a period. Rejected:
  b = 0, which spirals more outside the cycle but is a Hopf form with a twist.
- **A1 kicks one point in two directions at one instant.** The point and its two copies
  move together for one period; the copies then jump by `[[0.75, 0], [-0.75, 0]]` at
  phase 0.83, and Δφ = ±0.146. Each copy is colored by the sign of its Δφ (vermilion
  for a delay, green for an advance). Rejected: a radial kick, whose Δφ is the same at
  every phase by rotational symmetry; separate loops kicked at different phases; a
  kick of length 0.5, whose shifts reach only ±0.108.
- **A2's highlighted isochrons turn with the cycle.** Each group's isochron is redrawn
  every frame at the phase (θ₀ + t/T) mod 1, so its points stay on it while they slide
  onto the cycle; the ten faint isochrons at phases k/10 stay fixed. A point leaves a
  fixed isochron at once in the lab frame, because the foliation is invariant but each
  leaf turns. Rejected: a fixed curve, and the co-rotating frame of the first plan,
  which the time panel made redundant.
- **A2 is drawn without tails.** The notebook passes
  `copy.replace(style, tail_periods=0)`; A1 keeps its half-period tails. Isochron and
  tail in one color and one line style were hard to tell apart, and the time panel
  already shows the history. Rejected after a rendered comparison: lighter tails, and
  long-dash isochrons.
- **Timing gives each key instant time to register.** One period takes 2 s (40 frames
  at 20 fps), the animation pauses for `pause_seconds = 1.5` at A1's kick and at A2's
  start, and the final frame holds for 3 s. A1 lasts 12.55 s and A2 14.55 s. Rejected:
  1 s per period, which Joon found too fast to follow.
- **The figures draw under the caller's style.** The notebook calls
  `configure_rcparams()`, as the other notebooks do; the library never calls or copies
  it. Every save passes `dpi=style.dpi`, because `configure_rcparams` sets
  `savefig.dpi = 600`, which `FuncAnimation.save` would otherwise use.
- **The panels are placed in inches.** `_portrait_figure` places a square plane and the
  time axis from `style.figsize`, `_MARGINS` and `_PANEL_GAP`, so both panels share
  their top and bottom edges at any figure size. The grid lies below every artist.
  Rejected: constrained layout with a width ratio, which left the plane shorter than
  the time panel.
- **Text stays legible over the curves.** The clock and A2's φ labels sit in a white
  box (alpha 0.85). `_place_labels` tries every 45° around each landing point at
  offsets up to 1.2 and takes the position with the largest clearance from the cycle,
  the highlighted isochrons and the points, at least `_MIN_LABEL_CLEARANCE_PX = 8`
  when one exists; otherwise it takes the best position and draws the label on top.
  The faint isochrons are ignored, since they cover the whole plane. Rejected:
  preferring the outer side of the landing point, where a 205 px label does not fit in
  the 155 px to the panel edge; raising an error when no position clears 8 px.
- **Traces and tails are sampled 10 times finer than the frames**
  (`_SAMPLES_PER_FRAME`), so the waveforms show no corners at slide size.
- **The module became a subpackage.** `style.py`, `drawing.py`, `kick.py` and `sync.py`,
  as `deformation/` is split, because A2 would have taken one module to about 760
  lines, past the 600-line limit of `scripts/checks/check_docstrings.py`. The split
  landed as a pure move before A2.
- **The tests pin the numbers on screen.** The Δφ that A1 labels matches the angle
  difference of the trajectories twelve periods after the kick, to
  `TOL["relaxed_phase"]`, an oracle independent of the closed-form phase. Every A2 point
  has its drawn isochron's phase at every frame. The frame counts are summed GIF
  durations, because Pillow merges identical frames.

None of these meets the ADR test: the functions are leaves that nothing in the package
imports, and every decision is undone by editing the subpackage and one notebook.

## Bugs fixed

None.

## Tests

- `uv run just check`: clean on every commit except two that an implementer reported
  clean: `9f3b305` (ruff E501 in `sync.py`, amended to `b82e25d` before the push) and
  `ece9e3b` (the notebook's import order, fixed by `80303b6`). Clean at `80303b6`.
- The full tier, `uv run just test default` at `80303b6` (two cores, 7 GB, JAX's
  persistent compilation cache on): `699 passed, 10 skipped, 1 xfailed in 1307 s`. The
  skips are the `auto` tests without an AUTO-07p tree; the `xfail` is the pinned `sinh`
  second-derivative case of `tests/test_analytic.py`, unrelated. The branch point
  `63c1461` was recorded at `684 passed, 10 skipped, 1 xfailed`
  (`2026-10-30-deformation-plots.md`); the fifteen new passes are this branch's. The
  three warnings are beartype's PEP 585 deprecation on `experiment.compose`, unrelated.
- `tests/test_phase_portrait.py` at the `dev` profile: `15 passed`. Per commit: `6`
  (`400bbfa`, `7acdf2b`), `7` (`f477165`, `db9fdcd`), `14` (`b82e25d`), `15`
  (`ba61047`). `tests/test_plotting.py` stays at `26 passed`.
- What the tests pin: A1's Δφ against the angle difference twelve periods after the
  kick (`TOL["relaxed_phase"]`); the seeded A2 points on their isochrons, and every A2
  point on its moving isochron at every frame (`TOL["closed_form"]`); the summed GIF
  durations of both animations, pauses and hold included, under the project style; the
  shared top and bottom edges of the two panels; the φ labels inside the plane, off the
  clock and each other, and at least 8 px from the foreground curves; and the four
  `ValueError` cases of each function. The oracle, the moving-isochron, panel and label
  tests were each seen to fail on a planted mutation.
- Not covered by a test: the rendered pictures. Every revision was rendered on the
  default scenes and reviewed by Joon (A1 five times, A2 three times, plus a contact
  sheet of three styling variants). The notebook's MP4s were probed: `h264`, `yuv420p`,
  1920×800, 20 fps, 12.55 s (A1) and 14.55 s (A2).

## Open issues

- `BautinNormalForm` does not override `basin_radii`, so for −1 < b < 0 `in_basin`
  accepts points beyond the outer repelling cycle that its docstring names
  (`systems/normal_forms/bautin.py:8-10`; `base.py:190-200`). Found by unit 1 while
  testing the basin error, which therefore uses Winfree's model. It belongs to the
  systems layer, not to this branch.
- The bottom margin is fixed at 0.5 in (`_MARGINS`). It fits 18 pt tick labels; a much
  larger `font_size` would clip them.
- The notebook closes each figure through `FuncAnimation`'s private `_fig`, since the
  functions return only the animation.
- The A2 GIF is 19.8 MB and the A1 GIF 6.2 MB; the MP4s are about 1 MB each, so the
  slides should embed the MP4s where the deck allows.
- The label search runs on the final frame at the style's dpi and figure size; it is
  tested only on the default scene.
- Process, for the retro: the `implementer` agent type was not registered when unit 1
  started, so unit 1 ran as a general-purpose agent that read
  `.claude/agents/implementer.md` first. Twice an implementer reported `just check`
  clean while the committed tree failed ruff: `9f3b305` was amended to `b82e25d`, and
  `ece9e3b` was fixed by `80303b6`.

## Review notes


| Change | Thoughts | Modifications |
|---|---|---|
| `src/deep_isochron/plotting/phase_portrait.py` — `PortraitStyle`, the shared helpers, `_kick_scene`, `animate_kick` | | |
| `src/deep_isochron/plotting/__init__.py` — the exports of `PortraitStyle` and `animate_kick` | | |
| `tests/test_phase_portrait.py` — the Δφ oracle, the frame count, the errors | | |
| `tests/helpers.py` — `TOL["relaxed_phase"]` | | |
| `src/deep_isochron/plotting/phase_portrait.py` — slower A1, the kick pause, the longer hold, larger Δφ labels, finer traces | | |
| `tests/test_phase_portrait.py` — the frame durations with the kick pause and the longer hold | | |
| `src/deep_isochron/plotting/phase_portrait.py` — A1 under the project style, equal panel heights, larger text, zoomed plane | | |
| `tests/test_phase_portrait.py` — the frame count under the project style, the panel edges | | |
| `src/deep_isochron/plotting/phase_portrait.py` — deleted; its content moves to the subpackage below | | |
| `src/deep_isochron/plotting/phase_portrait/__init__.py` — the subpackage docstring and the exports of `PortraitStyle` and `animate_kick` | | |
| `src/deep_isochron/plotting/phase_portrait/style.py` — `PortraitStyle`, moved unchanged | | |
| `src/deep_isochron/plotting/phase_portrait/drawing.py` — the shared private helpers and constants, moved unchanged | | |
| `src/deep_isochron/plotting/phase_portrait/kick.py` — `MINUS`, `_KickScene`, `_kick_scene`, `_shift_label`, `animate_kick`, moved unchanged | | |
| `tests/test_phase_portrait.py` — the import of `_kick_scene` from `phase_portrait.kick` | | |
| `src/deep_isochron/plotting/phase_portrait/sync.py` — `_SyncScene`, `_sync_scene`, `animate_isochron_sync` with moving isochrons | | |
| `src/deep_isochron/plotting/phase_portrait/style.py` — `pause_seconds` for `kick_pause_seconds`; new `group_colors` | | |
| `src/deep_isochron/plotting/phase_portrait/kick.py` — reads `pause_seconds` | | |
| `src/deep_isochron/plotting/phase_portrait/__init__.py` — the export of `animate_isochron_sync` | | |
| `src/deep_isochron/plotting/__init__.py` — the export of `animate_isochron_sync` | | |
| `tests/test_phase_portrait.py` — the seeded phases, the moving isochron, the frame count, the errors of `animate_isochron_sync` | | |
| `src/deep_isochron/plotting/phase_portrait/drawing.py` — the clock in a white box; `_draw_cycle` returns its line | | |
| `src/deep_isochron/plotting/phase_portrait/sync.py` — φ labels in a white box, placed by `_place_labels` clear of the foreground | | |
| `tests/test_phase_portrait.py` — `test_sync_labels_clear_the_foreground` | | |
| `ksiam_animations.ipynb` — the A1 and A2 cells, the GIF and MP4 exports, the strips | | |
| `docs/changes/2026-11-02-ksiam-animations.md` — summary, design, tests, open issues | | |
| `docs/architecture.md` — the `phase_portrait/` subtree | | |
| `docs/roadmap.md` — the *Done* row | | |
| `src/deep_isochron/plotting/phase_portrait/labels.py` — the label search, moved from `sync.py` | | |
| `src/deep_isochron/plotting/phase_portrait/sync.py` — imports the label search from `labels.py` | | |
| `src/deep_isochron/plotting/phase_portrait/__init__.py` — the docstring names `labels.py` | | |
| `tests/test_phase_portrait.py` — `_MIN_LABEL_CLEARANCE_PX` imported from `phase_portrait.labels` | | |
