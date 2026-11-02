---
type: change
status: in progress (unit 1 of 3 next)
updated: 2026-11-02
---

# `ksiam-animations` — animations of a known oscillator for the KSIAM talk

## Summary

Pending: written in the documents commit. The branch adds `plotting/phase_portrait.py`,
which animates a normal form with the phase plane beside a time series: `animate_kick`
shows that a kick shifts the phase, with a delay or an advance, and
`animate_isochron_sync` shows that the points of one isochron fall into step. The design
is the Project document `claude/ksiam-animations-handoff-2026-11-02.md`.

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

## Design

The ledger of the change, in the order the commits will follow. Each item becomes a
*Files* bullet and a review row when its commit lands.

1. `plotting/phase_portrait.py`: `PortraitStyle`, the shared drawing helpers,
   `_kick_scene` and `animate_kick`; the exports; `tests/test_phase_portrait.py` with
   the Δφ oracle, the frame count and the errors; `TOL["relaxed_phase"]` — done.
2. `plotting/phase_portrait.py`: `_sync_scene` and `animate_isochron_sync`; the export;
   the seeded-phase, frame-count and error tests.
3. `ksiam_animations.ipynb`: the A1 and A2 cells, the GIF and MP4 exports, and the
   static strips.

The decisions and their rejected alternatives are written in the documents commit.

## Bugs fixed

None.

## Tests

Pending: the fast tier of `tests/test_phase_portrait.py` runs before each commit; the
full tier runs once before the documents commit. At `63c1461` (the branch point)
`uv run just test dev tests/test_plotting.py` is `26 passed`.

## Open issues

Pending.

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
