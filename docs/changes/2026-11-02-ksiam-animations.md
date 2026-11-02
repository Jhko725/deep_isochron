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

## Design

The ledger of the change, in the order the commits will follow. Each item becomes a
*Files* bullet and a review row when its commit lands.

1. `plotting/phase_portrait.py`: `PortraitStyle`, the shared drawing helpers,
   `_kick_scene` and `animate_kick`; the exports; `tests/test_phase_portrait.py` with
   the Δφ oracle, the frame count and the errors; `TOL["relaxed_phase"]`.
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
