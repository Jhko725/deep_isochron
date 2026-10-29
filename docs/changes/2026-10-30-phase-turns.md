---
type: change
status: ADR written 2026-10-30; the change pending
updated: 2026-10-30
---

# `phase-turns` — the phase as a fraction of the period in $[0, 1)$

## Summary

[ADR-0020](../decisions/0020-phase-in-turns.md) moves the phase from an angle in
$(-\pi, \pi]$ to a fraction of the period in $[0, 1)$, increasing with time on every
system, with angles kept only where the quantity is one. The ADR was written first and
is the only change so far; the code follows in dependency order, each layer in its own
commit, so that no commit leaves master with two conventions.

Irreversible in effect: the isochron hashes are repinned and every saved foliation is
recomputed (ADR-0020, Consequences).

## Files

- `docs/decisions/0020-phase-in-turns.md` (**new**) — the decision, with Joon's
  rationale (the angle reading misleads on a cycle that is not star-shaped), the
  coworker's existing fractions, and the orientation rule.
- `docs/index.md` — the ADR's row; this document's row.

## Design

The ledger of the change, in the order the commits will follow. Each item becomes a
*Files* bullet and a review row when its commit lands.

1. `utils/numerics.py`: `wrap_phase` to $[0, 1)$, `wrap_angle` for radians,
   `phase_difference` in $(-\tfrac12, \tfrac12]$, all against the array API.
2. `systems/normal_forms/base.py`: `phase`, `to_phase_amplitude`, `limit_cycle`,
   `isochron` at the seam; the polar angle negated for $\omega_1 < 0$.
3. `model/base.py`, `conjugacy.py`, `autoencoder.py`: the contract in turns;
   `cycle_point` and `isochron` take $2\pi\Theta$; `phase_sensitivity` in turns.
4. `analysis/cycle.py`, `analysis/data/cycle.py`: `Cycle`'s phase in turns, the
   Fourier series in $2\pi\varphi$ inside.
5. `analysis/ode/phase.py`: `phase = wrap(t/T)`, the discrepancy by
   `phase_difference`.
6. `analysis/model.py`: `phase_offset` in turns; `phase_sense` by `phase_difference`.
7. `analysis/ode/auto/`: `rotate_phase`'s $\tau_2 = (1 - \Theta)T$; `wrap_phases`;
   `Isochron.phase`; the netCDF `units` attribute and the refusal in `load`.
8. `experiment/isochrons.py`, `configs/isochrons/`: `phases.count` → $k/$`count`;
   the hashes repinned.
9. `training/evaluation.py`: `circular_std` and the sums with $e^{2\pi i\Delta}$;
   `align_phases` in turns.
10. `plotting/isochrons.py`: `PHASE_NORM = Normalize(0, 1)`, the sample grids, the
    gaps by `phase_difference`.
11. `tests/`: every phase assertion and the golden isochrons' phase labels
    ($k/100$); the three `_gap` helpers replaced by `phase_difference`.
12. `prototype_analysis.ipynb`; the design documents and ADR History lines named in
    ADR-0020; `architecture.md`; the roadmap's item M.

## Bugs fixed

None yet.

## Tests

- `uv run just check`: every hook passed on the ADR commit.
- The code's tests: pending.

## Open issues

- The foliation files on the cluster are recomputed after the merge.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/decisions/0020-phase-in-turns.md`, `docs/index.md` — ADR-0020 and its rows | | |
