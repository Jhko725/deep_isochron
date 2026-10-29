---
type: change
status: implemented 2026-10-29; review pending
updated: 2026-10-29
---

# `merge-conflict-check` — conflict markers refused by a hook

## Summary

The merge commit `84438de` brought conflict markers into four documents on master
(fixed by `e1c5513`), because I amended and force-pushed a commit that Joon had already
merged, and no check looks for markers. This branch adds `prek`'s `check-merge-conflict`
hook, which refuses a file with a marker line, and the matching rule goes to the
`code-conventions` skill: a pushed commit is never amended.

Reversible: one line of `prek.toml`.

## Files

- `prek.toml` — `check-merge-conflict` with `--assume-in-merge` among the builtin hooks.
- `docs/coding-standards.md` — the row in *Deterministic checks*.
- `docs/roadmap.md` — Done ledger row.

## Design

- **`--assume-in-merge`.** Without it the hook only runs while a merge is in progress
  (`MERGE_HEAD` present), so `just check` and CI, which run on a finished tree, would
  never scan; with it, every run scans every file, and a marker that a merge commit
  carried in is caught on the next check rather than by a reader.
- **The repository's global `exclude` still applies**, so `isochron2pbvp/` and
  `tests/golden/` are not scanned: the first follows the coworker's conventions, and
  the second is byte-for-byte reference data whose CSVs contain no marker lines.
- **The rule beside the check.** The hook catches the symptom; the cause was rewriting
  a published commit, which no check can refuse from this side, so the rule goes to the
  skill (proposal card, 2026-10-29).

## Bugs fixed

None on this branch; `e1c5513` removed the markers.

## Tests

- `uv run prek run check-merge-conflict --all-files`: passed on the tree.
- A planted file with `<<<<<<< HEAD`, `=======` and `>>>>>>> other` lines: all three
  flagged (then removed).
- `uv run just check`: every hook passed, `ty` clean.

## Open issues

None.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `prek.toml` — `check-merge-conflict --assume-in-merge` | Confirmed. | None |
| `docs/coding-standards.md`, `docs/roadmap.md` — the check's row; the ledger | Confirmed. | None |
