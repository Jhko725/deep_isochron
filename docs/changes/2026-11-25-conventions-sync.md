---
type: change
status: in-progress
updated: 2026-11-25
base: inn-schematics
sources:
  - "../plans/2026-11-25-conventions-sync.md"
---

# `conventions-sync`

## Summary

The branch brings `deep_isochron` up to the conventions of `UniversalForceReconstruction` (UFR) at its commit `b9e4cc3`. It ports review rounds in change documents, rule files and hooks under `.claude/`, the document checks, committed plans, the complexity and dependency checks, and a bibliography generated from `references.bib`. The plan of the round is [`../plans/2026-11-25-conventions-sync.md`](../plans/2026-11-25-conventions-sync.md).

The round is under way. This section is completed when the last unit lands.

## Design

Joon confirmed the nine points below on 2026-11-25, in the planning chat. The plan labels them A1 to A9, and the labels are kept here because the commits cite them.

- **A1. An old status sentence is kept.** Each change document whose status is normalized keeps its old status sentence as its last line, in the form `Status before 2026-11-25: <sentence>`.
- **A2. Three ADRs lose "review pending".** ADR-0016, ADR-0017 and ADR-0020 become `accepted`, and their three pending reviews are listed under *Open issues*.
- **A3. Each design document has a kind.** `method` for `cycle-estimation.md`, `floquet.md`, `isochrons.md`, `limit-cycle.md` and `phase-field.md`. `derivation` for `normal-forms.md` and `cubic-bspline.md`. `note` for `training-step-performance.md` and `validation-split.md`.
- **A4. `normal-forms.md` becomes `agreed`.** One line under its title names the sections that are still tentative, since the status vocabulary has no value for a document that is partly agreed.
- **A5. One ledger row moves.** The row of `docs/roadmap.md` dated 2026-10-27 that follows two rows dated 2026-10-28 moves above them. It is the one reordering the ledger ever gets.
- **A6. The architecture rule reads the fenced module tree.** A directory named inside the fenced tree of `docs/architecture.md` counts as listed, since that tree is this repository's module map.
- **A7. The glossary's fourth column is *Unit or range*.** This repository has no named-dimension table.
- **A8. Every undeclared import is declared.** None is removed from the code.
- **A9. `verification.md` is seeded from the method documents.** Its first rows come from § 5 of the five `method` design documents. A claim outside them gets its row when its code is next touched.

The remaining decisions of the plan (§ 10), each with its rejected alternative and how it is undone, are written here when the last unit lands.

## Bugs fixed

None so far.

## Tests

No test file has changed so far. `uv run just check` passes at each commit. The full quick tier runs once, after the last code commit, and its counts are recorded here then.

## Open issues

- The round is in progress. The plan's table of units (§ 3) lists what is left.

## Review notes

### Round 1: from `8ef9113`

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/plans/2026-11-25-conventions-sync.md` — the handoff of this round, committed verbatim under a frontmatter block. | | |
| `scripts/checks/markdown.py` — UFR's Markdown reader, copied with a source line. | | |
| `scripts/checks/check_changes.py` — UFR's rounds check in place of the rows check. A document's `base` field names the branch that round 1 starts from (`base_ref`), and the 24 old-format documents are skipped by name (`OLD_FORMAT`). | | |
| `prek.toml` — the `check-changes` hook always runs. | | |
| `pyproject.toml` — pytest and ty find `scripts/checks`, and `pyyaml` joins the `dev` group. | | |
| `uv.lock` — `pyyaml` in the `dev` group. | | |
| `.github/workflows/ci.yml` — the `checks` job fetches the full history. | | |
| `docs/changes/2026-11-25-conventions-sync.md` — this document. | | |
| `docs/document-conventions.md` — § Change documents in the rounds form, with the `base` field and the old-format rule, and a new § Plans. | | |
| `.claude/agents/implementer.md` — the implementer fills rows of the open round, in place of *Files* bullets. | | |
| `docs/index.md` — the plan and this document listed. | | |
