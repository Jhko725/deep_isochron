# 2026-10-25 · `conventions-docs`

Base commit: `symbolic-systems` tip (the latest work in the repository on this date).
Branch commit(s): see `git log master..conventions-docs`.

## Summary

Splits this repository's conventions across the four layers they belong to, so that each
one is stated once. The universal rules — jit and vmap, docstrings, parameter leaves,
American English, annotations — move to Joon's global `CLAUDE.md` and to a
`code-conventions` skill that loads on demand; this repository's instantiation of them —
tiers, named tolerances, the contracts the ADRs fix, the Markdown rules, and the checks
that replace prose — moves to a new `docs/coding-standards.md`; and `CLAUDE.md` keeps
only the session protocol, the change-document contract and the facts Claude cannot
infer from the code. `CLAUDE.md` loses about a third of its length, and nothing is lost:
every rule removed from it has a new home that the file points at. No code changes.

## Files

| File | Change |
|---|---|
| `CLAUDE.md` | pruned; gains a *Start of a session* and an *End of a session* section and a pointer to the new standards document |
| `docs/coding-standards.md` | **new** — tiers and commands, named tolerances and solvers, the contracts with their ADR numbers, the Markdown rules, the checks, deliberate exceptions |
| `docs/index.md` | the new document listed under *Read first*; the branch's change document listed; `updated` bumped |
| `scripts/check_spelling.py` | **new** — the rule for the `-ise` / `-isation` / `-yse` families, with the closed exception list |
| `docs/changes/2026-10-25-conventions-docs.md` | this document |

## Design

- **Four layers, one home per rule.** A rule that applies to every edit in every
  repository is loaded every session, so it belongs in the global `CLAUDE.md` and has to
  earn its place there; a rule that is mechanically checkable belongs in a check, because
  prose is advisory and a check is not; a rule that needs its reasoning to be arguable
  belongs in a skill, loaded when design work starts; and a fact particular to this
  repository belongs in this repository's `CLAUDE.md`. *Rejected*: keeping everything in
  `CLAUDE.md`, which is the state this branch leaves — at 5.5 kB it was approaching the
  length at which a file stops being followed, and it duplicated most of
  `context_flux_no/CLAUDE.md` line for line.
- **What moved out of `CLAUDE.md`, and where.** The jit/vmap rule, the docstring
  convention, the `raw`/`constrain` parameter rule and American English went to the
  global file (they hold in both repositories and in the next one). The `AbstractVar`
  field form, the registry requirement and the thirteen lines of Markdown escape rules
  went to `docs/coding-standards.md`, each beside the check or the ADR that enforces it.
- **What was added.** A *Start of a session* and an *End of a session* section, matching
  `context_flux_no`'s: the reading order and the end-of-session checklist were previously
  implicit here, scattered through the *Repository facts* bullets. The provenance labels
  ([cited] / [code] / [deduced]) are also stated explicitly for the first time, although
  `docs/design/analysis.md` has used them since Phase E began.
- **The spelling rule becomes two checks, because the rule has two halves.** `codespell`
  is not a dictionary: its `en-GB_to_en-US` data is a 535-entry word list, and measured
  against this repository's vocabulary it misses `discretise`, `parametrise`, <!-- codespell:ignore -->
  `orthogonalise`, `diagonalise`, `linearise` and `denormalise`; `breame`, the other <!-- codespell:ignore -->
  candidate, misses the same six [code: 2026-10-25]. A list cannot close over productive
  morphology. So `codespell` keeps the half a list is good at — the irregulars and the
  typos, which is how `centerd`, `implemention` and `constract` were found — and <!-- codespell:ignore -->
  `scripts/check_spelling.py` takes the half a rule is good at: `-ise` → `-ize` with an
  exception list of the words where `-ise` is also correct in American English, which is
  finite and enumerable where the inclusion list is not. `-isation` and `-isable` need no
  exceptions; `-yses` is excluded from the rule because it is also the plural of `-ysis`.
  *Rejected*: a hunspell differential (`en_GB` minus `en_US`), which is genuinely
  morphological but flags every domain term — `Floquet`, `isostable`, `jaxtyping` — and
  would need a personal word list larger than the exception list it replaces.

## Bugs fixed

None; no code changed.

## Tests

None added. `uv run python scripts/check_md_math.py` must report 0 problems, since three
Markdown files changed, and `uv run python scripts/check_spelling.py` is itself new. The suite is untouched, so the baseline recorded in the branch
documents still holds.

## Open issues

- The checks in `docs/coding-standards.md` are written down but not wired up. Turning the
  two spelling checks on needs one `style:` commit first: `codespell` reports 70 findings
  and `check_spelling.py` 23, each with one in `src` (`model/autoencoder.py`).
- `scripts/check_spelling.py` has no test of its own. Its behaviour is pinned only by the <!-- codespell:ignore -->
  measurement in the Design section: 13 of 13 on a line of British forms, 0 of 37 on a
  line of words where `-ise` is correct.
- The `code-conventions`, `diagnose`, `numerical-method`, `grilling` and `retro` skills
  live in Joon's account rather than in this repository, so a collaborator cloning it
  gets the standards document but not the reasoning behind it. Committing copies under
  `.claude/skills/` is the alternative if that matters.
- Found while adding a row: `docs/index.md`'s change-document table is missing the
  `ci-ty` and `symbolic-systems` branches. Not fixed here — it belongs to whoever lands
  those — but this repository has no equivalent of `context_flux_no`'s
  `scripts/check_docs.py`, which is what keeps that table honest there.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `CLAUDE.md`: pruned to protocol, change-document contract and repository facts; session and end-of-session sections added; pointer to the standards document | | |
| `docs/coding-standards.md` (new): tiers, named values, contracts with ADR numbers, Markdown rules, checks, exceptions | | |
| `docs/index.md`: the new document under *Read first*, and the branch's change document | | |
| `scripts/check_spelling.py` (new): the `-ise` / `-isation` / `-yse` rule with its exception list; 23 findings here, none of them false | | |
| `docs/changes/2026-10-25-conventions-docs.md`: this document | | |
