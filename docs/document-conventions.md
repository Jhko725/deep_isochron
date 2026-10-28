---
type: reference
status: current
updated: 2026-10-29
---

# Document conventions

This document says what each kind of document under `docs/` is for, the form each one
takes, and which check enforces what. The prose rules are the `clear-prose` skill's;
the conventions for code, tests and docstrings are in
[`coding-standards.md`](coding-standards.md).

## What goes where

| Place | Holds | Read it when |
|---|---|---|
| [`architecture.md`](architecture.md) | the module map and the data flow | starting any task |
| [`roadmap.md`](roadmap.md) | the single current plan, phase by phase, and the append-only Done ledger | choosing what to do next |
| `design/` | the mathematics behind the code: the survey, the derivation, the oracles and the limits, one document per method | implementing or checking a method |
| `decisions/` | numbered ADRs: why the structure is what it is, each describing the current design with its history | touching the area an ADR covers |
| `guides/` | how to do something: install AUTO, compute isochrons, run an experiment from a notebook | doing that thing |
| `changes/` | one document per branch, the reviewer's map of the change | reviewing or continuing a branch |
| [`references.md`](references.md) | the one bibliography | citing |
| [`index.md`](index.md) | every document above, with its status | finding a document |

A guide is named for what the reader wants to do, not for the tool
(`guides/auto07p.md` installs AUTO-07p; `guides/isochrons.md` computes isochrons with
it). A design document holds a method's reasoning and a guide holds its commands; when
one document starts to carry both, it is split.

## Frontmatter and status

Every document opens with a frontmatter block: `type` (`architecture`, `roadmap`,
`design`, `decision`, `guide`, `change`, `reference`, `index`), `status`, `updated` (the
date of the last edit, changed in the same commit as the text), and where relevant
`verified_by` (who or what has checked the claims), `sources` (what the document rests
on) and `id`. The block is the useful part of Google's Open Knowledge Format, adopted
2026-10-04 so that a document's standing can be read without reading the document.

`status` values: `current` (kept true at all times), `agreed` (a design settled with
Joon), `accepted` (a decision in force; `amended` when later changed in place),
`tentative` (written by Claude, not yet checked by Joon), `draft for review`.

The block has to be valid YAML, since GitHub renders it as a table only then, and
`sources` is a block list with every item quoted:

```yaml
sources:
  - "ADR-0012 (analysis layout)"
  - "Wilson & Moehlis 2016"
```

A flow list `[ADR-0012 (analysis layout), Wilson & Moehlis 2016]` parses, but YAML splits
it at every comma, so a parenthesized remark becomes two sources; the review of
`docs-layout` found the guides' lists rendered that way. `scripts/checks/check_md_math.py`
fails on a block that does not parse, lacks `type`, `status` or `updated`, or writes
`sources` as a flow list. The change documents that predate the block carry none and are
not checked for one.

## Change documents

Every branch has a companion document `changes/<YYYY-MM-DD>-<branch>.md`, committed with
the code and updated in the same commit as any later change on the branch. It is the
reviewer's map of the change, not a changelog. Its sections:

- **Summary**: what the branch does and why.
- **Files**: every file touched, one line each, confirmable against `git diff --stat`.
- **Design**: the decisions not obvious from the code, each with the alternative that was
  rejected, linking an ADR when the decision is structural, and one line saying whether
  the change can be undone.
- **Bugs fixed**, with the symptom.
- **Tests**: what they pin down and how to run them. The section is the baseline the next
  session reads, so its counts are precise, and a test's outcome is stated only after
  that test has run on the code the document describes; until then the document says
  the run is pending.
- **Open issues**, with the reason each is open.
- **Review notes**: a table *Change* | *Thoughts* | *Modifications*, one row per file
  touched. *Change* is the file name plus a one-line summary; the other two columns are
  Joon's, filled in while vetting, and never edited by Claude except when asked. A new
  row is appended after the table's last line, never inserted by matching the text of an
  existing row, since Joon's edits change that text.

A landed change document is history: its review columns keep Joon's words verbatim,
typos included, and the spelling checks skip `changes/` for that reason.

## Decision records

An ADR describes the current design, and its history follows it
([ADR-0019](decisions/0019-adrs-describe-the-current-design.md)). An ADR earns its place
when the decision is hard to reverse, surprising without context, and came from a real
trade-off; otherwise the change document's *Design* section is enough. Its sections, in
order:

```markdown
## Context
### Motivation    one bullet per point, each with its evidence, or *[not recorded]*
### Forces        prose: what any solution had to respect
## Decision       the current design
## Consequences
## Alternatives rejected   superseded designs too, marked *Superseded* with date and reason
## History        one dated line per change, with its branch
```

A change to a decision rewrites Decision and Consequences in the same commit; a change
that reverses the decision is a new ADR. The first mention of an ADR, design document or
code file in each section is a relative link, and `scripts/checks/check_references.py`
fails on a link whose target does not exist.

## Design documents

A design document follows the `numerical-method` skill: the survey with the routes and
where each fails, the derivation in one convention stated once, one subheading per stage
of the algorithm in the order the code runs them, the oracles with the tolerance each
justifies, and the limits. Every claim carries a provenance label: **[cited]** from a
named source, **[code]** verified in the tree at the moment of writing, **[deduced]**
Claude's inference. The labels stay when the text is edited, and a `[code]` claim is
re-read rather than recalled: ADR-0004's *Consequences* once described a `# ty: ignore`
three branches after it was removed.

## Paragraphs

A paragraph carries one theme, and a short paragraph is fine. Review of ADR-0017 found
"unrelated ideas grouped into a conjoined paragraph", and the fix was line breaks, not
more words. The `clear-prose` skill has the rules; the skim test (read only the first and
last sentence of every paragraph) is the check.

## References

The repository has one bibliography, `references.md`, and one in-text form. The key is
`Author Year` for one author (`Lust 2001`), `Author & Author Year` for two
(`Wilson & Moehlis 2016`), `Author et al. Year` for three or more (`Dieci et al. 1997`),
with a letter suffix on a collision (`Langfield et al. 2025a`). <!-- references:ignore: the example --> A locator follows the key after a comma, in one of
the forms `Eq. 7`, `Eqs. 9–14`, `§3`, `Prop. 5`, `Theorem 2`, `Table 5`, `Fig. 9`,
`pp. 210–214`, `Ch. 8`, `App. D`, `Remark 4`, and several citations are separated by
semicolons inside one pair of parentheses. Every key resolves to an entry in
`references.md`, which `scripts/checks/check_references.py` enforces, and the entry
carries the full reference, a DOI or arXiv identifier, and the locators the repository
uses. Design documents and ADRs carry no reference list of their own; their frontmatter
`sources:` lists keys. `README.md` is in the check's scope too: its acknowledgment once
named a paper without a citation.

## Markdown that renders in both previews

GitHub runs Markdown before MathJax, and VS Code's preview parses `$…$` first. These are
the rules `scripts/checks/check_md_math.py` enforces, with the reason each exists:

- There is no line-length limit, since ruff formats Python only, but an inline formula
  `$…$` never crosses a line.
- Display math only as a standalone block: `$$` alone on a line, the formula, `$$` alone
  on a line. Never `$$…$$` inside a paragraph.
- Inside inline math, no backslash followed by ASCII punctuation: `\,` `\;` `\{` `\|`
  `\\` become `\thinspace` `\medspace` `\lbrace` `\Vert`, and matrices move into blocks.
- No `*` inside inline math: use `\ast`, since `x^*` loses its star and italicizes the
  rest of the paragraph.
- Spaces around `<` and `>`.
- No `_` immediately after `}` `)` `|`: write `\mathbf e_r`, `C_{\rm loc}^{k}`,
  `\vert_{E}`, because such an `_` opens Markdown emphasis.

Every span has also been checked against KaTeX as VS Code configures it, so the two
previews agree.

## Checks

| Rule | Check | Runs in |
|---|---|---|
| both Markdown previews; the frontmatter parses, carries `type`, `status`, `updated`, and lists `sources` in block form | `scripts/checks/check_md_math.py` | `prek`, CI |
| every citation resolves to `references.md`; every relative Markdown link resolves | `scripts/checks/check_references.py` | `prek`, CI |
| American English | `codespell` and `scripts/checks/check_spelling.py`, described in [`coding-standards.md`](coding-standards.md) § Deterministic checks | `prek`, CI |
| `index.md` lists every document; `architecture.md` follows every module move; the Done ledger is append-only | review | — |
