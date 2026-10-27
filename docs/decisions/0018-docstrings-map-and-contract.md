---
type: decision
id: ADR-0018
status: accepted (review pending)
updated: 2026-10-27
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-26, before round 0 of the prose pass)
---

# ADR-0018 — Docstrings are the map and the contract; the reasoning lives in the ADRs; one bibliography

## Context

### Motivation

- By 2026-10-25 the tree had 780 public definitions, 45% of them with a docstring, and
  the convention was followed in a handful of modules.
- Where docstrings existed they had grown into design documents: the module docstring
  of [`analysis/ode/floquet.py`](../../src/deep_isochron/analysis/ode/floquet.py) ran
  70 lines and restated [ADR-0015](0015-floquet-by-seeded-periodic-schur.md) with its
  full citation trail,
  [`data/windows/batched.py`](../../src/deep_isochron/data/windows/batched.py) did the
  same for [ADR-0008](0008-dataset-format-and-sampling.md), and
  [`systems/symbolic.py`](../../src/deep_isochron/systems/symbolic.py) for
  [ADR-0016](0016-symbolic-systems.md). A reader opening the file got the derivation
  before the contract, and the same sentence lived in two places that could drift
  apart.
- Citations had drifted too. The same paper appeared under three forms:
`Dieci et al. 1997`, `Dieci, Russell & Van Vleck 1997`, `Dieci–Russell–Van Vleck 1997`. <!-- references:ignore: the drifted forms are the example -->
- Four design documents and one ADR each carried a reference list, and a docstring
  citing a paper had nowhere to resolve to.

### Forces

The global `CLAUDE.md` names the "equinox/diffrax convention" for docstrings without
fixing it for this repository. ruff's `D` rules parse the Google convention, which
diffrax follows. The manuscript's bibliography lives in Zotero.

## Decision

This ADR fixes the docstring convention that the global `CLAUDE.md` names for this
repository, and adds one bibliography for every citation in the tree.

### 1. The module docstring is the map, a function docstring is the contract

A module docstring says what the module provides, names the public objects and how
they relate, states the invariants that span several functions, and names the ADR or
design document that holds the reasoning (an ADR follows
[ADR-0019](0019-adrs-describe-the-current-design.md)). It is at most about 25 lines. A
function docstring carries everything the caller needs for that call, in the equinox
form (`**Arguments:**`, `**Returns:**`, `**Raises:**`), including the precision or
validity limits of that function and the citation for a formula it implements. A class
docstring lists its fields under `**Attributes:**`; fields carry no attribute
docstrings of their own. The sentence-level test is in
[`docs/coding-standards.md`](../coding-standards.md) § Docstrings.

The cost accepted is that a reader in an editor hover sees the pointer, not the
derivation.

### 2. Material cut from a docstring is moved, not dropped

A sentence that no ADR or design document holds goes to the design document as a
finding with its provenance label, or into the ADR's Decision section with a History
line (ADR-0019), in the same commit that shortens the docstring. The change document
lists each move.

### 3. One bibliography and one in-text form

[`docs/references.md`](../references.md) has one entry per work: the key, the full
reference, a DOI or arXiv identifier, and the locators the repository uses. The in-text
form is `(Key, locator)` with the key grammar of `docs/coding-standards.md`
§ References. Per-document reference lists are removed; frontmatter `sources:` lists
keys. [`scripts/check_references.py`](../../scripts/check_references.py) fails on a key
that does not resolve and lists uncited entries.

### 4. Presence is checked, content is reviewed

ruff `D100`–`D103` require a docstring on every public module, class, method and
function, switched on module group by module group as the prose pass reaches them. The
25-line standard, the map/contract split and the prose rules are reviewed, not checked;
a module that needs more lines gets them with the reason in its review row.

## Consequences

- Seventy-line module docstrings become pointers; `git log -p` on the ADR shows what
  moved.
- A new module needs a docstring before `prek` lets it through once its directory
  leaves `per-file-ignores`.
- A new citation needs an entry in `docs/references.md` in the same commit.

## Alternatives rejected

- *Decision 1*: keeping the long module docstrings and only cleaning their sentences.
  A docstring that restates an ADR is a second copy of the decision, and two copies
  drift apart. This ADR first claimed that the return-map trace remark and the quoted
  Dieci et al. claim of `floquet.py` were in the docstring and not in ADR-0015; the P2
  reading found both in the ADR (§3) and in
  [`analysis.md`](../design/analysis.md) §3, so the copies had not yet diverged, but
  nothing kept them from doing so.
- *Decision 1*: per-field attribute docstrings (the PEP 257 form that `PeriodicSchur`
  used). Pylance renders them on hover over the attribute, which is the better hover,
  but the Google convention that ruff's `D` rules parse, and diffrax, put the list in
  the class docstring; one form was wanted, and the checker sees only the list.
- *Decision 3*: a `.bib` file exported from Zotero as the source of truth. It is right
  for the manuscript, but it does not render on GitHub, its keys are not the ones that
  read well in a sentence, and the "used for" locators have no natural home in BibTeX.
  The manuscript's `.bib` stays in Zotero; an entry here may carry its Zotero key.
- *Superseded 2026-10-27*: material cut from a docstring went to the ADR as a dated
  amendment. Replaced because ADR-0019 makes an ADR's Decision section the current
  design, changed by a rewrite with a History line.

## History

- 2026-10-26 (`prose`): accepted.
- 2026-10-26 (`prose`, round 6): the Context's three drifted citation forms restored
  (round 0's own normalization had collapsed them to one key); Decision 1's rejection
  corrected to what the round-2 reading found in ADR-0015 and `analysis.md`.
- 2026-10-27 (`prose`): restructured to the format of ADR-0019; Decision 2 rewritten
  for it.
