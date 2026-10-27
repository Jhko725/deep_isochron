---
type: decision
id: ADR-0019
status: accepted
updated: 2026-10-27
verified_by: joon
---

# ADR-0019 — An ADR describes the current design; its history follows it

## Context

### Motivation

- A reader learned the current design only after reading the superseded one.
  [ADR-0011](0011-experiment-layer.md) describes the run layout in Decision 6 and
  replaces it in the amendment that follows ("Amended 2026-10-07");
  [ADR-0005](0005-constraint-primitives.md),
  [ADR-0007](0007-systems-hierarchy-and-flow-strategies.md),
  [ADR-0012](0012-analysis-layout.md) and [ADR-0014](0014-limit-cycle-by-shooting.md)
  carry amendments that also contradict part of their main text.
  Joon raised this in the review of round 6: "one invests in reading the main text,
  only to find in the amendment section that the design has been shifted."
- Two ADRs that described the same mechanism drifted apart.
  [ADR-0002](0002-identity-at-zero.md) said the identity-at-zero shift is "computed
  once with `math` in the primitive's constructor"; ADR-0005 had recorded on
  2026-10-04 that it is computed lazily under `jax.ensure_compile_time_eval`. The
  review of round 6 found the stale sentence.
- References in the ADRs were plain text, so nothing checked them. ADR-0007 named a
  test, `test_strategies_are_static_under_filter_jit`, that does not exist (the test is
  `test_integrations_are_static_under_filter_jit`), and ADR-0014 cited a "decision 7"
  that it does not have.
- The reason behind a decision was spread through the Context, so a reader had to
  reconstruct it. For 0008, 0011, 0012 and 0014 it could not be recovered from the
  repository at all and had to be supplied by Joon
  ([the change document](../changes/2026-10-26-prose.md) lists them).

### Forces

ADRs in the original form of [Nygard 2011](../references.md) are immutable: a changed
decision is a new record that supersedes the old one, and the old text stays as
written. That keeps the record of what was believed when, but in this repository most
amendments are refinements made by the same author within days, and a reader wants the
design as it stands. Git already keeps every earlier version of each file, and
`git log -p` on an ADR reads as its diary.

## Decision

Every ADR has the sections Context (with the subsections Motivation and Forces),
Decision, Consequences, Alternatives rejected and History, in that order.

Motivation lists what made the decision necessary, one bullet per point, each with its
evidence: a symptom, a failing test, a benchmark, a `file:line` of the code before the
decision, or a need the project had and what it would have been without the decision. A
point that cannot be recovered is written *[not recorded]* and raised in the review.
Forces is prose: what any solution had to respect.

Decision and Consequences describe the current design, written as if it had always been
so. A change to the design rewrites them in the same commit. A design that a change
replaces moves to Alternatives rejected, marked *Superseded* with its date and the
reason. History has one dated line per change, with its branch, saying what changed;
the reason lives in Motivation or in Alternatives rejected, not in History.

A change that reverses the decision itself is a new ADR, and the old one's status
becomes "superseded by ADR-NNNN".

The first mention of an ADR, a design document or a code file in each section is a
relative Markdown link.
[`scripts/check_references.py`](../../scripts/check_references.py) fails on a relative
link whose target does not exist.

An ADR describes its own decision and links to another ADR for a mechanism that ADR
decides, instead of restating it.

## Consequences

- A reader of any ADR learns the current design from the main text, and learns how it
  got there from History and Alternatives rejected.
- An amendment costs a rewrite of the Decision section instead of an appended
  paragraph. The rewrite is reviewed like any other change, in the branch's change
  document.
- The ADRs no longer show, without git, what was believed before a change. The dated
  *Superseded* entries keep the reason a design was replaced; the old wording is in
  `git log -p`.
- [`docs/coding-standards.md`](../coding-standards.md) § Decision records states the
  format, and the link check runs in `prek` and CI with the citation check.

## Alternatives rejected

- Immutable ADRs (Nygard 2011): every amendment a new ADR. Correct for decisions made
  by many people over years; here it would have turned ADR-0011's run-layout change
  and ADR-0005's lazy shift into ADRs of their own, and a reader would still need to
  follow the chain to learn the current state.
- The newest decision on top, the earlier ones below it (Joon's first proposal in the
  round-6 review). A reader still meets the history before learning whether it
  applies, only in the other order.
- Amendments kept as appended sections, with a note at the top of each superseded
  paragraph. Cheaper to write, but the main text would still be wrong where it was not
  noted.
- A second script for the links. Rejected because `check_references.py` already walks
  the same files and answers the same question, whether a pointer resolves.

## History

- 2026-10-27 (`prose`): accepted; ADRs 0001–0018 restructured to this format in the
  same branch.
