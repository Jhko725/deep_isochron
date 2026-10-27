---
type: change
status: in progress; the design document is out for review
updated: 2026-10-27
branch: auto-isochrons
---

# `auto-isochrons` — isochrons of a planar system from AUTO-07p (roadmap O7, step 2)

## Summary

This branch builds the AUTO-07p isochron pipeline of roadmap O7. The plan is the Project
document "Isochrons from AUTO-07p" (`claude/isochron-auto-plan-2026-10-22`, revised
2026-10-27), whose seventeen decisions were settled with Joon on 2026-10-27.

The first commit is the design document [`design/isochrons.md`](../design/isochrons.md),
which Joon asked for before any of the pipeline is written. It derives the method,
breaks down the published version (Langfield et al. 2014), the coworker's version and
ours, and records a defect found while writing it: the coworker's bundle stage
integrates $\dot w = +A^{\mathsf T}w$ instead of the adjoint equation
$\dot w = -A^{\mathsf T}w$. On FitzHugh–Nagumo this gives a transverse but wrong end
direction, $58°$ from the stable direction, and on the normal forms it finds no branch
point at all. The design document's §11 asks five questions, D1–D5; the pipeline
waits for the answers.

## Files

- `docs/design/isochrons.md` (**new**): the design document, §1–11.
- `docs/references.md`: entries for Hannam et al. 2022, Krauskopf & Rieß 2008 and
  Osinga & Moehlis 2010.
- `docs/index.md`: the design document's row.
- `docs/roadmap.md`: the O7 row points to the design document.
- `docs/changes/2026-10-27-auto-isochrons.md` (**new**): this document.

## Design

- **The design document precedes the pipeline.** The numerical-method skill puts the
  derivation before the code, and the bundle defect shows why: the plan's Step E
  oracle "bundle against O2" would have failed against the coworker's equation, with
  no document saying which side was wrong.
- **The document describes three versions of the method side by side.** Langfield's
  fixed base point with segments of $(k + 1 - \vartheta)T$ (§4), the coworker's
  rotating base point after Hannam et al. 2022 (§5), and ours (§6). A document about
  our version alone was rejected, because the review needs to see what each of our
  replacements replaces.
- **The choice of bundle equation is left open (D1).** The adjoint sign is measured to
  work, but Joon deferred the decision to the document.

## Bugs fixed

None yet. Two defects found while preparing the branch are recorded under *Open issues*.

## Tests

No code has changed. The document's numbers were measured on 2026-10-27 with AUTO-07p
built at commit `44cc1c4` (serial `make`; `make -j4` fails on `support.mod`) and
GNU Fortran 13.3.0, by running the coworker's script and variants of it in a scratch
directory:

- the coworker's runs 0–5, the base point $(-0.19007, -0.12319)$ and the bundle
  $w = (-0.58385, -0.81186)$, compared with $DF/\vert DF\vert$ and with $v_s$ from the
  variational transition matrix integrated by `Dopri8` at `rtol = 1e-11`;
- the same runs with `F(5:6)` and `F(7:8)` negated: $w = (-0.37971, 0.92511)$, $Jw = -v_s$
  to eight digits;
- the transition matrix of $\dot w = A^{\mathsf T}w$ for `HopfNormalForm(1.3, 1.5, 0.4)`
  and `BautinNormalForm(1.0, 0.3, 1.5, 0.5)`, with no multiplier 1;
- isochrons 1, 26 and 51 rerun, timed and compared with the stored files by the distance
  from points to the other curve's polyline ($3.6\times 10^{-6}$), and isochron 26 with
  the negated bundle ($8.1\times 10^{-6}$ from the stored file; 272 s);
- the stored points of files 26, 51 and 100 integrated forward four periods.

## Open issues

- **`indices.inc` does not compile.** `indices_block` writes every slot on one line of
  about 160 characters, past gfortran's 132. Fixed in Step A of the plan.
- **The bundle equation has the wrong sign.** `isochron_multiseg.f90` keeps the
  coworker's $+A^{\mathsf T}$. The fix waits for D1.
- **Decision 13's tolerance rested on a spacing artifact.** The $7.8\times 10^{-4}$
  between the rerun and the stored files was a nearest-point distance; the curves agree
  to $3.6\times 10^{-6}$. D3 proposes $2\times 10^{-5}$.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/design/isochrons.md` — the design document: three versions of the method, AUTO background, the stable direction and the $+A^{\mathsf T}$ defect, accuracy, oracles, D1–D5 | | |
| `docs/references.md` — Hannam et al. 2022, Krauskopf & Rieß 2008, Osinga & Moehlis 2010 | | |
| `docs/index.md` — the design document's row | | |
| `docs/roadmap.md` — O7 row points to the design document | | |
| `docs/changes/2026-10-27-auto-isochrons.md` — this change document | | |
