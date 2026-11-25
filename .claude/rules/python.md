---
paths:
  - "**/*.py"
---

# Design, Python and JAX

These are Joon's rules for code, copied from his global `CLAUDE.md` of 2026-11-21, with this
repository's names in place of the general ones. The
reasoning and the review checklist are in the `code-conventions` skill, and the checks
that enforce them here are in `docs/coding-standards.md`.

## Design

- One way to do each thing. A second route to the same result is a defect: no parallel
  path, no shim, no optional parameter that keeps the old way working. Changing an
  interface updates every call site in the same change.
- Apply the deletion test to anything that looks like a pass-through: if deleting it
  concentrates complexity it earns its keep, and if complexity only moves, delete it.
- No unrequested machinery: two adopters before an abstraction, and a flag, option or
  stage nobody asked for stays unbuilt.
- Raise on an unknown value, take named arguments rather than `**kwargs`, and report
  which items failed rather than dropping them.
- Results are fixed-shape pytrees (`eqx.Module` or frozen dataclass) carrying what
  downstream needs; a ragged search record is returned beside the result, not inside it.

## Python and JAX

- A naturally jit- and vmap-compatible function is left undecorated, so the caller jits
  at the level it wants; a host-side driver pre-jits the stages it calls.
- jaxtyping annotations on every signature, test strategies and helpers included, and
  the annotation states the true shape.
- Trainable parameters: one unconstrained leaf, `constrain(raw)` as the single
  conversion site, the identity at `raw = 0`.
- `copy.replace`, since the project's Python floor is 3.13; `math.ceil`;
  `jnp.linspace(..., endpoint=False)`; `bool` for a binary switch; decimal literals
  such as `65536`.
- Tolerances and solver configurations are named entries with a one-line reason in
  `tests/helpers.py`, and call sites use the name.
- Docstrings follow the equinox/diffrax convention (`**Arguments:**`, `**Returns:**`);
  a constant or claim from a paper cites author, year and section.
- American English throughout, identifiers included.
