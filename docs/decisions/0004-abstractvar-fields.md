---
type: decision
id: ADR-0004
status: accepted
updated: 2026-10-27
verified_by: joon
---

# ADR-0004 — `eqx.AbstractVar` is implemented as `eqx.field(static=True, init=False)`

## Context

### Motivation

- Type checkers rejected each of the runtime implementations of an `AbstractVar` (the
  table below), so the code carried `# ty: ignore` on them: in `systems/`,
  `dim: ClassVar[int] = … # ty: ignore`
  ([`2026-10-02-invertible-cleanup.md`](../changes/2026-10-02-invertible-cleanup.md),
  lines 48–49), and in the invertible package on the same names
  ([the scalar-param change document](../changes/2026-10-01-scalar-param-refactor.md),
  lines 99–103).

### Forces

Equinox's `AbstractVar` accepts any implementation at runtime: a class attribute, a
`ClassVar`, a property or a field. In contrast, type checkers (`ty`, pyright) see the
base-class `AbstractVar` as an ordinary instance attribute. This is
[equinox issue #1256](https://github.com/patrick-kidger/equinox/issues/1256): the
annotation is stripped at runtime but looks like a dataclass field to a checker, so it
becomes a phantom required `__init__` argument in subclasses.

The maintainer's position is that Python has no static spelling for "accessible
attribute, not an init field". The goal of this document is to choose a convention that
indicates that spelling *for the subclass*, while being compatible with type checkers.

The table below shows the verified behavior (equinox 0.13.8, ty 0.0.84, pyright 1.1) for
different conventions:

| subclass implementation | runtime | type checkers |
|---|---|---|
| bare class attribute `dim = 1` | fine | "argument missing for `dim`" at **every call site** |
| `dim: ClassVar[int] = 1` | fine | incompatible override at the definition (the old `# ty: ignore`) |
| `@property` | fine | incompatible override at the definition |
| `dim: int = eqx.field(static=True, default=1, init=False)` | fine | **accepted** |

The `init=False` field is runtime-equivalent to the others: not a constructor argument
(`Leaf(dim=5)` is a `TypeError`), not a pytree leaf, zero trainable size, one static
entry in the treedef.

## Decision

Implement every `AbstractVar` as `eqx.field(static=True, init=False)`: with `default=`
when the value is fixed by the class, assigned in a custom `__init__` when it is derived
(`SequentialINN.dim`, `ScalarChain.smoothness`). Use `eqx.AbstractClassVar` + `ClassVar`
only for values that are never derived per instance. That excludes `dim`, `num_params`
and `smoothness`, all of which have container-dependent or configuration-dependent
implementations. `ty check src/deep_isochron/model/invertible` is kept clean, with no
`# ty: ignore` on these names.

A subclass that narrows a base attribute keeps the base's annotation and adds a default:
the template classes declare
`raw: Float[Array, " num_params"] | None = eqx.field(static=True, default=None, init=False)`,
not `raw: None`.

## Consequences

- Pinned by `test_abstractvars_are_not_init_args` and `test_smoothness_declared`.
- The rule is in [`CLAUDE.md`](../../CLAUDE.md). `systems/` was migrated the same way in
  `invertible-cleanup` A8 (2026-10-02, `07f6979`); no `# ty: ignore` remains anywhere
  under `src/`.
- ty 0.0.88 enforces what the table above predicted and 0.0.84 let through. A
  `@property` implementing `num_params` is an `invalid-property-type-override`, a
  read-only attribute over a writable one. ty 0.0.88 also rejects narrowing a mutable
  attribute in a subclass (`invalid-mutable-override`), which is why the template
  classes' `raw` keeps the base's annotation.
- CI runs `uvx ty@latest` on purpose, because the checker is young and we want to be
  clean against its newest rules, while the dev group carries a locked copy for
  `uv run ty`; the two are reconciled with `uv lock --upgrade-package ty` and a
  fix-forward.

## Alternatives rejected

- A bare class attribute, a `ClassVar` or a `@property`: each fails a type checker, as
  the table shows.
- *Superseded 2026-10-15*: `num_params` as a `@property` on five classes (`ScalarChain`,
  `OffsetedBijection`, the three splines); replaced because ty 0.0.88 rejects it, by
  `num_params: int = eqx.field(static=True, init=False)` assigned in `__init__`, which
  is the decision as written.
- *Superseded 2026-10-15*: the template classes declaring `raw: None`; replaced because
  ty 0.0.88 rejects the narrowing.

## History

- 2026-10-01: accepted.
- 2026-10-02 (`invertible-cleanup`, A8): `systems/` migrated to the rule.
- 2026-10-15 (`ci-ty`): five `num_params` properties became fields and the templates'
  `raw: None` became the base annotation with a default, under ty 0.0.88; CI runs
  `uvx ty@latest`.
