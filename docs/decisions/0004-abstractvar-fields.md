# ADR-0004 — `eqx.AbstractVar` is implemented as `eqx.field(static=True, init=False)`

**Status**: accepted (2026-10-01)

## Context

Equinox's `AbstractVar` accepts any implementation at runtime — class attribute, `ClassVar`,
property or field. Type checkers (`ty`, pyright) see the base-class `AbstractVar` as an
ordinary instance attribute. Verified behaviour (equinox 0.13.8, ty 0.0.84, pyright 1.1):

| subclass implementation | runtime | type checkers |
|---|---|---|
| bare class attribute `dim = 1` | fine | "argument missing for `dim`" at **every call site** |
| `dim: ClassVar[int] = 1` | fine | incompatible override at the definition (the old `# ty: ignore`) |
| `@property` | fine | incompatible override at the definition |
| `dim: int = eqx.field(static=True, default=1, init=False)` | fine | **accepted** |

The `init=False` field is runtime-equivalent to the others: not a constructor argument
(`Leaf(dim=5)` is a `TypeError`), not a pytree leaf, zero trainable size, one static entry in
the treedef.

## Decision

Implement every `AbstractVar` as `eqx.field(static=True, init=False)`: with `default=` when
the value is fixed by the class, assigned in a custom `__init__` when it is derived
(`SequentialINN.dim`, `ScalarChain.smoothness`). Use `eqx.AbstractClassVar` + `ClassVar`
only for values that are never derived per instance — which excludes `dim`, `num_params`
and `smoothness`, all of which have container or configuration-dependent implementations.
`ty check src/deep_isochron/model/invertible` is kept clean; no `# ty: ignore` on these
names.

## Consequences

- Pinned by `test_abstractvars_are_not_init_args` and `test_smoothness_declared`.
- The rule is in `CLAUDE.md`; `systems/` still carries `dim: ClassVar[int]` with
  `# ty: ignore` and should be migrated the same way when touched.
