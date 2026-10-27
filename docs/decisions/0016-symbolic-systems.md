---
type: decision
id: ADR-0016
status: accepted (review pending)
updated: 2026-10-27
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-22, plan `claude/isochron-auto-plan-2026-10-22`)
---

# ADR-0016 — Systems declared once, symbolically; JAX and AUTO-07p Fortran derived from the same expressions

## Context

### Motivation

- The isochrons of FitzHugh–Nagumo are to be computed by continuation of a two-point
  boundary value problem in AUTO-07p (Langfield et al. 2014; a coworker's reproduction
  in [`src/Isochron2PBVP`](../../src/Isochron2PBVP), multi-segment form). AUTO needs
  the vector field and its Jacobian as a Fortran `RHS` subroutine.
- Written by hand, that subroutine is a second copy of the physics next to the JAX
  `rhs`, and a third once the Jacobian is typed. The coworker's files are the instance:
  [`fncMultiSegIso.f90`](../../src/Isochron2PBVP/fncMultiSeg/fncMultiSegIso.f90) and
  [`fncIso.f90`](../../src/Isochron2PBVP/deprecated/fncIso.f90) each type the
  FitzHugh–Nagumo `F` and `A` by hand (lines 9–33).
- The Jacobian is not optional, because the adjoint stages of the extended system
  integrate `Aᵀw`.

### Forces

The project will want isochrons of other planar systems (the normal forms, for an
end-to-end check against their closed-form phase), so the tie must be generic, not an
FHN test. A learned or pushforward vector field is still an ODE without being
symbolic, so the symbolic declaration cannot be a requirement of every ODE.

## Decision

### 1. One definition, in SymPy; `rhs` is derived

`systems.symbolic.AbstractSymbolicODE(AbstractODE)` requires a classmethod
`_symbolic_template() -> SymbolicODE`: the states, the parameter symbols keyed by the
attribute that holds their value, and one expression per state. `rhs` is concrete and
evaluates the compiled expressions with the instance's current parameter values;
`params()` and the AUTO Fortran (`analysis.ode.auto.codegen`) are derived from the same
tuple. `AbstractODE` itself is unchanged: a learned or pushforward vector field is still
an ODE without being symbolic. The code generator sees one protocol, `SupportsSymbolic`
(`symbolic() -> SymbolicODE`).

### 2. The normal forms' defining data are symbolic too

`AbstractNormalForm` requires `_symbolic_rates() -> (r, ρ(r), ω(r), params)` and
`AbstractEvenNormalForm` requires `_symbolic_rates_sq() -> (s, ρ̃(s), ω̃(s), params)`,
from which `_symbolic_rates` is derived by `s → r²`. The JAX hooks `log_growth_rate`,
`angular_rate`, `_log_growth_rate_sq`, `_angular_rate_sq` are now concrete and derived;
a concrete normal form supplies its two expressions and nothing else about the field.
The cartesian `symbolic()` is the chain rule `ẋ = xρ − yω`, `ẏ = yρ + xω` with
`r = sqrt(x² + y²)` substituted, or `s = x² + y²` for the even family, which is why
their cartesian fields are polynomial.

The closed forms `h(r)`, `Ψ(r)` stay hand-derived JAX, since they are reviewed and
Winfree's carry basin logic; [`tests/test_symbolic.py`](../../tests/test_symbolic.py)
verifies them *symbolically* against the symbolic rates: `ω + h'rρ = ω₁` and
`(log Ψ)' rρ = κ` with `κ = ρ'(1)`.
[ADR-0007](0007-systems-hierarchy-and-flow-strategies.md)'s rule that a class carries
only what is analytically available is unchanged; the language of the defining data
changed.

### 3. Compiled once per class, in `__init_subclass__`, as a class attribute

The `sympy2jax` module for a template is built when the concrete class is created and
stored as a `ClassVar` (`_compiled`, `_compiled_rates`, `_compiled_rates_sq`). The
expression structure is a class property; only parameter values are per instance, and
they are read at call time (`getattr(self, name)`, so constrained values behind
properties, Bautin's `a` and `b`, work unchanged). It is not a dataclass field and not a
pytree leaf, so `filter_vmap`, `jacfwd`, `tree_at` and jaxtyping see nothing new. Under
`jit`, the Python bookkeeping in `rhs` runs at trace time; the jaxpr is the bare
arithmetic (checked: FHN traces to the same operations as the former hand-written
field). A malformed template fails at import.

The guard builds only when `_symbolic_template` (or the rates hook) is defined on
*this* class and is not abstract, because `inspect.isabstract(cls)` is unreliable
inside `__init_subclass__`, which runs before `ABCMeta` computes `__abstractmethods__`.
Abstract intermediates (`AbstractEvenNormalForm` defines a *derived* `_symbolic_rates`
while its `_symbolic_rates_sq` is still abstract) build nothing.

**These class attributes are an edge of ADR-0004, not a violation.**
[ADR-0004](0004-abstractvar-fields.md) forbids class attributes *for `AbstractVar`
implementations*. They implement no `AbstractVar`; `ty` accepts them as `ClassVar`.
Recorded so that nobody "fixes" them into fields.

### 4. Fortran assembled by `INCLUDE`, not a template engine

The generic equation file
[`analysis/ode/auto/fortran/isochron_multiseg.f90`](../../src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90),
the coworker's staged multi-segment BVP with its literal `PAR(k)` slots renamed to
`IP_*` constants, is plain, hand-maintained Fortran. Per system, `codegen` writes two
includes next to it: `rhs.inc`, the body of `RHS` with `F(i)` and the exact Jacobian
`A(i,j)`, printed by SymPy's Fortran printer with states as `U(i)` and parameters as
`PAR(j)`; and `indices.inc` (`INTEGER, PARAMETER :: IP_PERIOD = 11, …`). The generic
file stays compilable, diffable and reviewable as Fortran; a wrong slot is a compile
error rather than a silently wrong `PAR(13)`.

Long expressions are wrapped by `codegen` itself with a trailing *and* a leading `&`,
which makes the split point irrelevant; SymPy's own wrapper omits the leading `&` and
can cut `**` in two.

### 5. Parameter slots

System parameters occupy `PAR(1..n)` in template order. The pipeline's block starts
after `max(n, 10)`, so the period is `PAR(11)`, what AUTO's `IPS = 2` expects, for up to
ten parameters; more than ten raises `NotImplementedError` until the period copy is
implemented (plan step 4; decision 9 makes it unnecessary once the orbit comes from
`analysis.ode`). The generic file assumes `dim = 2` (segment slicing, the stable
direction as the perpendicular of `w`); `write_problem` refuses other dimensions.
Hodgkin–Huxley is symbolic for `rhs` only.

## Consequences

- Adding a planar system is one classmethod: `rhs`, `params()`, the Fortran `RHS` and
  Jacobian, the `PAR` dict and the constants file follow.
- `sympy` and `sympy2jax` are core dependencies (`rhs` depends on them). `EXTRA_FUNCS`
  maps `expm1`/`log1p` (used by Hodgkin–Huxley for its removable singularities) to
  `jnp`; the Fortran printer knows neither, so such a system is not AUTO material, and
  it is not planar anyway.
- `sympy2jax` emits `Mul(-1, x)` rather than a negation; harmless after XLA, but a test
  of HLO *structure* would differ from hand-written code. Tests are numeric.
- `AbstractODE.params()` iterates `dataclasses.fields(self)`, not `__dataclass_fields__`,
  which lists `ClassVar` pseudo-fields.
- The AUTO pipeline itself (the staged runs, the dataclasses, the dependency script) is
  the next branch, ADR-0017.

## Alternatives rejected

- *Decision 1*: keeping the hand-written `rhs` and only generating the Fortran, which
  leaves two places to edit per system; the conformance test catches divergence but
  does not prevent it.
- *Decision 1*: tracing the JAX function to emit Fortran, a custom tracer for a problem
  that SymPy already solves, with no exact Jacobian.
- *Decision 2*: deriving `h`, `Ψ` by `sympy.integrate`, which is possible for Bautin's
  rational integrands; the reviewed closed forms are kept and checked instead.
- *Decision 3*: a module-level `functools.cache` keyed on the class, which has the same
  cost, fails lazily and is a global.
- *Decision 3*: the module as a field, which is rebuilt per instance and per `tree_at`
  and would need `static=True`.
- *Decision 4*: Jinja, a dependency after which the generic file would no longer be
  Fortran.
- *Decision 4*: `sympy.codegen.ast` for the whole routine, the staging logic as a
  Python AST, unreadable.

## History

- 2026-10-22 (`symbolic-systems`): accepted. Amends ADR-0007 (the normal forms'
  defining data, Decision 2) and notes an edge of ADR-0004 (Decision 3).
- 2026-10-27 (`prose`): restructured to the format of
  [ADR-0019](0019-adrs-describe-the-current-design.md).
