---
type: decision
id: ADR-0016
status: accepted (review pending)
updated: 2026-10-28
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-22, plan `claude/isochron-auto-plan-2026-10-22`)
---

# ADR-0016 — Systems declared once, symbolically; JAX and AUTO-07p Fortran derived from the same expressions

## Context

### Motivation

- The isochrons of FitzHugh–Nagumo are to be computed by continuation of a two-point
  boundary value problem in AUTO-07p (Langfield et al. 2014; a coworker's reproduction
  in [`src/Isochron2PBVP`](https://github.com/Jhko725/deep_isochron/tree/3af99f2/src/Isochron2PBVP), since removed, multi-segment form). AUTO needs
  the vector field and its Jacobian as a Fortran `RHS` subroutine.
- Written by hand, that subroutine is a second copy of the physics next to the JAX
  `rhs`, and a third once the Jacobian is typed. The coworker's files are the instance:
  [`fncMultiSegIso.f90`](https://github.com/Jhko725/deep_isochron/blob/3af99f2/src/Isochron2PBVP/fncMultiSeg/fncMultiSegIso.f90) and
  [`fncIso.f90`](https://github.com/Jhko725/deep_isochron/blob/3af99f2/src/Isochron2PBVP/deprecated/fncIso.f90) each type the
  FitzHugh–Nagumo `F` and `A` by hand (lines 9–33).
- The Jacobian is not optional, because the adjoint stages of the extended system
  integrate `Aᵀw`.
- A normal form first wrote its template in a different shape from an observed
  system's: Hopf returned a tuple `(s, ρ̃(s), ω̃(s), params)` from `_symbolic_rates_sq`,
  Winfree a tuple in `r` from `_symbolic_rates`, and FitzHugh–Nagumo a `SymbolicODE`
  from `_symbolic_template`. The review of 2026-10-27 named this a "smell of
  inconsistent API design"
  ([the change document](../changes/2026-10-22-symbolic-systems.md), *Review notes*).

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
template. `AbstractODE` itself is unchanged: a learned or pushforward vector field is
still an ODE without being symbolic. The code generator takes an `AbstractSymbolicODE`,
and every symbolic system is one, the normal forms included.

### 2. A normal form declares its field in polar coordinates

`AbstractNormalForm` is an `AbstractSymbolicODE` whose subclasses write the classmethod
`_symbolic_template_polar() -> SymbolicODE`. The template has the states $(r, \theta)$
and the polar field $(\dot r, \dot\theta)$, so it is the type that an observed system
returns, in another chart. The base class reads the defining data off it, with
$\rho = \dot r / r$ canceled and $\omega = \dot\theta$. It derives the cartesian
`_symbolic_template` by substituting $r = \sqrt{x^2 + y^2}$ into
$\dot x = x\rho - y\omega$ and $\dot y = y\rho + x\omega$. SymPy reduces
$(\sqrt q)^2$ to $q$, so a form whose rates are functions of $r^2$ comes out polynomial
with no further step. The JAX hooks `log_growth_rate` and `angular_rate`, and the even
family's `_log_growth_rate_sq` and `_angular_rate_sq`, are compiled from the same
rates. `params()` and `symbolic()` are inherited from `AbstractSymbolicODE`.

When a normal form is created, the base class raises `ValueError` in three cases: the
template does not have two states, an expression depends on $\theta$ (a normal form is
rotationally symmetric), or $\dot r$ lacks the factor $r$, which would make $\rho$
singular at $r = 0$.

`AbstractEvenNormalForm` stays a declared family, because membership changes behavior
and is a precondition. It selects the square-root-free `rhs` and `eigenvalues_origin`,
and the `r_squared` integration refuses any other form
([ADR-0013](0013-normal-form-basins.md)). A form joins the family by declaring it as its
base class, and it writes the same hook as any normal form. The class then checks that
both rates are functions of $r^2$: after $r = \sqrt s$ is substituted, no power of $s$
with a non-integer exponent may remain. The test is stricter than evenness, and it is
the property that the square-root-free field needs. $\cos r$ is even, but
$\cos\sqrt s$ has no derivative at $s = 0$.

Both `rhs` overrides stay. The compiled cartesian field of Winfree's form has a `nan`
Jacobian at the origin, so `AbstractNormalForm.rhs` keeps its safe square root, and
`AbstractEvenNormalForm.rhs` keeps its square-root-free body.

The closed forms `h(r)` and `Ψ(r)` stay hand-derived JAX, since they are reviewed and
Winfree's carry basin logic; [`tests/test_symbolic.py`](../../tests/test_symbolic.py)
verifies them *symbolically* against the rates of the polar template: `ω + h'rρ = ω₁`
and `(log Ψ)' rρ = κ` with `κ = ρ'(1)`.
[ADR-0007](0007-systems-hierarchy-and-flow-strategies.md)'s rule that a class carries
only what is analytically available is unchanged; the language of the defining data
changed.

### 3. Compiled once per concrete class, through one extension point

`AbstractSymbolicODE.__init_subclass__` calls the classmethod `_compile` when the new
class has no abstract methods left. `AbstractSymbolicODE._compile` validates the
cartesian template and compiles it into `_compiled`. `AbstractNormalForm._compile`
calls `super()._compile()`, reads the rates off the polar template with the checks of
Decision 2, and compiles them into `_compiled_rates`. `AbstractEvenNormalForm._compile`
calls `super()._compile()`, checks that the rates are functions of $r^2$, and compiles
them in $s$ into `_compiled_rates_sq`. Every concrete class therefore has `_compiled`,
the normal forms included, although their `rhs` overrides do not read it; the build
costs one `sympy2jax` module per class at import and checks that the cartesian template
translates to JAX.

The guard "no abstract methods left" is the private function `_abstract_methods(cls)`,
which applies the rule of `ABCMeta.__new__` (CPython `Lib/_py_abc.py`): the class's own
abstract attributes, plus every inherited abstract name that the class does not override
with a concrete one. `inspect.isabstract(cls)` cannot be used, because `ABCMeta` sets
`__abstractmethods__` only after `type.__new__`, and so `__init_subclass__`, has run.
The rule needs no hook names. It skips `AbstractNormalForm`, whose derived
`_symbolic_template` is concrete while `_symbolic_template_polar` is still abstract, and
it fires on every concrete system. A concrete subclass of a concrete system recompiles
the template it inherits, which takes milliseconds.

The compiled forms are stored as `ClassVar`s. The expression structure is a class
property; only parameter values are per instance, and they are read at call time
(`getattr(self, name)`, so constrained values behind properties, Bautin's `a` and `b`,
work unchanged). A compiled form is not a dataclass field and not a pytree leaf, so
`filter_vmap`, `jacfwd`, `tree_at` and jaxtyping see nothing new. Under `jit`, the
Python bookkeeping in `rhs` runs at trace time; the jaxpr is the bare arithmetic
(checked: FHN traces to the same operations as the former hand-written field). A
malformed template fails at import.

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

- Adding a planar system is one classmethod, `_symbolic_template` for an observed
  system or `_symbolic_template_polar` for a normal form: `rhs`, `params()`, the
  Fortran `RHS` and Jacobian, the `PAR` dict and the constants file follow. A normal
  form chooses its family by its base class, and a wrong choice fails at import.
- A normal form's $\rho$ is compiled in SymPy's canceled form, an expanded polynomial
  for Hopf and Bautin rather than the factored product that the template writes.
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

- *Superseded 2026-10-27*: `_symbolic_rates() -> (r, ρ(r), ω(r), params)` on
  `AbstractNormalForm` and `_symbolic_rates_sq() -> (s, ρ̃(s), ω̃(s), params)` on
  `AbstractEvenNormalForm`, under an `AbstractNormalForm(AbstractODE)` that shared only
  the protocol `SupportsSymbolic` with the observed systems. Replaced because the
  template's shape differed from an observed system's and changed a second time for the
  even family (Motivation); the protocol lost its second adopter when the normal forms
  became `AbstractSymbolicODE`s.
- *Superseded 2026-10-27*: the guard "the class itself defines the hook concretely"
  (`defines_concretely`), which needed a special case for the even family's derived
  `_symbolic_rates`. Replaced by the `ABCMeta` rule of Decision 3.
- *Decision 1*: keeping the hand-written `rhs` and only generating the Fortran, which
  leaves two places to edit per system; the conformance test catches divergence but
  does not prevent it.
- *Decision 1*: tracing the JAX function to emit Fortran, a custom tracer for a problem
  that SymPy already solves, with no exact Jacobian.
- *Decision 2*: deriving `h`, `Ψ` by `sympy.integrate`, which is possible for Bautin's
  rational integrands; the reviewed closed forms are kept and checked instead.
- *Decision 2*: detecting evenness and dropping the family, either as a
  `ClassVar[bool]` that `r_squared` tests or as behavior that switches on the detection
  alone. A typo that added an odd power of $r$ to Bautin's $\rho$ would then move Bautin
  to the square-root `rhs` and remove `r_squared` without an error.
- *Decision 2*: $(\rho, \omega)$ in the `rhs` slot of a `SymbolicODE`, which keeps the
  type but stores something that is not a vector field.
- *Decision 2*: keeping `AbstractNormalForm` under `AbstractODE`, which keeps three
  hand-written `params()` (or a second copy of the derived one) and a protocol as the
  type that the two hierarchies share.
- *Decision 3*: an `__init_subclass__` per layer, which repeats the guard in three
  places and keeps the helper public.
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
- 2026-10-27 (`symbolic-systems`, review round 1): Decision 2 rewritten (normal forms
  write `_symbolic_template_polar` and subclass `AbstractSymbolicODE`; the even family is
  checked, not declared by a second hook); Decision 3 rewritten (the `ABCMeta` guard and
  the `_compile` extension point); `SupportsSymbolic` removed from Decision 1.
- 2026-10-28 (`auto-isochrons`): the links to `src/Isochron2PBVP` point at commit
  `3af99f2`, since the pipeline replaced the directory and it was deleted.
