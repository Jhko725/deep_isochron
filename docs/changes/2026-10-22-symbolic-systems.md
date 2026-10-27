---
type: change
status: review pending
updated: 2026-10-22
branch: symbolic-systems
---

# `symbolic-systems` — one symbolic definition per system; JAX `rhs` and AUTO-07p Fortran derived from it

## Summary

First step of the AUTO isochron work (roadmap O7; plan `claude/isochron-auto-plan-2026-10-22` in the Project; decisions 1–6 there). The observed systems and the normal forms now declare their vector field *once*, as SymPy expressions; the JAX `rhs` (and the normal forms' rate hooks) are compiled from those expressions once per class, and the same expressions generate the Fortran `RHS` — with its exact Jacobian — that AUTO-07p needs, `INCLUDE`d into a generic, hand-maintained multi-segment BVP file whose parameter slots are named constants. Nothing here runs AUTO; the pipeline that does is the next branch (`auto-isochrons`, ADR-0017). ADR-0016 records the decisions; ADR-0007 is amended (the normal forms' defining data).

## Files

- `pyproject.toml` / `uv.lock`: `sympy` and `sympy2jax` (and `mpmath`) added as core dependencies (`uv add`; `rhs` depends on them). The lock diff looks large only because the session's uv (0.11) writes lock revision 3 (adds `upload-time` to every entry); no other version changed. `uv lock` with Joon's uv restores his format if preferred.
- `src/deep_isochron/systems/symbolic.py` (**new**): `SymbolicODE` (states, params keyed by attribute name, rhs; `validate()`), `CompiledExpressions` + `compile_expressions` (the sympy2jax module with its call bookkeeping; `make_array=False`, `EXTRA_FUNCS` for `expm1`/`log1p`), `defines_concretely`/`is_abstract` (the `__init_subclass__` guard), `AbstractSymbolicODE` (`_symbolic_template` classmethod abstract; `__init_subclass__` compiles; `rhs`, `symbolic()`, `params()` derived), `SupportsSymbolic` protocol.
- `src/deep_isochron/systems/base.py`: `params()` iterates `dataclasses.fields(self)` (excludes the `ClassVar` pseudo-field `_compiled`).
- `src/deep_isochron/systems/fitzhugh_nagumo.py`: rewritten as a template (module docstring with the model and its facts; the hand-written `rhs` is gone).
- `src/deep_isochron/systems/hodgekin_huxley.py`: rewritten as a template with `expm1` for the removable singularities of `α_m`, `α_n`; the `alpha_*`/`beta_*` methods are gone (they were used only by `rhs`).
- `src/deep_isochron/systems/normal_forms/base.py`: `AbstractNormalForm._symbolic_rates` (classmethod, abstract), `__init_subclass__` compiling `_compiled_rates` (and `_compiled_rates_sq` for the even family), `log_growth_rate`/`angular_rate` derived (were abstract), `symbolic()`; `AbstractEvenNormalForm._symbolic_rates_sq` (abstract), `_symbolic_rates` derived by `s → r²`, `_log_growth_rate_sq`/`_angular_rate_sq` derived (were abstract), `symbolic()` without square root; helpers `_validate_rates`, `_cartesian`.
- `src/deep_isochron/systems/normal_forms/{hopf,bautin,winfree}.py`: the two JAX hooks replaced by one `_symbolic_rates[_sq]` classmethod each; everything else unchanged.
- `src/deep_isochron/analysis/ode/auto/__init__.py` (**new**), `codegen.py` (**new**): `Idx`/`internal_indices`, `parameter_map`, `auto_pars`, `rhs_block`, `indices_block`, `constants_file`, `generic_source`, `AutoProblem` (`slot(name)`, `fortran_sha`), `write_problem`; safe `&`…`&` line wrapping (`LINE_WIDTH`).
- `src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90` (**new**): the coworker's `fncMultiSegIso.f90` with `RHS` replaced by `INCLUDE 'rhs.inc'`, every internal `PAR(k)` literal replaced by an `IP_*` constant from `INCLUDE 'indices.inc'`, and a header explaining the staged system. Shipped as package data (hatchling includes it).
- `tests/test_symbolic.py` (**new**), `tests/test_auto_codegen.py` (**new**), `tests/golden/fitzhugh_nagumo_rhs.inc` (**new**): see *Tests*.
- `docs/decisions/0016-symbolic-systems.md` (**new**); `docs/decisions/0007-systems-hierarchy-and-flow-strategies.md` (amendment 2026-10-22); `docs/design/normal-forms.md` (§9 note); `docs/architecture.md` (module map: `symbolic.py`, `analysis/ode/auto/`; systems prose; ADR table); `docs/index.md` (ADR table); `docs/roadmap.md` (O7 row; Done ledger).

## Design

- One definition, SymPy, both projections derived — ADR-0016 §1–2. The alternative of keeping the hand-written `rhs` and generating only the Fortran was rejected because it leaves two places to edit; the alternative of deriving `h`, `Ψ` by `sympy.integrate` was rejected in favor of keeping the reviewed closed forms and *verifying* them symbolically (plan open question 1, default taken).
- Compiled per class in `__init_subclass__`, stored as a `ClassVar` — ADR-0016 §3, with the guard for abstract intermediates (`inspect.isabstract` is not usable there). `AbstractEvenNormalForm` is the subtle case: it defines a *derived* `_symbolic_rates`, so the guard also checks that `_symbolic_rates_sq` is still abstract before building. A module-level `functools.cache` and a module-as-field were rejected (ADR-0016 §3).
- Parameter symbols named after attributes, values read by `getattr` at call time — so Bautin's constrained `a`, `b` (properties over `raw_a`, `raw_b`) work unchanged, the normal forms' parameters stay differentiable leaves, and `params()`, the `PAR` dict and the Fortran slot numbering share one order.
- `INCLUDE`, not Jinja — ADR-0016 §4. The generic Fortran file is kept as Fortran. Line wrapping is done here with a leading `&` on continuations (SymPy's wrapper can cut `**` into `* *`).
- Slots: `PAR(1..n)` for the system, the pipeline's block after `max(n, 10)`; `NPAR = max(last, 36)`; more than ten parameters raises (ADR-0016 §5). `write_problem` refuses `dim ≠ 2`.
- `constants_file` reproduces the coworker's `c.fncMultiSeg` (NTST 300, NCOL 5, EPS 1e-10) with `NDIM`, `NPAR`, `THL` generated; run-specific constants are the pipeline's business.
- The generic file keeps the `NDIM = 2` branches (`PVLS`'s anchor, `STPNT`) that the plan's step 4 drops together with the Hopf path: without a Fortran compiler in this environment the file was changed only where the change is mechanical (renaming), so the next branch — which can compile it against AUTO — makes the structural edit.

## Bugs fixed

- None in shipped code. Noted for the next branch (in the coworker's script, not touched here): `UZSTOP={36: …, 74: …, 74: …}` has a duplicate key; the second `74` is presumably `75`, so the `y`-bound of the isochron continuation was never applied.

## Tests

`uv run pytest tests/test_symbolic.py tests/test_auto_codegen.py` (28 + 24 tests, ~15 s; the four `f2py` tests skip without `gfortran`). The existing `test_systems.py` and `test_normal_forms.py` pass unchanged (36 tests), which pins that the derived fields and hooks reproduce the former hand-written ones numerically. The rest of the suite (`-m "not slow"`, 4 workers) did not finish inside the session's background time limit in this sandbox; CI's `checks` job is the first full run — the modules it exercises beyond those four files use the systems only through `rhs`/`flow`, which the four files pin.

- `test_symbolic.py`: (1) derived `rhs` equals SymPy's own `lambdify` of the template on all seven system instances (Hypothesis points in the basin), jit/vmap/`jacfwd` through the derived field against the symbolic Jacobian; the even hooks equal the template; (2) the compiled form is a class attribute of the concrete class only — not a dataclass field, not a pytree leaf, no per-instance rebuild — abstract intermediates build nothing, and a misnamed parameter symbol, a stray symbol or a wrong expression count fail at class creation; (3) the closed forms `h`, `Ψ` of Hopf, Bautin and Winfree, transcribed from their docstrings, satisfy `ω + h'rρ = ω₁` and `(log Ψ)' rρ = κ`, `Ψ(1) = 0`, `Ψ'(1) = 1`, `h(1) = 0` symbolically (the logarithmic derivative because `simplify` does not close the product form with Bautin's symbolic exponent `b`); (4) cartesian → polar of `symbolic()` recovers the rates with no θ-dependence for the normal forms and does not separate for FHN; the even forms' cartesian fields are polynomial, Winfree's is not.
- `test_auto_codegen.py`: (1) the printed `F(i)`, `A(i,j)` parsed back from the Fortran text into SymPy equal the template's expressions and Jacobian at random points (no compiler needed); every line fits `LINE_WIDTH`, continuations are marked on both ends; the FHN block equals the golden file `tests/golden/fitzhugh_nagumo_rhs.inc` (the coworker's hand-written `RHS` up to term order); (2) the slot block for 0–10 parameters starts at 11 and is contiguous and fully named; 11 raises; (3) the generic file references only `IP_*` slots, all declared by `indices.inc`, and has both `INCLUDE`s; (4) `write_problem` lays down the four files, the `PAR` dict, `slot("c") == 3`, a stable source hash, `NDIM`/`NPAR`/`THL` in the constants; Hodgkin–Huxley is refused; (5) `slow`, skipped without `gfortran`: the generated `RHS` compiled through `numpy.f2py` agrees with `rhs` and `jacfwd(rhs)` — **not run here** (no compiler in the sandbox); first run on Joon's machine will tell whether `numpy.f2py -c` needs `meson`/`ninja` installed (NumPy 2) — if so, add them to the dev group.

## Open issues

- The `f2py` test and the `INCLUDE` resolution inside AUTO's own build are unverified in this environment (no `gfortran`); both are the first thing to run on a machine with a compiler.
- `src/Isochron2PBVP` (the coworker's script, Fortran and 100 CSVs) is left in place; the next branch removes it once the pipeline reproduces its output.
- Hypothesis prints `FitzhughNagumo(_compiled=…)` in failure reports because its pretty-printer walks `__dataclass_fields__`, which lists `ClassVar` pseudo-fields; cosmetic (the repo's own `__repr__` is unaffected).
- `sympy2jax` emits `Mul(-1, x)` rather than `neg`; numerically irrelevant, so no HLO-structure test is written (plan open question 3, default taken).

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `pyproject.toml` / `uv.lock`: sympy, sympy2jax core deps | | |
| `systems/symbolic.py`: the layer (`SymbolicODE`, `AbstractSymbolicODE`, compile machinery, protocol) | | |
| `systems/base.py`: `params()` over `dataclasses.fields` | | |
| `systems/fitzhugh_nagumo.py`: template | | |
| `systems/hodgekin_huxley.py`: template with `expm1` | | |
| `systems/normal_forms/base.py`: symbolic rates, derived hooks, `symbolic()` | | |
| `systems/normal_forms/hopf.py`: `_symbolic_rates_sq` | | |
| `systems/normal_forms/bautin.py`: `_symbolic_rates_sq` | | |
| `systems/normal_forms/winfree.py`: `_symbolic_rates` | | |
| `analysis/ode/auto/__init__.py`, `codegen.py`: the generator | | |
| `analysis/ode/auto/fortran/isochron_multiseg.f90`: generic BVP with named slots and includes | | |
| `tests/test_symbolic.py` | | |
| `tests/test_auto_codegen.py`, `tests/golden/fitzhugh_nagumo_rhs.inc` | | |
| `docs/decisions/0016-symbolic-systems.md` | | |
| `docs/decisions/0007-…`: amendment | | |
| `docs/design/normal-forms.md`: §9 note | | |
| `docs/architecture.md`, `docs/index.md`, `docs/roadmap.md` | | |
