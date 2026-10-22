r"""Vector fields declared once, symbolically, and projected onto JAX and Fortran.

An observed system (FitzHugh–Nagumo, Hodgkin–Huxley) writes its right-hand side as
SymPy expressions in a classmethod ``_symbolic_template``; everything else is derived
from that one definition: the JAX ``rhs`` (through ``sympy2jax``), ``params()``, and —
in ``analysis.ode.auto`` — the AUTO-07p Fortran ``RHS`` with its exact Jacobian. The
normal forms do the same one level down (their defining rates $\rho$, $\omega$ are
symbolic; see ``normal_forms.base``) and expose the same cartesian ``symbolic()`` view,
so the code generator sees one protocol, ``SupportsSymbolic`` (ADR-0016).

**Compilation happens once per class.** ``AbstractSymbolicODE.__init_subclass__``
builds the ``sympy2jax`` module when a *concrete* subclass is created and stores it as
a class attribute (``ClassVar``, not a dataclass field, not a pytree leaf): the
expression structure is a property of the class, only the parameter values are per
instance and they are read at call time. Under ``jit`` the Python bookkeeping in
``rhs`` (dict construction, ``getattr``) runs at trace time only and the jaxpr is the
bare arithmetic of the expressions. A malformed template fails at import, when the
class is created, not at the first call.

Conventions a template must follow (checked at class creation):

- parameter symbols are named after the fields (or properties) that hold their values,
  so ``getattr(self, symbol.name)`` is the value; ``params`` is keyed by those names and
  its order is the AUTO parameter order ``PAR(1..n)``;
- ``states`` are the cartesian coordinates in the order of the state vector;
- numeric literals are plain constants (``make_array=False``), never trainable leaves.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
import sympy as sp
import sympy2jax
from jaxtyping import Array, ArrayLike, Float
from sympy.codegen.cfunctions import expm1, log1p

from .base import AbstractODE


EXTRA_FUNCS: dict = {expm1: jnp.expm1, log1p: jnp.log1p}
"""SymPy functions without a ``sympy2jax`` translation that templates may use. The
Fortran printer knows neither, so a system that needs them (Hodgkin–Huxley) is not a
candidate for AUTO — which it is not anyway, being four-dimensional."""


class SymbolicODE(NamedTuple):
    """A vector field as SymPy: the single definition the JAX ``rhs`` and the AUTO
    Fortran are generated from.

    **Attributes:**

    - `states`: The cartesian coordinates, in the order of the state vector.
    - `params`: Parameter symbols keyed by the name of the attribute that holds their
        value (field or property); the order is the AUTO parameter order.
    - `rhs`: One expression per state, in `states` and `params` only.
    """

    states: tuple[sp.Symbol, ...]
    params: dict[str, sp.Symbol]
    rhs: tuple[sp.Expr, ...]

    @property
    def dim(self) -> int:
        return len(self.states)

    def validate(self) -> SymbolicODE:
        """Raise `ValueError` unless the conventions in the module docstring hold."""
        if len(self.rhs) != len(self.states):
            raise ValueError(
                f"{len(self.rhs)} expressions for {len(self.states)} states."
            )
        for name, symbol in self.params.items():
            if symbol.name != name:
                raise ValueError(
                    f"parameter symbol {symbol.name!r} must be named after its "
                    f"attribute {name!r}."
                )
        allowed = set(self.states) | set(self.params.values())
        for expr in self.rhs:
            stray = expr.free_symbols - allowed
            if stray:
                raise ValueError(f"free symbols {stray} are neither states nor params.")
        return self


@runtime_checkable
class SupportsSymbolic(Protocol):
    """What the AUTO code generator needs of a system: its cartesian ``symbolic()``
    view and the attributes named by that view's ``params``."""

    def symbolic(self) -> SymbolicODE: ...


class CompiledExpressions(NamedTuple):
    """SymPy expressions compiled to JAX, with the bookkeeping to evaluate them on an
    owner object: positional inputs by symbol name, parameters by attribute name.

    **Attributes:**

    - `module`: The `sympy2jax.SymbolicModule` of the expressions.
    - `inputs`: Names of the input symbols, in call order.
    - `params`: Attribute names (equal to the parameter symbols' names) read from the
        owner at every call.
    """

    module: sympy2jax.SymbolicModule
    inputs: tuple[str, ...]
    params: tuple[str, ...]

    def __call__(self, owner: Any, *values: ArrayLike) -> list[Array]:
        """The expressions at `values` (one per input, same order) with the owner's
        current parameters; a list, one array per expression."""
        inputs = dict(zip(self.inputs, values, strict=True))
        params = {name: getattr(owner, name) for name in self.params}
        return self.module(**inputs, **params)


def compile_expressions(
    exprs: tuple[sp.Expr, ...], inputs: tuple[sp.Symbol, ...], params: tuple[str, ...]
) -> CompiledExpressions:
    """Build a `CompiledExpressions` (numeric literals stay constants)."""
    module = sympy2jax.SymbolicModule(
        list(exprs), extra_funcs=EXTRA_FUNCS, make_array=False
    )
    return CompiledExpressions(module, tuple(s.name for s in inputs), tuple(params))


def is_abstract(attr: Any) -> bool:
    """Whether `attr` (a function, classmethod or staticmethod) is abstract."""
    func = getattr(attr, "__func__", attr)  # unwrap classmethod/staticmethod
    return bool(getattr(func, "__isabstractmethod__", False))


def defines_concretely(cls: type, name: str) -> bool:
    """Whether `cls` itself (not a base) defines a non-abstract attribute `name`.

    The guard for building per-class machinery in `__init_subclass__`:
    `inspect.isabstract(cls)` is not usable there, because `ABCMeta` computes
    `__abstractmethods__` only after `type.__new__` — and `__init_subclass__` — has
    run."""
    attr = cls.__dict__.get(name)
    return attr is not None and not is_abstract(attr)


class AbstractSymbolicODE(AbstractODE):
    """An ODE whose vector field is declared by `_symbolic_template`; `rhs`,
    `symbolic()` and `params()` are derived.

    A subclass supplies the classmethod `_symbolic_template` and the attributes it
    names (fields, or properties computing constrained values from raw leaves); the
    compiled form is built once when the class is created (see the module docstring).
    Intermediate abstract subclasses that do not define the template build nothing.
    """

    _compiled: ClassVar[CompiledExpressions]
    """Set on each concrete subclass by `__init_subclass__` (a class attribute, not an
    `AbstractVar` implementation — ADR-0004 does not apply; ADR-0016)."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not defines_concretely(cls, "_symbolic_template"):
            return
        sym = cls._symbolic_template().validate()
        cls._compiled = compile_expressions(sym.rhs, sym.states, tuple(sym.params))

    @classmethod
    @abc.abstractmethod
    def _symbolic_template(cls) -> SymbolicODE:
        """The vector field as SymPy, with symbols only (no values)."""

    def symbolic(self) -> SymbolicODE:
        """The cartesian symbolic view, for the code generator (`SupportsSymbolic`)."""
        return self._symbolic_template()

    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " {self.dim}"], args: Any = None
    ) -> Float[Array, " {self.dim}"]:
        """The vector field evaluated from the compiled template with the instance's
        parameter values (diffrax signature; `t` may be a Python float)."""
        del t, args
        components = type(self)._compiled(self, *u)
        return jnp.stack(components, axis=0)

    def params(self) -> dict[str, Any]:
        """The parameters the template names, as floats."""
        return {name: float(getattr(self, name)) for name in self.symbolic().params}


__all__ = [
    "AbstractSymbolicODE",
    "CompiledExpressions",
    "EXTRA_FUNCS",
    "SupportsSymbolic",
    "SymbolicODE",
    "compile_expressions",
    "defines_concretely",
    "is_abstract",
]
