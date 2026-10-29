r"""Vector fields declared once, as SymPy expressions, with JAX and Fortran derived.

A system writes its vector field in a classmethod of an ``AbstractSymbolicODE``:
``_symbolic_template`` in cartesian coordinates for an observed system such as
FitzHugh–Nagumo, or ``_symbolic_template_polar`` for a normal form
(``normal_forms.base``), whose cartesian template is then derived. The JAX ``rhs``,
``params()``, and, in ``analysis.ode.auto``, the AUTO-07p Fortran ``RHS`` with its
exact Jacobian all come from that one definition.

The expressions are compiled when a class with no abstract methods left is created:
``__init_subclass__`` calls the classmethod ``_compile``, which each layer extends
through ``super()``. The compiled forms are class attributes, so a malformed template
fails when the class is created and only the parameter values are read per call.
Under ``jit`` the jaxpr is the bare arithmetic of the expressions. A template must name
each parameter symbol after the attribute that holds its value, list ``states`` in the
order of the state vector, and write numeric literals as plain constants;
``SymbolicODE.validate`` checks the first two.

ADR-0016 has the reasoning, the rejected alternatives and the ADR-0004 edge.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp
import sympy as sp
import sympy2jax
from jaxtyping import Array, ArrayLike, Float
from sympy.codegen.cfunctions import expm1, log1p

from .base import AbstractODE


EXTRA_FUNCS: dict = {expm1: jnp.expm1, log1p: jnp.log1p}
"""SymPy functions that templates may use and ``sympy2jax`` does not translate on its
own. The Fortran printer knows neither, so a system that needs them, such as
Hodgkin–Huxley, cannot be written for AUTO."""


@dataclass(frozen=True)
class SymbolicODE:
    """A vector field as SymPy: the single definition that the JAX ``rhs`` and the
    AUTO Fortran are generated from.

    **Attributes:**

    - `states`: The cartesian coordinates, in the order of the state vector.
    - `params`: Parameter symbols keyed by the name of the attribute (field or
        property) that holds their value; the order is the AUTO parameter order
        ``PAR(1..n)``.
    - `rhs`: One expression per state, in `states` and `params` only.
    """

    states: tuple[sp.Symbol, ...]
    params: dict[str, sp.Symbol]
    rhs: tuple[sp.Expr, ...]

    @property
    def dim(self) -> int:
        """The number of states."""
        return len(self.states)

    def validate(self) -> SymbolicODE:
        """Check the template conventions and return `self`.

        **Raises:**

        `ValueError` when the number of expressions differs from the number of states, a
        parameter symbol is not named after its key, or an expression has a free symbol
        that is neither a state nor a parameter.
        """
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


@dataclass(frozen=True)
class CompiledExpressions:
    """SymPy expressions compiled to JAX, with the bookkeeping to evaluate them on an
    owner object: positional inputs by symbol name, parameters by attribute name.

    **Attributes:**

    - `module`: The ``sympy2jax.SymbolicModule`` of the expressions.
    - `inputs`: The names of the input symbols, in call order.
    - `params`: The attribute names, equal to the parameter symbols' names, read from
        the owner at every call.
    """

    module: sympy2jax.SymbolicModule
    inputs: tuple[str, ...]
    params: tuple[str, ...]

    def __call__(self, owner: Any, *values: ArrayLike) -> list[Array]:
        """Evaluate the expressions with the owner's current parameter values.

        **Arguments:**

        - `owner`: The object whose attributes hold the parameter values.
        - `values`: One value per input symbol, in the order of `inputs`.

        **Returns:**

        A list with one array per expression.
        """
        inputs = dict(zip(self.inputs, values, strict=True))
        params = {name: getattr(owner, name) for name in self.params}
        return self.module(**inputs, **params)

    @classmethod
    def from_expressions(
        cls,
        exprs: tuple[sp.Expr, ...],
        inputs: tuple[sp.Symbol, ...],
        params: tuple[str, ...],
    ) -> CompiledExpressions:
        """Compile `exprs`; numeric literals stay constants rather than becoming
        trainable leaves.

        **Arguments:**

        - `exprs`: The expressions to compile.
        - `inputs`: The symbols supplied positionally at each call.
        - `params`: The attribute names of the remaining free symbols.

        **Returns:**

        The compiled expressions.
        """
        module = sympy2jax.SymbolicModule(
            list(exprs), extra_funcs=EXTRA_FUNCS, make_array=False
        )
        return cls(module, tuple(s.name for s in inputs), tuple(params))


def _abstract_methods(cls: type) -> set[str]:
    """The names of the abstract methods that `cls` still has.

    This is the rule that ``ABCMeta.__new__`` applies (CPython ``Lib/_py_abc.py``): the
    class's own abstract attributes, plus every abstract name of a base that `cls`
    does not override with a concrete one. ``ABCMeta`` stores the result in
    ``__abstractmethods__`` only after ``type.__new__``, and so ``__init_subclass__``,
    has run, which is why the guard computes it itself.
    """

    def is_abstract(attr: Any) -> bool:
        func = getattr(attr, "__func__", attr)  # unwrap classmethod and staticmethod
        return bool(getattr(func, "__isabstractmethod__", False))

    own = {name for name, value in cls.__dict__.items() if is_abstract(value)}
    inherited = {
        name
        for base in cls.__bases__
        for name in getattr(base, "__abstractmethods__", ())
        if is_abstract(getattr(cls, name, None))
    }
    return own | inherited


class AbstractSymbolicODE(AbstractODE):
    """An ODE whose vector field is declared by `_symbolic_template`, with `rhs`,
    `symbolic()` and `params()` derived.

    A subclass supplies the classmethod `_symbolic_template` and the attributes it
    names: fields, or properties computing constrained values from raw leaves. When a
    class with no abstract methods left is created, `__init_subclass__` calls
    `_compile`; a layer that compiles more, such as `AbstractNormalForm`, extends
    `_compile` and calls `super()._compile()` first.

    **Attributes:**

    - `_compiled`: The compiled cartesian template, set on each concrete class by
        `_compile`. A class attribute rather than a field: it implements no
        `AbstractVar`, so ADR-0004 does not apply (ADR-0016).
    """

    _compiled: ClassVar[CompiledExpressions]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not _abstract_methods(cls):
            cls._compile()

    @classmethod
    def _compile(cls) -> None:
        """Validate the cartesian template and compile it into `_compiled`.

        **Raises:**

        `ValueError` from `SymbolicODE.validate`.
        """
        sym = cls._symbolic_template().validate()
        cls._compiled = CompiledExpressions.from_expressions(
            sym.rhs, sym.states, tuple(sym.params)
        )

    @classmethod
    @abc.abstractmethod
    def _symbolic_template(cls) -> SymbolicODE:
        """The vector field in cartesian coordinates as SymPy, with symbols only and
        no values."""

    def symbolic(self) -> SymbolicODE:
        """The cartesian template, for the AUTO code generator."""
        return self._symbolic_template()

    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " {self.dim}"], args: Any = None
    ) -> Float[Array, " {self.dim}"]:
        """The vector field at `u`, from the compiled template and the instance's
        current parameter values.

        **Arguments:**

        - `t`: Unused; may be a Python float.
        - `u`: The state, shape ``(dim,)``.
        - `args`: Unused.

        **Returns:**

        ``du/dt``, shape ``(dim,)``.
        """
        del t, args
        components = type(self)._compiled(self, *u)
        return jnp.stack(components, axis=0)

    def params(self) -> dict[str, Any]:
        """The parameters that the template names, as floats keyed by name, in the
        template's order."""
        return {
            name: float(getattr(self, name)) for name in type(self)._compiled.params
        }


__all__ = [
    "AbstractSymbolicODE",
    "CompiledExpressions",
    "EXTRA_FUNCS",
    "SymbolicODE",
]
