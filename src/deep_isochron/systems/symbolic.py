r"""Vector fields declared once, as SymPy expressions, with JAX and Fortran derived.

An observed system such as FitzHugh–Nagumo writes its right-hand side in the classmethod
``_symbolic_template`` of an ``AbstractSymbolicODE``; the JAX ``rhs`` (through
``sympy2jax``), ``params()``, and, in ``analysis.ode.auto``, the AUTO-07p Fortran
``RHS`` with its exact Jacobian are derived from that one definition. The normal forms
declare their rates $\rho(r)$ and $\omega(r)$ the same way (``normal_forms.base``) and
expose the same cartesian ``symbolic()`` view, so the code generator sees one protocol,
``SupportsSymbolic``.

The expressions are compiled once per concrete class, in ``__init_subclass__``, and
stored as a class attribute, so a malformed template fails when the class is created
and only the parameter values are read per call. Under ``jit`` the jaxpr is the bare
arithmetic of the expressions. A template must name each parameter symbol after the
attribute that holds its value, list ``states`` in the order of the state vector, and
write numeric literals as plain constants; ``SymbolicODE.validate`` checks the first
two.

ADR-0016 has the reasoning, the rejected alternatives and the ADR-0004 edge.
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
"""SymPy functions that templates may use and ``sympy2jax`` does not translate on its
own. The Fortran printer knows neither, so a system that needs them, such as
Hodgkin–Huxley, cannot be written for AUTO."""


class SymbolicODE(NamedTuple):
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


@runtime_checkable
class SupportsSymbolic(Protocol):
    """What the AUTO code generator needs of a system: its cartesian ``symbolic()``
    view, and the attributes that the view's ``params`` names."""

    def symbolic(self) -> SymbolicODE:
        """The cartesian vector field as SymPy."""
        ...


class CompiledExpressions(NamedTuple):
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


def compile_expressions(
    exprs: tuple[sp.Expr, ...], inputs: tuple[sp.Symbol, ...], params: tuple[str, ...]
) -> CompiledExpressions:
    """Compile `exprs` to a `CompiledExpressions`; numeric literals stay constants.

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
    return CompiledExpressions(module, tuple(s.name for s in inputs), tuple(params))


def is_abstract(attr: Any) -> bool:
    """Whether `attr` (a function, classmethod or staticmethod) is abstract."""
    func = getattr(attr, "__func__", attr)  # unwrap classmethod/staticmethod
    return bool(getattr(func, "__isabstractmethod__", False))


def defines_concretely(cls: type, name: str) -> bool:
    """Whether `cls` itself, not a base, defines a non-abstract attribute `name`.

    This is the guard for building per-class machinery in `__init_subclass__`.
    `inspect.isabstract(cls)` is not usable there, because `ABCMeta` computes
    `__abstractmethods__` only after `type.__new__`, and so `__init_subclass__`, has
    run.
    """
    attr = cls.__dict__.get(name)
    return attr is not None and not is_abstract(attr)


class AbstractSymbolicODE(AbstractODE):
    """An ODE whose vector field is declared by `_symbolic_template`, with `rhs`,
    `symbolic()` and `params()` derived.

    A subclass supplies the classmethod `_symbolic_template` and the attributes it
    names: fields, or properties computing constrained values from raw leaves. The
    compiled form is built once when the class is created; an intermediate abstract
    subclass that does not define the template builds nothing.

    **Attributes:**

    - `_compiled`: The compiled expressions, set on each concrete subclass by
        `__init_subclass__`. A class attribute rather than a field: it implements no
        `AbstractVar`, so ADR-0004 does not apply (ADR-0016 §3).
    """

    _compiled: ClassVar[CompiledExpressions]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not defines_concretely(cls, "_symbolic_template"):
            return
        sym = cls._symbolic_template().validate()
        cls._compiled = compile_expressions(sym.rhs, sym.states, tuple(sym.params))

    @classmethod
    @abc.abstractmethod
    def _symbolic_template(cls) -> SymbolicODE:
        """The vector field as SymPy, with symbols only and no values."""

    def symbolic(self) -> SymbolicODE:
        """The cartesian symbolic view, for the code generator (`SupportsSymbolic`)."""
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
        """The parameters that the template names, as floats keyed by name."""
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
