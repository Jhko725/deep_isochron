r"""Bijection interfaces.

``AbstractBijection``
    A diffeomorphism of $\mathbb{R}^{\text{dim}}$: ``__call__``, ``inverse``,
    ``jacobian``, plus two declared properties — ``dim`` and ``smoothness`` (the map is
    $C^k$; ``None`` means $C^\infty$).

``AbstractScalarBijection``
    $\mathbb{R}\to\mathbb{R}$ maps parametrized by a single *unconstrained* vector
    ``raw``, which is the only trainable leaf. Constrained parameters are computed on
    read by ``constrain(raw)`` (see ``params``); ``constrain(0)`` is the identity map.
    This is what makes a standalone scalar bijection safe to optimize and what lets
    ``CouplingFlow`` write a conditioner's output straight into ``raw``.

``ScalarChain`` / ``SequentialINN``
    Compositions of scalar / vector bijections.

Implementing an ``eqx.AbstractVar`` (``dim``, ``smoothness``, ``num_params``, ``raw``):
use a static ``init=False`` field — ``eqx.field(static=True, default=..., init=False)``
when the value is fixed by the class, assigned in ``__init__`` when it is derived. A
bare class attribute or a ``ClassVar`` is rejected by type checkers as an incompatible
override; a property is too. See ``docs/decisions/0004-abstractvar-fields.md``.
"""

import abc
import itertools
from collections.abc import Sequence
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def min_smoothness(values: Sequence[int | None]) -> int | None:
    """Minimum regularity of a composition: ``None`` (C^∞) only if all members are."""
    finite = [v for v in values if v is not None]
    return min(finite) if finite else None


class AbstractBijection(eqx.Module):
    dim: eqx.AbstractVar[int]
    smoothness: eqx.AbstractVar[int | None]
    """The map is C^k with k = smoothness; None means C^∞."""

    @abc.abstractmethod
    def __call__(
        self, x: Float[Array, " {self.dim}"]
    ) -> Float[Array, " {self.dim}"]: ...

    @abc.abstractmethod
    def inverse(
        self, y: Float[Array, " {self.dim}"]
    ) -> Float[Array, " {self.dim}"]: ...

    def jacobian(
        self, x: Float[Array, " {self.dim}"]
    ) -> Float[Array, " {self.dim} {self.dim}"]:
        return eqx.filter_jacfwd(self)(x)

    @property
    def num_trainable_params(self) -> int:
        """Total size of inexact-array leaves. Equal to the number of trainable
        parameters."""
        return sum(
            leaf.size
            for leaf in jax.tree.leaves(eqx.filter(self, eqx.is_inexact_array))
        )


B = TypeVar("B", bound="AbstractScalarBijection")


class AbstractScalarBijection[P: tuple](AbstractBijection):
    r"""Bijections $\mathbb{R}\rightarrow\mathbb{R}$ acting on 0-d arrays, parametrized
    by ``num_params`` unconstrained reals.

    **Implementation checklist.** A subclass is complete when it has exactly these
    parts, in this order (``Affine`` in ``affine.py`` is the minimal worked example;
    ``CubicBSpline`` the largest):

    1. ``class FooParams(NamedTuple)`` of the *constrained* parameters, and
       ``class Foo(AbstractScalarBijection[FooParams])``.
    2. Fields: ``raw: Float[Array, " n"] | None`` — the **only** array leaf — then
       static configuration as ``eqx.field(static=True)``, then ``num_params`` and
       ``smoothness`` as ``eqx.field(static=True, default=..., init=False)`` (or
       ``num_params`` as a property when it depends on the configuration, as the
       splines do). Nothing else is an array: no cached constrained values, no
       constraint objects (ADR-0005).
    3. ``__init__(self, <static config>, *, raw=None, ...)``: validate and store the
       configuration, then ``self.raw = self._init_raw(jnp.zeros(n) if raw is None
       else raw)``. So ``Foo(config)`` is the identity map and ``Foo(config,
       raw=...)`` is any other member of the family; templates (``raw=None``) are
       produced by ``CouplingFlow``, not by the constructor.
    4. ``constrain(self, raw) -> FooParams``: the single place where unconstrained
       values become constrained ones, built from ``constraints.*`` primitives so that
       ``constrain(zeros)`` is the identity map's parameters (ADR-0001, ADR-0002).
    5. ``__call__``/``inverse``: ``p = self.params`` and compute from ``p`` only.
    6. Optionally ``@classmethod from_constrained(cls, <params>, *, <config>)``,
       inverting the same primitives, for construction from constrained values.

    Everything else is final and inherited: ``params`` (``constrain(raw)``, raising on a
    template), ``jacobian`` (``jax.grad``), ``from_unconstrained`` (``tree_at`` on
    ``raw``), ``identity_like``. Register the class in ``tests/registry.py`` as
    ``"foo": Foo(config)``; every law in ``tests/test_bijections.py`` then applies with
    no per-class test code, and ``tests/test_registry.py`` fails until the entry exists.

    Why the shape: ``raw`` being the only leaf is what makes a conditioner's output
    writable straight into the bijection (``from_unconstrained`` is one ``tree_at``) and
    what makes any optimizer step safe (no step can leave the constrained set). An
    instance with ``raw=None`` is a *template*: static configuration only, hashable,
    zero trainable size — what ``CouplingFlow`` stores.
    """

    dim: int = eqx.field(static=True, default=1, init=False)
    raw: eqx.AbstractVar[Float[Array, " num_params"] | None]
    num_params: eqx.AbstractVar[int]

    @abc.abstractmethod
    def constrain(self, raw: Float[Array, " {self.num_params}"]) -> P:
        """Unconstrained vector -> constrained parameters. ``constrain(0)`` is the
        identity map's parameters."""

    @property
    def params(self) -> P:
        """Constrained parameters of this instance (requires ``raw`` to be set)."""
        if self.raw is None:
            raise ValueError(
                f"{type(self).__name__} is a template (raw=None); call "
                "from_unconstrained(raw) or identity_like() to get a usable instance."
            )
        return self.constrain(self.raw)

    def jacobian(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """Scalar bijections act on 0-d arrays, so the Jacobian is ``f'(x)``."""
        return jax.grad(self)(x)

    def from_unconstrained(
        self: B, params_raw: Float[Array, " {self.num_params}"]
    ) -> B:
        """Copy of ``self`` (same static configuration) with ``raw = params_raw``.
        Final: identical for every subclass."""
        return eqx.tree_at(
            lambda m: m.raw, self, params_raw, is_leaf=lambda x: x is None
        )

    def identity_like(self: B) -> B:
        """Copy of ``self`` that is the identity map (``raw = 0``)."""
        return self.from_unconstrained(jnp.zeros((self.num_params,)))

    def _init_raw(self, raw: Float[Array, " {self.num_params}"] | None):
        """Helper for subclass ``__init__``: validate and store ``raw`` (``None`` keeps
        a template; omitted -> identity)."""
        if raw is not None:
            raw = jnp.asarray(raw)
            if raw.shape != (self.num_params,):
                raise ValueError(
                    f"raw must have shape ({self.num_params},), got {raw.shape}."
                )
        return raw


class ScalarChain(AbstractScalarBijection[tuple]):
    """Composition of scalar bijections, applied left to right. Itself a scalar
    bijection parametrized by the concatenation of its members' raw vectors, so it is a
    valid coupling template.

    The chain's ``raw`` is ``None`` by construction; members hold their own ``raw``
    (``None`` for a template chain). ``from_unconstrained`` distributes the chunks to
    the members, so ``params`` is the tuple of member ``params``.
    """

    members: tuple[AbstractScalarBijection, ...]
    raw: None = eqx.field(static=True, default=None, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)

    def __init__(self, members: Sequence[AbstractScalarBijection]):
        members = tuple(members)
        if not members:
            raise ValueError("ScalarChain needs at least one member.")
        for m in members:
            if not isinstance(m, AbstractScalarBijection):
                raise TypeError(
                    f"{type(m).__name__} is not an AbstractScalarBijection."
                )
        self.members = members
        self.smoothness = min_smoothness([m.smoothness for m in members])

    @property
    def num_params(self) -> int:
        return sum(m.num_params for m in self.members)

    def _chunks(self, raw):
        sizes = [m.num_params for m in self.members]
        return jnp.split(raw, list(itertools.accumulate(sizes))[:-1])

    def constrain(self, raw):
        return tuple(m.constrain(c) for m, c in zip(self.members, self._chunks(raw)))

    @property
    def params(self):
        return tuple(m.params for m in self.members)

    def from_unconstrained(self, params_raw):
        members = tuple(
            m.from_unconstrained(c)
            for m, c in zip(self.members, self._chunks(params_raw))
        )
        return eqx.tree_at(lambda s: s.members, self, members)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        for m in self.members:
            x = m(x)
        return x

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        for m in self.members[::-1]:
            y = m.inverse(y)
        return y


class SequentialINN(AbstractBijection):
    """Composition of vector bijections of equal ``dim``, applied left to right."""

    transforms: tuple[AbstractBijection, ...]
    dim: int = eqx.field(static=True, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)

    def __init__(self, transforms: Sequence[AbstractBijection]):
        transforms = tuple(transforms)
        if not transforms:
            raise ValueError("SequentialINN needs at least one transform.")
        dims = {t.dim for t in transforms}
        if len(dims) != 1:
            raise ValueError(
                "Each element of transforms must have the same dimensionality."
            )
        self.transforms = transforms
        self.dim = dims.pop()
        self.smoothness = min_smoothness([t.smoothness for t in transforms])

    def __call__(self, x: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        for T in self.transforms:
            x = T(x)
        return x

    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        for T in self.transforms[::-1]:
            y = T.inverse(y)
        return y


__all__ = [
    "AbstractBijection",
    "AbstractScalarBijection",
    "ScalarChain",
    "SequentialINN",
    "min_smoothness",
]
