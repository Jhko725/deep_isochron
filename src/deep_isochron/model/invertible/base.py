r"""The bijection interfaces.

``AbstractBijection`` is a diffeomorphism of $\mathbb{R}^{\text{dim}}$ with
``__call__``, ``inverse`` and ``jacobian``, and two declared properties, ``dim`` and
``smoothness``. ``AbstractScalarBijection`` is a map $\mathbb{R}\to\mathbb{R}$
parametrized by a single unconstrained vector ``raw``, the only trainable leaf, whose
constrained parameters are computed on read by ``constrain(raw)``, with
``constrain(0)`` the identity map; that shape is what makes a standalone scalar
bijection safe to optimize and lets ``CouplingFlow`` write a conditioner's output
straight into ``raw`` (ADR-0001, ADR-0002). ``ScalarChain`` and ``SequentialINN``
compose scalar and vector bijections.

An ``eqx.AbstractVar`` such as ``dim``, ``smoothness``, ``num_params`` or ``raw`` is
implemented as a static ``init=False`` field, with ``default=`` when the class fixes
the value and assigned in ``__init__`` when it is derived; a bare class attribute, a
``ClassVar`` or a property is rejected by type checkers as an incompatible override
(ADR-0004).
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
    r"""The regularity of a composition: the minimum of `values`, with ``None``, that
    is $C^\infty$, only if every member is ``None``."""
    finite = [v for v in values if v is not None]
    return min(finite) if finite else None


class AbstractBijection(eqx.Module):
    r"""A diffeomorphism of $\mathbb{R}^{\text{dim}}$.

    **Attributes:**

    - `dim`: The dimension.
    - `smoothness`: The map is $C^k$ with ``k = smoothness``; ``None`` means
        $C^\infty$ (ADR-0003).
    """

    dim: eqx.AbstractVar[int]
    smoothness: eqx.AbstractVar[int | None]

    @abc.abstractmethod
    def __call__(self, x: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        """The image of `x`."""

    @abc.abstractmethod
    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        """The preimage of `y`."""

    def jacobian(
        self, x: Float[Array, " {self.dim}"]
    ) -> Float[Array, " {self.dim} {self.dim}"]:
        """The Jacobian at `x`, by forward-mode autodiff."""
        return eqx.filter_jacfwd(self)(x)

    @property
    def num_trainable_params(self) -> int:
        """The total size of the inexact-array leaves: the number of trainable
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
       ``smoothness`` as ``eqx.field(static=True, default=..., init=False)`` —
       assigned in ``__init__`` (no ``default=``) when it depends on the
       configuration, as the splines do; never a property (ADR-0004). Nothing else
       is an array: no cached constrained values, no constraint objects (ADR-0005).
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
        """The derivative ``f'(x)``, since a scalar bijection acts on 0-d arrays."""
        return jax.grad(self)(x)

    def from_unconstrained(
        self: B, params_raw: Float[Array, " {self.num_params}"]
    ) -> B:
        """A copy of `self`, with the same static configuration, whose ``raw`` is
        `params_raw`. Final: identical for every subclass."""
        return eqx.tree_at(
            lambda m: m.raw, self, params_raw, is_leaf=lambda x: x is None
        )

    def identity_like(self: B) -> B:
        """A copy of `self` that is the identity map, with ``raw = 0``."""
        return self.from_unconstrained(jnp.zeros((self.num_params,)))

    def _init_raw(self, raw: Float[Array, " {self.num_params}"] | None):
        """`raw` validated for a subclass ``__init__``: ``None`` keeps a template.

        **Raises:** `ValueError` for a shape other than ``(num_params,)``.
        """
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
    raw: Float[Array, " num_params"] | None = eqx.field(
        static=True, default=None, init=False
    )
    num_params: int = eqx.field(static=True, init=False)
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
        self.num_params = sum(m.num_params for m in members)
        self.smoothness = min_smoothness([m.smoothness for m in members])

    def _chunks(self, raw):
        """`raw` split into one chunk per member."""
        sizes = [m.num_params for m in self.members]
        return jnp.split(raw, list(itertools.accumulate(sizes))[:-1])

    def constrain(self, raw):
        """The tuple of the members' constrained parameters."""
        return tuple(m.constrain(c) for m, c in zip(self.members, self._chunks(raw)))

    @property
    def params(self):
        """The tuple of the members' `params`."""
        return tuple(m.params for m in self.members)

    def from_unconstrained(self, params_raw):
        """A copy whose members received their chunks of `params_raw`."""
        members = tuple(
            m.from_unconstrained(c)
            for m, c in zip(self.members, self._chunks(params_raw))
        )
        return eqx.tree_at(lambda s: s.members, self, members)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """The members applied left to right."""
        for m in self.members:
            x = m(x)
        return x

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """The members' inverses applied right to left."""
        for m in self.members[::-1]:
            y = m.inverse(y)
        return y


class SequentialINN(AbstractBijection):
    """The composition of vector bijections of one `dim`, applied left to right.

    **Raises:** `ValueError` for no transforms, or transforms of different
    dimensions.
    """

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
        """The transforms applied left to right."""
        for T in self.transforms:
            x = T(x)
        return x

    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        """The transforms' inverses applied right to left."""
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
