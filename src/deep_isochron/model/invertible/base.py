import abc
import itertools
from collections.abc import Sequence
from typing import ClassVar, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class AbstractBijection(eqx.Module):
    dim: eqx.AbstractVar[int]

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


class AbstractScalarBijection(AbstractBijection):
    r"""Bijections mapping $\mathbb{R}\rightarrow\mathbb{R}$, parametrized by
    `num_params` parameters.

    The parameters can be constrained. Initialization from unconstrained parameters is
    done by `self.from_unconstrained`..
    """

    dim: ClassVar[int] = 1  # ty:ignore
    num_params: eqx.AbstractVar[int]

    def jacobian(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """Scalar bijections act on 0-d arrays, so the Jacobian is ``f'(x)``."""
        return jax.grad(self)(x)

    @abc.abstractmethod
    def from_unconstrained(
        self: B, params_raw: Float[Array, " {self.num_params}"], **kwargs
    ) -> B:
        """Create a copy of self, with parameters set from the unconstrained parameter
        values.

        This is typically done by first mapping the values to the constrained set, then
        instantiating the class.
        The method is written so that from_unconstrained(jnp.zeros(self.num_params))
        returns the identity.

        This function is implemented as a plain method instead of a classmethod because
        certain bijections (ex. Spline variants) have instance-dependent num_params."""

    def identity_like(self: B) -> B:
        """Create a copy of self, with parameters set so that the resulting bijection is
        the identity.

        The base implementation assumes that `from_unconstrained` is written so that
        passing unconstrained parameters of zeros results in identity.
        """
        return self.from_unconstrained(jnp.zeros((self.num_params,)))


class SequentialINN(AbstractBijection):
    transforms: tuple[AbstractBijection, ...]

    def __init__(self, transforms: Sequence[AbstractBijection]):
        dims = set([t.dim for t in transforms])
        if len(dims) != 1:
            raise ValueError(
                "Each element of transforms must have the same dimensionality."
            )
        self.transforms = tuple(transforms)

    @property
    def dim(self) -> int:
        return self.transforms[0].dim

    # Shapes follow the members' (0-d for scalar bijections, (dim,) otherwise).
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        for T in self.transforms:
            x = T(x)
        return x

    def inverse(self, y: Float[Array, "..."]) -> Float[Array, "..."]:
        for T in self.transforms[::-1]:
            y = T.inverse(y)
        return y

    # A sequence of scalar bijections is itself a scalar bijection parametrised by
    # the concatenation of its members' unconstrained parameters, so it can serve
    # as a coupling template.
    @property
    def num_params(self) -> int:
        return sum(T.num_params for T in self.transforms)

    def from_unconstrained(
        self, params_raw: Float[Array, " {self.num_params}"], **kwargs
    ) -> "SequentialINN":
        sizes = [T.num_params for T in self.transforms]
        chunks = jnp.split(params_raw, list(itertools.accumulate(sizes))[:-1])
        return SequentialINN(
            [T.from_unconstrained(p, **kwargs) for T, p in zip(self.transforms, chunks)]
        )
