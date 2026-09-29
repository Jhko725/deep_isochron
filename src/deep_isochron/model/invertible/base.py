import abc
from collections.abc import Sequence
from typing import Self

import equinox as eqx
import jax
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


class AbstractScalarBijection(AbstractBijection):
    """Bijections mapping $\mathbb{R}\rightarrow\mathbb{R}$, parametrized by
    `num_params` parameters.

    The parameters can be constrained. Initialization from unconstrained parameters is
    done by `self.from_unconstrained`..
    """

    dim = 1
    num_params: eqx.AbstractClassVar[int]

    @classmethod
    @abc.abstractmethod
    def from_unconstrained(cls, params_raw: Float[Array, " {cls.num_params}"]) -> Self:
        """Instantiate the class from unconstrained parameter values.

        This is typically done by first mapping the values to the constrained set, then
        instantiating the class."""


# class CouplingTransformBase(AbstractBijection):

# TODO: could create a CouplingTransformBase class


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

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        for T in self.transforms:
            x = T(x)
        return x

    def inverse(self, y: Float[Array, "dim"]) -> Float[Array, "dim"]:
        for T in self.transforms[::-1]:
            y = T.inverse(y)
        return y
