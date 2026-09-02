import abc
from collections.abc import Sequence

import equinox as eqx
from jaxtyping import Array, Float


class AbstractInvertibleTransform(eqx.Module):
    dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(self, x): ...

    @abc.abstractmethod
    def inverse(self, y): ...


# class CouplingTransformBase(AbstractInvertibleTransform):

# TODO: could create a CouplingTransformBase class


class SequentialINN(AbstractInvertibleTransform):
    transforms: tuple[AbstractInvertibleTransform, ...]

    def __init__(self, transforms: Sequence[AbstractInvertibleTransform]):
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
