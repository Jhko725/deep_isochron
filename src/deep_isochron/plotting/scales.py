"""Color scale utilities for any figure: a color range clipped at a quantile of the
values, so that a few extreme entries do not wash out the rest.

`clipped_norm` pools the values of every panel that shares one colorbar and returns a
clipped `matplotlib.colors.Normalize`, either symmetric about 0 or between a low and a
high quantile.
"""

from collections.abc import Sequence

import jax
import numpy as np
from jaxtyping import Array, Float
from matplotlib.colors import Normalize


def clipped_norm(
    values: Sequence[Float[np.ndarray, "..."] | Float[Array, "..."]]
    | Float[np.ndarray, "..."]
    | Float[Array, "..."],
    quantile: float,
    symmetric: bool,
) -> Normalize:
    """Build a clipped color range from the `quantile` of the pooled `values`.

    The entries of every array are pooled, and exact zeros are left out, so that an
    identity stage neither narrows a symmetric range nor widens a sequential one.

    **Arguments:**

    - `values`: One array, or a sequence of arrays such as one per panel, NumPy or
        JAX, of any shapes.
    - `quantile`: The quantile of the pooled values at the ends of the range.
    - `symmetric`: Whether the range is `[-m, m]`, with `m` the `quantile` of the
        magnitudes; otherwise it runs from the `1 - quantile` to the `quantile` of the
        values.

    **Returns:**

    A `Normalize` with `clip=True`. If no non-zero entry remains, the range is
    `(-1, 1)` when `symmetric` and `(0, 1)` otherwise.

    **Raises:**

    `ValueError` if `quantile` is not in `(0.5, 1]`.
    """
    if not 0.5 < quantile <= 1.0:
        raise ValueError(f"quantile must lie in (0.5, 1], got {quantile}.")
    arrays = [values] if isinstance(values, np.ndarray | jax.Array) else values
    pool = np.concatenate([np.ravel(np.asarray(v)) for v in arrays])
    pool = pool[pool != 0.0]
    if symmetric:
        m = float(np.quantile(np.abs(pool), quantile)) if pool.size else 1.0
        return Normalize(-m, m, clip=True)
    if not pool.size:
        return Normalize(0.0, 1.0, clip=True)
    lo, hi = (float(q) for q in np.quantile(pool, [1.0 - quantile, quantile]))
    return Normalize(lo, hi, clip=True)
