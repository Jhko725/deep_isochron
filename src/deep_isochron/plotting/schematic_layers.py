r"""The hand-set layers and the sheet of the slide-9 schematics: real invertible layers
of the model whose parameters are set by hand, so that each figure shows the picture it
is drawn for by design rather than by training.

`schematic_linear` builds the linear layer of `inn_stack` as a `BiLipschitzLinear`.
`schematic_affine_coupling` and `schematic_spline_coupling` build the two coupling
layers of `coupling_warp`, the spline one also drawn by `inn_stack`, as `CouplingFlow`s
whose conditioners are hand-set functions of $x_1$. Every one of these layers is the
identity at `raw = 0`, so the nonzero raw entries are all that the picture costs. The
affine conditioner is nonlinear in $x_1$, so the affine image grid curves as well, and
the contrast with the spline is the spacing of the lines within one column. The
spline's conditioner does not set raws one by one: it builds a desired knot layout from
$x_1$, a steep band whose center moves with $x_1$ and a pinched region fixed in $x_2$,
and maps the layout back to raws through the inverses of the spline's constraints, so
that the warp shows a step and a plateau and its image grid spaces the lines of one
column unevenly, which no affine layer can. `schematic_trajectories` builds
trajectories of the Bautin normal form fitted into a square box, and `schematic_sheet`
the regular grid on that box that carries them.

The design and its decisions are in `docs/changes/2026-11-04-inn-schematics.md`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..model.invertible import (
    Affine,
    BiLipschitzLinear,
    CouplingFlow,
    MonotonicRQSpline,
)
from ..model.invertible.constraints import GreaterThan, Widths
from ..systems.normal_forms import BautinNormalForm
from .deformation import Sheet, SheetConfig


def schematic_linear(
    angle_raw: float = 0.3, stretch_raw: float = 1.0, max_lipschitz: float = 3.0
) -> BiLipschitzLinear:
    r"""The linear layer of the figure: a rotation followed by a bounded stretch along
    one axis and a matching squeeze along the other.

    The layer is the identity `BiLipschitzLinear` of dimension 2 without bias, with
    `raw_U[1, 0]` set to `angle_raw` and `raw_s` to `(stretch_raw, -stretch_raw)`. So
    `U` is the rotation by $-2 \arctan$ `angle_raw`, `V` is the identity, and the
    singular values are the images of $\pm$ `stretch_raw` in $(1/L, L)$. Their
    product is 1 for the defaults, and $\det W > 0$ for every value.

    **Arguments:**

    - `angle_raw`: The raw generator of the rotation `U`.
    - `stretch_raw`: The raw of the first singular value; the second is its negative.
    - `max_lipschitz`: $L > 1$, the bound of the singular values.

    **Returns:**

    A `BiLipschitzLinear` of dimension 2 without bias.
    """
    identity = BiLipschitzLinear(
        2, max_lipschitz, use_bias=False, key=jax.random.PRNGKey(0)
    )
    raw_U = jnp.zeros((2, 2)).at[1, 0].set(angle_raw)
    raw_s = jnp.array([stretch_raw, -stretch_raw])
    return eqx.tree_at(lambda m: (m.raw_U, m.raw_s), identity, (raw_U, raw_s))


def schematic_affine_coupling(
    log_scale_offset: float = 0.3,
    scale_amplitude: float = 0.3,
    shift_curvature: float = 0.25,
) -> CouplingFlow[Affine]:
    r"""The affine coupling layer of the figure: $x_2 \mapsto a(x_1)\, x_2 + b(x_1)$,
    with a scale and a shift that are nonlinear in $x_1$.

    The conditioner returns the raw `Affine` parameters `(loc, log_scale)` $=
    (\kappa\, x_1^2,\ \lambda + A \sin \pi x_1)$, with $\lambda$ =
    `log_scale_offset`, $A$ = `scale_amplitude` and $\kappa$ = `shift_curvature`. So
    the shift is $b(x_1) = \kappa\, x_1^2$, and the scale is `Affine`'s soft clamp of
    the raw log-scale at its default bound 2, $a(x_1) = \exp(2 \tanh((\lambda + A \sin
    \pi x_1) / 2))$. At $x_1 = 0$ the layer is the pure scale $\exp(2 \tanh(\lambda /
    2))$, so the warp drawn at the middle column is a line through the origin, steeper
    than the identity for a positive $\lambda$.

    The conditioner is nonlinear in $x_1$ so that the image's lines of constant $x_2$
    curve and fan, as a trained affine layer's do. Within one column of constant $x_1$
    the layer is still affine, so the images of equally spaced lines stay equally
    spaced there; varying that spacing within a column is what the spline layer does
    and an affine layer cannot.

    **Arguments:**

    - `log_scale_offset`: The raw log-scale $\lambda$ at $x_1 = 0$.
    - `scale_amplitude`: The amplitude $A$ of the raw log-scale's sinusoid in $x_1$,
        nonnegative.
    - `shift_curvature`: The coefficient $\kappa$ of the quadratic shift.

    **Returns:**

    A `CouplingFlow` of dimension 2 that keeps $x_1$ and maps $x_2$.

    **Raises:**

    `ValueError` if `scale_amplitude` is negative.
    """
    if scale_amplitude < 0:
        raise ValueError(f"scale_amplitude must be nonnegative, got {scale_amplitude}.")

    def conditioner(x_const: Float[Array, " 1"]) -> Float[Array, " 2"]:
        """The raw `(loc, log_scale)` at `x_const`."""
        x1 = x_const[0]
        log_scale = log_scale_offset + scale_amplitude * jnp.sin(jnp.pi * x1)
        return jnp.stack([shift_curvature * x1**2, log_scale])

    return CouplingFlow(dim=2, bijection=Affine(), conditioner=conditioner)


def _allocate_gap_bins(
    lengths: Float[np.ndarray, " 3"], num_bins: int
) -> tuple[int, ...]:
    """Split `num_bins` bins, at least 3, over three gaps: one bin each, and the rest
    in proportion to the gaps' `lengths` by the largest remainder."""
    share = (num_bins - 3) * lengths / lengths.sum()
    extra = np.floor(share).astype(int)
    largest_remainders = np.argsort(extra - share)
    extra[largest_remainders[: num_bins - 3 - extra.sum()]] += 1
    return tuple(int(n) + 1 for n in extra)


def schematic_spline_coupling(
    num_bins: int = 9,
    xy_range: tuple[float, float] = (-1.0, 1.0),
    band_center: tuple[float, float] = (-0.5, 0.3),
    band_width: float = 0.25,
    band_span: float = 0.55,
    pinch_center: float = 0.7,
    pinch_width: float = 0.4,
    pinch_span: float = 0.04,
) -> CouplingFlow[MonotonicRQSpline]:
    r"""The spline coupling layer of the figure: a `MonotonicRQSpline` warp of $x_2$
    with a steep band that moves with $x_1$ and a pinched region fixed in $x_2$, so
    that the warp shows a step and a plateau.

    The conditioner builds the spline's knot layout from $x_1$ and returns its raw
    vector. The band is the interval of width `band_width` whose center moves linearly
    from `band_center[0]` at $x_1 = $ `lo` to `band_center[1]` at $x_1 = $ `hi`, held
    at those ends outside the range; its image takes the fraction `band_span` of the
    range, so the warp is steep there. The pinch is the interval of width
    `pinch_width` centered at `pinch_center`, and its image takes the fraction
    `pinch_span`, so the warp is nearly flat there. Each feature is one bin. The
    `num_bins - 2` other bins split the three gaps that the features leave, in
    proportion to the gaps' lengths at the middle $x_1$ and at least one bin each,
    and the gaps share the rest of the image at one common slope. Every interior knot
    derivative is the geometric mean of the slopes of its two bins, so the map is
    linear inside the gaps and joins each feature smoothly. The inverses of the
    template's `Widths` and `GreaterThan` map the layout back to raws, after the widths
    are clamped to twice the `Widths` floor.

    **Arguments:**

    - `num_bins`: The number of bins, at least 5.
    - `xy_range`: The range `(lo, hi)` of the spline; the identity outside it.
    - `band_center`: The center of the band in $x_2$ at $x_1 = $ `lo` and at `hi`.
    - `band_width`: The width of the band in $x_2$.
    - `band_span`: The fraction of the range that the band's image takes.
    - `pinch_center`: The center of the pinch in $x_2$.
    - `pinch_width`: The width of the pinch in $x_2$.
    - `pinch_span`: The fraction of the range that the pinch's image takes.

    **Returns:**

    A `CouplingFlow` of dimension 2 that keeps $x_1$ and maps $x_2$.

    **Raises:**

    `ValueError` naming the argument if `num_bins` is less than 5, a width or span is
    not positive, `band_span + pinch_span` is not below 1, the band at either end of
    its travel or the pinch leaves the open range, or the pinch meets the band
    anywhere along its travel.
    """
    lo, hi = xy_range
    if num_bins < 5:
        raise ValueError(f"num_bins must be at least 5, got {num_bins}.")
    for name, value in [
        ("band_width", band_width),
        ("band_span", band_span),
        ("pinch_width", pinch_width),
        ("pinch_span", pinch_span),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if band_span + pinch_span >= 1:
        raise ValueError(
            f"band_span + pinch_span must be below 1, got {band_span + pinch_span}."
        )
    band_lo = min(band_center) - 0.5 * band_width
    band_hi = max(band_center) + 0.5 * band_width
    pinch = (pinch_center - 0.5 * pinch_width, pinch_center + 0.5 * pinch_width)
    if not lo < band_lo < band_hi < hi:
        raise ValueError(
            f"band_center and band_width must keep the band inside {xy_range}, got "
            f"({band_lo}, {band_hi})."
        )
    if not lo < pinch[0] < pinch[1] < hi:
        raise ValueError(
            f"pinch_center and pinch_width must keep the pinch inside {xy_range}, got "
            f"{pinch}."
        )
    if not (band_hi < pinch[0] or pinch[1] < band_lo):
        raise ValueError(
            f"pinch_center must keep the pinch {pinch} clear of the band's travel "
            f"({band_lo}, {band_hi})."
        )
    band_first = band_hi < pinch[0]
    template = MonotonicRQSpline(num_bins, xy_range)
    width = template.range_width
    widths = Widths(width, template.min_rel_width)
    derivatives = GreaterThan(template.min_derivative, at_zero=1.0)
    floor = 2 * template.min_rel_width * width
    feature_spans = (band_span, pinch_span) if band_first else (pinch_span, band_span)
    gap_span = 1 - band_span - pinch_span

    def features(x1: Float[Array, ""]) -> Float[Array, "2 2"]:
        """The band's and the pinch's ends at `x1`, lower feature first."""
        t = jnp.clip((x1 - lo) / width, 0.0, 1.0)
        center = band_center[0] + (band_center[1] - band_center[0]) * t
        band = jnp.stack([center - 0.5 * band_width, center + 0.5 * band_width])
        both = [band, jnp.asarray(pinch)]
        return jnp.stack(both if band_first else both[::-1])

    def gaps(ends: Float[Array, "2 2"]) -> Float[Array, " 3"]:
        """The lengths of the three gaps the two features leave in the range."""
        return jnp.stack([ends[0, 0] - lo, ends[1, 0] - ends[0, 1], hi - ends[1, 1]])

    counts = _allocate_gap_bins(
        np.asarray(gaps(features(jnp.asarray(0.5 * (lo + hi))))), num_bins - 2
    )

    def conditioner(x_const: Float[Array, " 1"]) -> Float[Array, " raw"]:
        """The raw vector of the knot layout at `x_const`."""
        ends = features(x_const[0])
        lengths = gaps(ends)
        gap_slope = gap_span * width / jnp.sum(lengths)
        x_pieces, y_pieces = [], []
        for k, count in enumerate(counts):
            gap = jnp.full(count, lengths[k] / count)
            x_pieces.append(gap)
            y_pieces.append(gap_slope * gap)
            if k < 2:
                x_pieces.append((ends[k, 1] - ends[k, 0])[None])
                y_pieces.append(jnp.array([feature_spans[k] * width]))
        x_widths = jnp.maximum(jnp.concatenate(x_pieces), floor)
        y_widths = jnp.maximum(jnp.concatenate(y_pieces), floor)
        slopes = y_widths / x_widths
        knot_derivatives = jnp.sqrt(slopes[:-1] * slopes[1:])
        return jnp.concatenate(
            [
                widths.inverse(x_widths),
                widths.inverse(y_widths),
                derivatives.inverse(knot_derivatives),
            ]
        )

    return CouplingFlow(dim=2, bijection=template, conditioner=conditioner)


def schematic_trajectories(
    box: tuple[float, float], n: int = 4, samples: int = 200, scale: float = 0.7
) -> Float[np.ndarray, "n samples 2"]:
    r"""Trajectories of the talk's Bautin normal form, fitted into the square `box`,
    for the sheet of the figure.

    The system is `BautinNormalForm(a=1/3, b=0.5, w=2π, w0=2π − 2/3)`, the one of
    `ksiam_animations.ipynb`, whose limit cycle is the unit circle and whose period is
    1. The `n` trajectories start at radii equally spaced from 0.35 to 1.4, inside and
    outside the cycle, at equally spaced angles, and run for 1.2 periods. They are then
    scaled by `scale` times half the box's width and shifted to its center, so that
    the cycle is a circle of radius `scale` times that half-width. Since no trajectory
    leaves the radius 1.4, every sample lies inside the box when `scale` is below
    1 / 1.4.

    **Arguments:**

    - `box`: The box `(lo, hi)` on both axes.
    - `n`: The number of trajectories.
    - `samples`: The number of samples of each trajectory, equally spaced in time.
    - `scale`: The radius of the cycle as a fraction of the box's half-width.

    **Returns:**

    The trajectories in the box's coordinates, as NumPy.
    """
    lo, hi = box
    system = BautinNormalForm(a=1 / 3, b=0.5, w=2 * jnp.pi, w0=2 * jnp.pi - 2 / 3)
    radii = np.linspace(0.35, 1.4, n)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    u0s = jnp.asarray(radii[:, None] * np.stack([np.cos(angles), np.sin(angles)], -1))
    ts = jnp.linspace(0.0, 1.2 * system.period(), samples)
    ys = jax.vmap(lambda u0: system.flow(ts, u0).ys)(u0s)
    return 0.5 * (lo + hi) + scale * 0.5 * (hi - lo) * np.asarray(ys)


def schematic_sheet(
    box: tuple[float, float],
    trajectories: Float[np.ndarray, "t q 2"],
    cells: int = 16,
    line_samples: int = 200,
) -> Sheet:
    """The sheet of the figure: a regular grid of `cells` cells per axis on the square
    `box`, carrying `trajectories`.

    **Arguments:**

    - `box`: The box `(lo, hi)` on both axes.
    - `trajectories`: The trajectories on the sheet, such as `schematic_trajectories`.
    - `cells`: The number of grid cells along each axis, so `cells + 1` lines of each
        kind.
    - `line_samples`: The number of samples of each grid line.

    **Returns:**

    The `Sheet` on `(lo, lo, hi, hi)`.
    """
    lo, hi = box
    config = SheetConfig(cells=cells, line_samples=line_samples)
    return Sheet((lo, lo, hi, hi), trajectories, config=config)
