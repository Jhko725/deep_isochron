---
type: design
status: agreed
updated: 2026-10-02
verified_by: joon (review 2026-10-02)
sources: [Hong & Chun]
---

# `CubicBSpline` — the mathematics behind the code

Companion to [ADR-0006](../decisions/0006-cubic-bspline-boundary-and-inverse.md), which
records *why* the boundary and inverse differ from Hong & Chun [1]. This note records
*what* the code computes: index conventions, the parametrisation, the boundary argument,
the piecewise power-basis form, and what the inverse does and does not guarantee. File:
`src/deep_isochron/model/invertible/splines/cubic.py`.

Notation: order `k = 4` (cubic), `K = num_bins`, range `[a, b] = xy_range`,
`B_{j,k}` the B-spline of order `k` with support `[t_j, t_{j+k})`.

## 1. Knots and coefficients

A cubic B-spline on `K` bins `[t_0, t_1), …, [t_{K−1}, t_K)` with `t_0 = a`, `t_K = b`
needs two extra knots beyond each end (the cubic pieces at the boundary reference them):

```
t_{-2} < t_{-1} < t_0 = a < t_1 < … < t_K = b < t_{K+1} < t_{K+2}        (K + 5 knots)
```

On bin `[t_i, t_{i+1})` the active basis functions are `B_{i−3}, …, B_i`, so the
coefficients run `α_{-3}, …, α_{K−1}` (`K + 3` of them). The supports of `B_{-3}` and
`B_{K−1}` begin at `t_{-3}` and end at `t_{K+3}`, which are *not* stored: the polynomial
piece of a cubic B-spline on the last (first) interval of its support does not involve
the support's first (last) knot, so the `K + 5` knots above determine the spline on
`[a, b]` (this is why `test_bspline_matches_scipy` can pad scipy's knot vector with
arbitrary values). Arrays are 0-based:

| array                 | length  | entry `i`                        |
|-----------------------|---------|----------------------------------|
| `knots`               | `K + 5` | `t_{i−2}`                        |
| `knot_widths`         | `K + 4` | `t_{i−1} − t_{i−2}`              |
| `coeffs`              | `K + 3` | `α_{i−3}`                        |
| `_greville(knots)`    | `K + 3` | `ξ_{i−3} = (t_{i−2} + t_{i−1} + t_i) / 3` |
| `xs` = `knots[2:K+3]` | `K + 1` | `t_i` (the range knots)          |

So `coeffs[i]` and `_greville(knots)[i]` refer to the same basis function `B_{i−3,4}`,
whose Greville abscissa is `ξ_j = (t_{j+1} + t_{j+2} + t_{j+3}) / 3` (de Boor [2], Ch. IX).

## 2. Parametrisation (`constrain`)

`raw` has `num_params = (K + 4) + (K − 2) = 2K + 2` entries: a knot block and a
coefficient block, each passed through `Widths` (floored softmax; ADR-0005).

**Knots.** `w = Widths(1, min_rel_knot_width)(t_raw)` gives `K + 4` positive relative
widths summing to 1, each at least `min_rel_knot_width`. They are then rescaled so that
the **`K` interior widths** `w[2 : K+2]` (those between `t_0` and `t_K`) sum to the range
width `b − a`; the four exterior widths scale by the same factor. `_knots` accumulates,
shifts so that `t_0 = a` exactly, and snaps `t_K = b` exactly (so the range endpoints are
knots regardless of rounding, as nflows does).

*Deviation from [1].* Lines 1–3 of Algorithm 1 normalise the whole extended width vector
(all `s − r + 2k − 4` widths, in the paper's indexing) to sum to 1 and then place the
knots so that `t_r = 0`. Read literally, that leaves the position of `t_s` depending on
the exterior widths; the paper's lines 13–17 then compensate by rescaling the
*coefficients* so that `f(1) = 1`. Normalising only the interior widths removes the need:
`t_K = b` by construction. (This reading of lines 1–3 is deduced from the algorithm as
summarised in [1]; the deviation itself is a design choice, not a correction.)

**Coefficients.** The pinned ends are `α_{-3}, α_{-2}, α_{-1} = ξ_{-3}, ξ_{-2}, ξ_{-1}`
and `α_{K−3}, α_{K−2}, α_{K−1} = ξ_{K−3}, ξ_{K−2}, ξ_{K−1}`. The `K − 3` free
coefficients `α_0, …, α_{K−4}` are written as cumulative sums of `K − 2` positive
increments `Δ_0, …, Δ_{K−3}` starting at `α_{-1}` with `Σ Δ = ξ_{K−3} − ξ_{-1}` (the
"Greville span"), via `Widths(span, min_rel_coeff_incr)`: the last increment is the gap
up to the pinned `α_{K−3}`, so the whole sequence `α_{-3} < … < α_{K−1}` is strictly
increasing (Greville abscissae are strictly increasing because knots are).

**Identity at `raw = 0`.** Zero raw gives equal widths (uniform knots) and equal
increments; with uniform knots the Greville abscissae are equally spaced, so equal
increments from `ξ_{-1}` to `ξ_{K−3}` land exactly on `ξ_0, …, ξ_{K−4}`. All coefficients
equal their Greville abscissae and, by §3, `f = id` on the range. This is what makes the
class a valid coupling template (ADR-0002).

## 3. Monotonicity and the `C²` boundary

**Monotonicity.** The derivative of a spline is a spline of one order lower (de Boor [2],
(X.12); stated as Theorem 1 of [1] for this setting):

```
f'(x) = Σ_j 3 · (α_j − α_{j−1}) / (t_{j+3} − t_j) · B_{j,3}(x)      on [t_0, t_K].
```

With strictly increasing coefficients and positive knot spacings every term is positive
and the `B_{j,3}` are a non-negative partition of unity, so `f' > 0`. The floors give a
positive lower bound (`l < f' < u` in the notation of [1]); the map is bi-Lipschitz on the
range.

**Linear precision.** Marsden's identity ([2], Ch. IX) specialises to
`Σ_j ξ_j B_{j,4}(x) = x` for all `x` in the knot span. Hence wherever only basis functions
with `α_j = ξ_j` are active, `f(x) = x`.

**The `C²` join.** At `x = a = t_0` from the inside, the basis functions whose supports
contain `t_0` in their interior are `B_{-3}, B_{-2}, B_{-1}` (supports `[t_{-3}, t_1)`,
…, `[t_{-1}, t_3)`). `B_{0,4}` has support `[t_0, t_4)` and, being `C²` at the simple
knot `t_0` with zero to its left, satisfies `B_0(t_0) = B_0'(t_0) = B_0''(t_0) = 0`. So
`f`, `f'`, `f''` at `t_0` depend only on the three pinned coefficients, and since those
equal their Greville abscissae the identity's values are reproduced: `f(a) = a`, `f'(a) =
1`, `f''(a) = 0`. The same argument with `B_{K−4}` (support `[t_{K−4}, t_K)`, vanishing
with two derivatives at `t_K` from the left) handles `x = b`. Pinning a fourth coefficient
per side would make the edge bins exactly the identity; pinning only two would leave
`f''` free. Pinned by the tests `test_bspline_identity_join_regularity`,
`test_join_regularity` and `test_bspline_is_c2_at_interior_knots` in `tests/test_splines.py`.

## 4. Power-basis pieces (`pieces`)

For evaluation and inversion each bin's cubic is held in the local coordinate
`u = (x − t_j) / (t_{j+1} − t_j) ∈ [0, 1]` as `p_j(u) = c_0 + c_1 u + c_2 u² + c_3 u³`
(Horner form in `_cubic`/`_dcubic`). The coefficients are the Taylor expansion at `t_j`,
obtained from the derivative recursion applied twice (first and second divided
differences `D`, `E` of the coefficients over the appropriate knot spans) and from the
fact that `f'''` is constant on a bin. Rather than reproduce the algebra here, the code
is pinned against an independent evaluation: `test_bspline_matches_scipy` compares `f`
with `scipy.interpolate.BSpline(knots, coeffs, 3)` on random parameters. `ys`
(`= p_j(0)` plus the endpoint `b`) and `knot_derivs` (`= c_1 / (t_{j+1} − t_j)`, plus `1`
at `b`) are read off the same coefficients.

## 5. The inverse (`monotone_cubic_root`)

Given `y` in the range, the bin is `k` with `ys[k] ≤ y < ys[k+1]` (`get_bin_and_offset`),
and the root `u ∈ [0, 1]` of `p_k(u) = y` is found by `newton_iters` iterations of

```
r      = p(u) − y
[lo,hi] ← [u, hi] if r < 0 else [lo, u]          # the root stays bracketed
u_N    = u − r / p'(u)                             # Newton
u      ← u_N if lo ≤ u_N ≤ hi else (lo + hi) / 2   # else bisect
```

from the secant initial guess `u_0 = clip((y − c_0) / (c_1 + c_2 + c_3), 0, 1)`.

**Invariants (hold for every parameter value and every iteration count).**
The root is in `[lo, hi]` after every update, and the iterate is in `[0, 1]`; therefore
the inverse is always finite and lands in the correct bin. This is what the closed-form
root cannot promise (ADR-0006).

**Convergence.** `p' ≥ l > 0` on `[0, 1]` (§3), so the Newton step `r / p'(u)` vanishes
only at the root; every accepted Newton step strictly shrinks the bracket; the iterates
converge to the root, quadratically once `|u − u*|` is below `~ min p' / max |p''|`.

**Error bound.** A bisection step halves the bracket; an accepted Newton step shrinks it
by a data-dependent factor that can be less than one half. So the uniform bound
`|u − u*| ≤ 2^{−newton_iters}` holds for the *bisection fall-back alone*, and is **not**
a worst case for the mixed iteration. In practice (and in the suite: round trips to
`1e-8` and finite second derivatives of the inverse at `|raw| ≤ 30`,
`test_bspline_inverse_converges_for_extreme_params`,
`test_bspline_inverse_second_derivative`) 20 iterations reach round-off. If a hard bound
is ever needed, forcing a bisection every other iteration gives `2^{−newton_iters/2}` at
the price of slower typical convergence; it has not been needed.

**Derivatives.** `monotone_cubic_root` is a `jax.custom_jvp` with the implicit-function
rule `du = (dy − Σ_i dc_i u^i) / p'(u)` (differentiating `p(u; c) = y`). The rule evaluates
`u` by calling the root function itself, so higher-order derivatives apply the same rule
recursively; nothing differentiates through the `fori_loop`.

## References

[1] S. Hong and S. Y. Chun. Neural Diffeomorphic Non-uniform B-spline Flows. AAAI (2023).
    arXiv:2304.04555. — Algorithm 1 (parametrisation), Theorem 1 (monotonicity and the
    derivative formula), Discussion (root-formula instability and the 1-in-10 outliers).
[2] C. de Boor. A Practical Guide to Splines, revised ed. Springer (2001). — Ch. IX
    (Marsden's identity, Greville abscissae), Ch. X (derivative of a spline, eq. (X.12)).
