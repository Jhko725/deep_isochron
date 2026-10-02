# ADR-0006 — `CubicBSpline`: Greville-pinned boundary and a bracketed Newton inverse

**Status**: accepted (2026-10-02; records decisions taken on `splines-refactor`, 2026-09-30)

## Context

`CubicBSpline` ports the monotone non-uniform cubic B-spline flow of Hong & Chun [1] into
the `AbstractSpline` frame: a strictly increasing map that is a spline on `xy_range = [a, b]`
and the identity outside it. Two places in that port deviate from the reference algorithm,
and the reasons were recorded only in the branch's change document. This ADR makes them
decisions. The mathematics (index conventions, the derivative recursion, the normalisation,
the inverse's guarantees) is in `docs/design/cubic-bspline.md`.

Two requirements drive both deviations:

1. **The tails must join `C²`.** The spline's declared `smoothness` is 2 (ADR-0003), which
   means the whole map `R → R` — spline inside the range, identity outside — is `C²`. At
   `x = a` and `x = b` this needs `f = id`, `f' = 1`, `f'' = 0` from the inside.
2. **The inverse must be finite for every parameter value and every `y` in the range.** The
   coupling layer evaluates `inverse` on whatever the conditioner produces, under `vmap`
   and `grad`; a NaN at one point poisons the batch.

## Decision 1 — boundary by Greville pinning, not by affine rescaling

Algorithm 1 of [1] (its lines 13–17) enforces `f(0) = 0`, `f(1) = 1` by computing the
spline's provisional values at the two range endpoints and rescaling *all* coefficients
affinely so the endpoint values land on `0` and `1`. That fixes the function values only;
`f'` and `f''` at the endpoints are whatever the free coefficients make them, which is
enough for a flow on the fixed domain `[0, 1]` (the paper's setting) but not for a map that
continues as the identity outside the range.

Here the **three outermost coefficients on each side are pinned to their Greville
abscissae** `ξ_j = (t_{j+1} + t_{j+2} + t_{j+3}) / 3`. By the linear-precision case of
Marsden's identity (de Boor [2], Ch. IX), `Σ_j ξ_j B_{j,4}(x) = x`, so on any stretch where
the coefficients equal the Greville abscissae the spline *is* the identity. Only the three
pinned basis functions on each side are non-zero — together with their first two
derivatives — at the range endpoint, so `f(a) = a`, `f'(a) = 1`, `f''(a) = 0` (and likewise
at `b`) hold for every value of the free coefficients. The derivation is in the design note.

- **Rejected: lines 13–17 as written.** Gives `C⁰` tails; the declared `C²` would be false.
- **Rejected: pinning four coefficients per side.** Would make the two edge bins exactly
  the identity, losing a bin of flexibility on each side for no extra regularity (three
  already give `C²`, and a cubic B-spline is at most `C²` at a simple knot).
- **Rejected: the "naïve" solution [1] mentions** — coefficients `0` for `t_j < 0` and `1`
  for `t_{j+k} > 1` — which the authors themselves reject as severely reducing expressive
  power near the endpoints; it also does not give `f' = 1` there.

Cost: `K − 3` free coefficients instead of `K + 1`, i.e. reduced flexibility in the two
edge bins; `num_params = 2K + 2` rather than the paper's count. Coefficients live in real
units (no normalise/denormalise to `[0, 1]`), so the pinned values are the Greville
abscissae themselves and the free ones are strictly increasing increments between the
pinned ends (`Widths` with `total` = the Greville span).

## Decision 2 — inverse by bisection-safeguarded Newton with an implicit JVP, not the closed-form cubic root

On each bin the spline is a strictly increasing cubic in the local coordinate `u ∈ [0, 1]`,
so `f⁻¹` is a cubic root. The reference implementation uses the root formula of Peters
(2016), a modification of Blinn (2007) (as cited in [1]); the authors report that "about once
in ten times" training produces an outlier with large reverse KL and offer, as "one possible
explanation", "the numerical instability of the cubic equation root-finding formula", noting
that in floating point "it is impossible to guarantee whether the root-finding formula will
actually find the root correctly beyond machine precision" [1, Discussion].

Here the root is found by **`newton_iters` fixed iterations of Newton's method safeguarded
by bisection** on the bracket `[0, 1]`: the residual's sign at the current iterate shrinks
the bracket, the Newton step is taken if it lands inside the bracket and the midpoint
otherwise. The derivative is supplied by a `jax.custom_jvp` rule from the implicit function
theorem, `du = (dy − ∂_c p(u; c) · dc) / p'(u; c)`, which calls the root function itself and
is therefore differentiable to any order (the test suite differentiates the inverse twice).

- **Rejected: closed-form root.** Needs the quadratic/linear fall-backs for near-degenerate
  leading coefficients (`c₃ ≈ 0`, `c₂ ≈ 0` — the thresholds in the reference are `1e-7`),
  is the authors' own suspect for their outliers, and gives no bracket: a mis-rooted result
  can lie outside the bin, after which the tail `where` selects the wrong branch.
- **Rejected: `jax.lax.while_loop` to tolerance.** Not reverse-differentiable; a fixed
  `fori_loop` is, and 20 iterations of a quadratically convergent method on a well-scaled
  cubic reach round-off long before the budget (the suite checks round trips to `1e-8`).
- **Rejected: differentiating through the iterations.** Correct but wasteful (20 unrolled
  steps in the tangent) and, for reverse mode through the `where`s, a NaN source; the
  implicit rule is exact and costs one cubic evaluation.

What the scheme guarantees, for any parameters and any `y` in the range: the root is in
the bracket at every iteration and the iterate never leaves `[0, 1]`, so the inverse is
always finite and in-bin; the iterate converges to the root (every accepted Newton step
strictly shrinks the bracket and the step length vanishes only at the root, since `f'` is
bounded below by Theorem 1 of [1]); and convergence is quadratic once close. What it does
*not* guarantee is a uniform `2^{-newton_iters}` worst-case error: that bound holds for the
bisection fall-back steps alone, and an accepted Newton step may shrink the bracket by less
than half. The earlier docstring claim to that effect was corrected on this branch
(`invertible-cleanup`); the design note states the bound precisely.

## Consequences

- `CubicBSpline` is `C²` as declared and identity at `raw = 0` (uniform knots, Greville
  coefficients), so it is a valid coupling template under ADR-0002.
- Expressiveness in the two edge bins is reduced relative to [1]; if that matters in
  practice, the lever is `num_bins`, not the boundary rule.
- Periodic boundary conditions (Theorem 2 of [1]) are a different pinning rule on the same
  machinery — an endpoint-handling option on `AbstractSpline`, parked to Phase E of the
  roadmap.

## References

[1] S. Hong and S. Y. Chun. Neural Diffeomorphic Non-uniform B-spline Flows. AAAI (2023).
    arXiv:2304.04555.
[2] C. de Boor. A Practical Guide to Splines, revised ed. Springer (2001). Ch. IX
    (Marsden's identity, the Greville abscissae) and Ch. X (the derivative formula).
