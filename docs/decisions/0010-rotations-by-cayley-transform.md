---
type: decision
id: ADR-0010
status: accepted
updated: 2026-10-05
verified_by: pending (Joon; measured basis 2026-10-05 on a V100)
---

# ADR-0010 — Rotations in `BiLipschitzLinear` are Cayley transforms, not matrix exponentials

**Status**: accepted (2026-10-05, branch `trainer`, Joon: "let's resolve this now").

## Context

`BiLipschitzLinear` parametrizes `U, V ∈ SO(d)` from unconstrained leaves as
`expm(raw − rawᵀ)`. Benchmarking the training step on a V100 (`scripts/bench_dataloader.py
--hlo-stats`, change document 2026-10-04 round 2) found the step host-bound at ≈ 90 ms with
the data already on the device, and the compiled step's loop census explained it:

- 114 `while` loops, ≈ 15 400 kernel launches per step; 112 of the loops were
  `16 iterations × 10 launches` — `jax.scipy.linalg.expm`'s squaring step, a
  `lax.scan` of `max_squarings = 16` `lax.cond`s (JAX source, `_squaring`). Two rotations
  × 8 layers × (`__call__` and `inverse` both read `params`) × (primal and the Fréchet
  `expm` of the custom JVP) ≈ 112.
- ≈ 1 700 `conditional` executions per step. On the GPU backend a conditional needs its
  predicate on the host, so each is a device-to-host round trip the launching thread waits
  for; the host can never run ahead of the device, hence dispatch ≈ total.
- Measured on the V100: one 2×2 `expm` costs **0.649 ms** per call (16 round trips ≈ 40 µs
  each); the closed-form rotation of the same angle **0.049 ms**. 1 700 × 40 µs ≈ 70 ms is
  the step's fixed cost.

All this for the exponential of a 2×2 skew-symmetric matrix, i.e. a rotation.

## Decision

`U = cayley(raw_U − raw_Uᵀ)`, `V = cayley(raw_V − raw_Vᵀ)` with the Cayley transform

$$
Q = (I - A)(I + A)^{-1}, \qquad A^{\top} = -A .
$$

- `Q` is orthogonal with `det Q = +1` for every skew `A` (deduced: `(I − A)` and
  `(I + A)⁻¹` commute and `Qᵀ Q = (I + A)^{-\top}(I − A)^{\top}(I − A)(I + A)^{-1} = I`
  since `(I − A)ᵀ = I + A`; the determinant is continuous in `A`, nonzero, and `1` at
  `A = 0`), and the eigenvalues of `I + A` are `1 + iλ`, so the solve is always well
  conditioned — including at the extreme raw values `tests/test_linear.py` draws.
- `Q = I` at `A = 0`: ADR-0002 holds, the identity initialization is unchanged.
- Loop-free and conditional-free: for `dim = 2` the inverse is written out (`Q` is the
  rotation by `−2 arctan a`, `a = A[1, 0]`); for `dim > 2` one `jnp.linalg.solve` (an LU
  custom call; its pivot handling keeps a `dim`-trip loop without conditionals).
- Coverage: every rotation except those with an eigenvalue `−1` (a half turn in some
  plane; for `d = 2` the rotation by `π`). Those are not reached by a continuous path from
  the identity within the chart, and an INN layer does not need them exactly.

`tests/test_linear.py::test_rotations_have_no_conditionals` pins the absence of
`conditional` (and, for `dim = 2`, of `while`) in the compiled parameter map;
`test_cayley_dim2_closed_form_matches_the_solve` pins the written-out inverse against the
general formula. The existing parameter-space laws (orthogonality, `det > 0`, singular
values in `(1/L, L)`) cover the rest.

## Alternatives rejected

- **Keeping `expm` with a smaller `max_squarings`** — fewer conditionals, not none; the
  Padé-order selection adds its own `cond`s, and the Fréchet derivative doubles them.
- **Householder products** — `d` reflections, loop-free, but the sign (`det = (−1)^d`)
  needs a fix-up and the parametrization is not the identity at zero without extra care.
- **Changing the step** (unrolling, XLA command buffers for loops) — treats the symptom;
  the root solve's two `fori_loop`s remain the only loops in the step and are the next
  candidate if the step is still launch-bound.

## Consequences

- `raw_U`/`raw_V` keep their meaning as skew generators but the map to the rotation
  changes (angle `−2 arctan a` instead of `a` for `d = 2`): checkpoints from before this
  change do not reproduce the same rotations. None exist yet.
- The roadmap's parked note on `SO(d)` parametrizations is resolved; reinstating
  `InvertibleLinear` for speed is off the table.
- `scripts/bench_dataloader.py --hlo-stats` on CPU, 8 blocks: 3 `while` loops, 0
  conditionals, ≈ 4 100 launches (from 7 / 40 / ≈ 2 000 with 2 blocks before). The V100
  figure is recorded in the change document when measured.
