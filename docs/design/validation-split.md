---
type: design
status: tentative
updated: 2026-10-06
verified_by: pending (Joon; the remedy chosen in discussion 2026-10-06)
sources: [ADR-0008 (window sampling), ADR-0009 §5 (evaluation)]
---

# Validation error split by window start: `val/mse_early` and `val/mse_late`

## 1. The problem: the validation mean is a near-cycle metric

Every trajectory in a dataset starts off the limit cycle and converges to it (the initial
conditions are drawn around the cycle, `OnCycleGaussian`, or in a box or annulus). Along a
trajectory of $T$ time steps the transient occupies the first $T_{\rm tr}$ steps and the
remaining $T - T_{\rm tr}$ steps are near-periodic motion on the cycle. Validation
(`validation_windows`, ADR-0009 §5) enumerates every window of length $L$ with stride $s$,
so a trajectory contributes about $(T - L)/s$ windows, of which only about $T_{\rm tr}/s$
start in the transient. For the Bautin set of Phase B ($T = 501$, $L = 100$, settling in
roughly the first hundred steps) that is one window in four; for a longer run of the same
system it is fewer still.

`val/mse` is the mean over all windows, so it is dominated by windows that start on the
cycle. A model that reproduces the cycle and gets the transient wrong — the approach to the
cycle is exactly where the amplitude coordinate, the Floquet exponent and the isochrons'
shape are decided — scores almost as well as one that gets both right. Selecting the best
checkpoint on `val/mse` (ADR-0009 §6) therefore selects for the cycle **[deduced from the
window count; not a measurement of a trained model yet]**.

The training sampler can be made to oversample the transient (`weighted`, `mixed`,
ADR-0008), but the validation metric must not follow the training distribution blindly —
it should *report* what the model does on each part.

## 2. The split

Every validation window is $\lbrace t_k \rbrace_{k=0}^{L-1}$ with data $u_k$; its start
time $t_0$ says where on its trajectory it lies, and since all trajectories share the grid
and the settling time is a property of the system, $t_0$ is a usable proxy for "transient
or cycle". Given a threshold $t_{\rm split}$, the `Evaluator` reports

$$
\mathrm{val/mse\_early} = \frac{1}{N_{\rm early}} \sum_{w : t_0(w) < t_{\rm split}} e(w), \qquad
\mathrm{val/mse\_late} = \frac{1}{N_{\rm late}} \sum_{w : t_0(w) \ge t_{\rm split}} e(w),
$$

with $e(w)$ the window's trajectory MSE (mean over time of the squared Euclidean error of
the rollout from the window's first point, `losses.trajectory_mse`) and $N_{\rm early} +
N_{\rm late} = N$. `val/mse` is unchanged — the overall mean, $(N_{\rm early} \cdot
\mathrm{early} + N_{\rm late} \cdot \mathrm{late}) / N$ — so existing runs and the default
checkpoint metric are unaffected. The counts are reported too (`val/n_early`,
`val/n_late`); an empty bucket is `NaN`, never a silent zero.

Implementation: `Evaluator(val_data, reference, t_split=None)`; the jitted
`prediction_sums(model, batch, t_split)` returns per-batch sums (total and early) and the
`Evaluator` accumulates them across batches exactly as it does for the phase and amplitude
statistics (ADR-0009 §5) — nothing is held in memory, one pass. `t_split = None` disables
the split.

## 3. Choosing $t_{\rm split}$

- **With the `mixed` training sampler**, the natural value is the sampler's own boundary,
  $t_{\rm split} = t_{\rm s}$ with `split_idx` the index the two start ranges meet at
  (ADR-0008): training oversamples "before $t_{\rm s}$" and validation reports "before
  $t_{\rm s}$" — the same definition of the transient on both sides. `validation.t_split:
  null` resolves to this automatically (`experiment.validation_t_split`).
- **With `weighted` sampling** (`transient_weight(boost, tau)`) there is no hard boundary;
  a few multiples of `tau` after $t_0$ is the analogous choice, set explicitly.
- **Without either**, set it from the system: the settling time of the amplitude
  coordinate, $\sim 1/\vert\kappa\vert$ for a normal form with Floquet exponent $\kappa$,
  times a small factor; for FitzHugh–Nagumo, from the numerical Floquet exponent once
  Phase E's `analysis` provides it (and `t_settle` per trajectory in the dataset, roadmap).

In the config: `validation.t_split` (a time, or `null` for the rule above), and
`checkpoint.metric: val/mse_early` to make the saved "best" model the one that gets the
transient right.

## 4. What the split does not do

- It buckets by **where a window starts along its trajectory**, not by **how far from the
  cycle the data are**. If every trajectory starts within a hair of the cycle, `early` is
  an easy test; the fix is a validation set with far-from-cycle initial conditions
  (`UniformAnnulus`, a dedicated `data` config) — Phase E, remedy (c) of the discussion.
- It does not reweight training. The training distribution is the sampler's business
  (ADR-0008); the split only makes the fit on each part visible.
- Two buckets, not a curve. A per-start-time curve (`val/mse_by_start/<k>`) is a useful
  occasional diagnostic (a notebook over the `Evaluator`'s sums with several thresholds),
  not a checkpoint-selection metric; it is not implemented.

## 5. Alternatives considered

- *Importance-weighting the validation MSE* by the training weight ($w(t_0)$): one number,
  but it inherits the training bias and hides the cycle fit — the opposite failure.
- *Validating on early windows only* (`start_range` on the validation set): loses the check
  that the cycle is still fit; the two-bucket report keeps both.
- *A far-from-cycle validation dataset* (different initial-condition sampler): the real
  remedy for the coverage question in §4, orthogonal to the split; planned, not a
  replacement.
