---
type: architecture
status: current
updated: 2026-10-30
sources:
  - "code"
---

# Architecture

This page is the shape of the `deep_isochron` package: what each module is for, how data
flows through a training step, and where the design decision behind each part is
recorded. Decisions live in `docs/decisions/` (ADRs), the mathematics in `docs/design/`, the
how-tos in `docs/guides/`, the current plan in
`docs/roadmap.md`, and branch-level detail in `docs/changes/`; `docs/index.md` lists
every document. The page is kept current with the code: when a phase of the roadmap
lands, the relevant section here changes in the same commit.

## Purpose

The project learns a smooth conjugacy `Φ` between an oscillator with a stable limit cycle,
observed through trajectories, and a normal form whose phase/amplitude structure is known
in closed form, so that the oscillator's isochrons and phase response are read off `Φ`.
Concretely, the observed system is FitzHugh–Nagumo, the latent one is the Bautin normal
form, and `Φ` is an invertible neural network. The conjugacy equation
`Φ ∘ φ_t = ψ_t ∘ Φ`, with `φ_t` and `ψ_t` the flows of the two systems, is enforced on
trajectories.

## Module map

```
deep_isochron
├── systems/                 ODEs (observed and normal forms)      ── AbstractODE
│   ├── base.py              rhs(t, u) · params() · in_basin(u) · SolverConfig ·
│   │                        flow -> Solution (numerical; normal forms choose via
│   │                        integration=) · flow_to(t, u0) -> state
│   ├── symbolic.py          SymbolicODE (states, params, rhs as SymPy); AbstractSymbolicODE:
│   │                        _symbolic_template() classmethod, rhs/params() derived; compiled
│   │                        by _compile once a class has no abstract methods ── ADR-0016
│   ├── fitzhugh_nagumo.py   FitzhughNagumo (dim 2; symbolic)
│   ├── level_set.py         LevelSetCycle: f = s(R∇g − g∇g), the zero level set of g is the
│   │                        attracting cycle; guckenheimer_meloon(), cassini() test orbits
│   ├── hodgekin_huxley.py   HodgekinHuxley (dim 4; symbolic, rhs only — not AUTO material)
│   └── normal_forms/
│       ├── base.py          AbstractNormalForm(AbstractSymbolicODE): the polar template
│       │                    _symbolic_template_polar() gives ρ = ṙ/r, ω = θ̇ and the derived
│       │                    cartesian template, checked at class creation; on a basin
│       │                    (r_in, r_out); closed-form phase Θ, isostable Ψ, isochrons, κ,
│       │                    polar and (Θ, Ψ) charts (nan outside the basin); closed_form
│       │                    default. AbstractEvenNormalForm: declared family whose rates
│       │                    are functions of s = r² (checked), `_sq` hooks, smooth at the
│       │                    origin, eigenvalues at 0                         ── ADR-0013
│       ├── hopf.py          HopfNormalForm   (even)
│       ├── bautin.py        BautinNormalForm (even)
│       ├── winfree.py       WinfreeNormalForm: Winfree's model with a hole (Langfield et
│       │                    al. 2025) — not even; basin (a, ∞), phaseless disk r <= a
│       └── integration.py   CartesianIntegration / PolarIntegration /
│                            RadiusSquaredIntegration (even forms only) — flow(..., integration=)
├── model/
│   ├── invertible/          the INN vocabulary                     ── AbstractBijection
│   │   ├── base.py          AbstractBijection, AbstractScalarBijection, ScalarChain,
│   │   │                    SequentialINN
│   │   ├── constraints.py   Free, Arcsinh, Positive, BoundedPositive, Interval, Widths
│   │   ├── affine.py        Shift, Affine; AffineCoupling/ResidualCoupling factories
│   │   ├── analytic.py      CubicRational, SinhConjugation, CubicConjugation
│   │   ├── splines/         AbstractSpline; LinearSpline (C⁰), MonotonicRQSpline (C¹),
│   │   │                    CubicBSpline (C²)
│   │   ├── coupling.py      CouplingFlow[B] — any scalar template × any conditioner
│   │   ├── polar.py         OffsetedBijection, RadialBijection, PolarCouplingFlow,
│   │   │                    CircularMonotonicRQCoupling
│   │   └── linear.py        BiLipschitzLinear (the linear layer)
│   ├── fourier.py           TruncatedFourier (conditioner on S¹)
│   ├── base.py              AbstractPhaseAmplitudeModel: phase · amplitude · cycle_point ·
│   │                        isochron_coordinate · isochron · __call__; phase_sensitivity derived
│   ├── conjugacy.py         ConjugateLatentDynamics = bijection + normal form + SolverConfig;
│   │                        equilibrium() = H⁻¹(0)
│   ├── latent_dynamics.py   PhaseAmplitudeLatentDynamics(omega, kappa): rotation ⊕ decay
│   ├── autoencoder.py       PhaseAmplitudeAutoencoder — Yawata et al. 2024 baseline;
│   │                        phase(x), phase_sensitivity(θ)
│   └── utils.py             zero_final_layer
├── data/                    one netCDF4 file per dataset (xarray)       ── ADR-0008
│   ├── dataset.py           DatasetMetadata (grouped); TimeSeriesDataSource = whole
│   │                        trajectories (frozen dataclass; splits; save/load)
│   ├── initial_conditions.py  AbstractICSampler; UniformBox, UniformAnnulus, OnCycleGaussian
│   ├── generate.py          generate(system, sampler, ts, n, seed=…) with loud failures;
│   │                        generation_metadata, dataset_path
│   ├── device.py            consuming any pipeline next to JAX: single_threaded();
│   │                        to_device(ds, device); resolve_device()
│   └── windows/             two ways to cut windows (ADR-0008 D2, D3) — see its __init__
│       ├── common.py        Element/Batch/WeightFn, range checks, transient_weight,
│       │                    categorical, start_weights (shared)
│       ├── elementwise.py   RandomWindow / WeightedWindow (grain RandomMap); windows();
│       │                    mixed_windows(); validation_windows() — the reference
│       └── batched.py       WindowBatchSource (batches by one gather, epochs of every
│                            window once); window_batches(); mixed_window_batches() — the
│                            training-run path
├── analysis/                the limit cycle and its neighborhood, numerically  ── ADR-0012
│   ├── cycle.py             Cycle: planar closed curve as a Fourier series in a phase
│   │                        uniform in time, + period; from_points, circle, distance,
│   │                        nearest_phase (shared by both halves; design: cycle-estimation.md)
│   ├── data/                from sampled, possibly noisy trajectories (host-side NumPy)
│   │   └── cycle.py         estimate_cycle (E1): tails → polar protophase → Kralemann
│   │                        PTP phase → period by regression → Fourier curve → converged;
│   │                        orientation(points) (signed area of a closed polygon)
│   ├── errors.py            NotConvergedError (stage, message, last iterate)
│   ├── model.py             a trained model against a reference: phase_offset(model, cycle)
│   │                        -> (offset, sense), phase_sense — the seam between model/ and
│   │                        analysis.ode (neither knows the other)
│   └── ode/                 from an AbstractODE (status: roadmap Phase E)     ── ADR-0014/15
│       │                    results are fixed-shape eqx.Modules; functions of them
│       │                    (f(cycle, …)) jit/vmap as they stand and are left unjitted;
│       │                    the host-side drivers jit the stages they call
│       ├── auto/            isochrons by AUTO-07p continuation (planar)        ── ADR-0016
│       │   ├── codegen.py   write_problem(ode, workdir) -> AutoProblem: rhs.inc (F and the
│       │   │                exact Jacobian from symbolic()), indices.inc (IP_* slots), c.<name>;
│       │   │                internal_indices -> Idx; auto_pars; never imports AUTO
│       │   ├── _env.py      load_auto() -> AutoHandle: finds the built AUTO tree
│       │   │                (DEEP_ISOCHRON_AUTO_DIR, AUTO_DIR, ~/.local/opt/auto-07p), sets
│       │   │                AUTO_DIR, imports `auto`; the only importer of AUTO (docs/guides/auto07p.md)
│       │   ├── results.py   IsochronSettings; Isochron (resample, log_polar, hairpins);
│       │   │                IsochronFailure;
│       │   │                IsochronFoliation (dataset, save/load as netCDF, loaded against
│       │   │                its cycle) — docs/design/isochrons.md, ADR-0017
│       │   ├── runner.py    auto_session(ode) -> Runner: AUTO on the generated problem in a
│       │   │                temporary directory, one thread, output captured; AutoSolution
│       │   │                (the one shape the stages exchange); bvp(ndim, nbc)
│       │   ├── stages.py    periodic_orbit -> adjoint_bundle -> rotate_phase -> isochron:
│       │   │                the four stages of design §6, each Runner + AutoSolution in,
│       │   │                AutoSolution (or Isochron) out
│       │   ├── foliation.py isochron_foliation(cycle, phases, settings) -> IsochronFoliation:
│       │   │                stages 1–3 once, trace_isochron per phase in a spawn pool
│       │   │                (JAX on the CPU in the workers); wrap_phases; provenance
│       │   └── fortran/     isochron_multiseg.f90 — the generic multi-segment BVP (plain
│       │                    Fortran; INCLUDEs the two generated files; adjoint bundle)
│       ├── sections.py      CrossSection: a hyperplane (point, unit normal); signed_distance;
│       │                    normal_to_flow(ode, p)
│       ├── poincare.py      PoincareMap: first-return map on a section (direction, reverse
│       │                    time); __call__ and iterate jitted/vmappable and differentiable
│       │                    in the point (custom JVP, Parker & Chua 5.21); fixed_point
│       │                    (host-side settle + minimum-period test) -> (FixedPoint,
│       │                    FixedPointIteration)
│       ├── shooting.py      multiple_shooting: LM on the time-T flow, AD Jacobians, jitted;
│       │                    SHOOTING_SOLVER_CONFIG
│       ├── limit_cycle.py   find_limit_cycle = settle → initial_guess → polish → guards
│       │                    (is_equilibrium, period_consistent) → LimitCycle (ode, the
│       │                    shooting mesh x0s, period, config; sample(n), orientation());
│       │                    nearest_cycle_time(cycle, y) (Newton in time); reanchor(cycle,
│       │                    point) moves phase 0 to the nearest orbit point
│       ├── equilibrium.py   find_equilibrium(ode, guess) -> (point, converged): Newton on
│       │                    the field (Optimistix), for the phaseless point of a cycle
│       ├── phase.py         asymptotic_phase(cycle, x, t_end=) → AsymptoticPhase (phase,
│       │                    distance, discrepancy): flow, read off the cycle twice;
│       │                    phase_grid (host-side, lax.map chunks) → field, mask, result
│       └── floquet.py       floquet(cycle, method=) → Floquet, dispatching over
│                            floquet_periodic_schur (seeded chain on the segment
│                            Jacobians: the method, with Floquet vectors at every mesh
│                            point), floquet_monodromy and floquet_liouville (checks);
│                            return_map_multiplier
├── training/                                                            ── ADR-0009
│   ├── trainer.py           TrainerState (model · opt_state · step · key · schedule
│   │                        state); Trainer(optimizer, loss, schedule): jitted
│   │                        train_step, train(loader, logger, checkpointer, evaluate)
│   ├── losses.py            building blocks (trajectory_mse, step_weighted_consistency,
│   │                        alpha_schedule, batch_center_of_mass); AbstractLoss =
│   │                        weighted named terms; ConjugacyTrajectoryLoss,
│   │                        PhaseAutoencoderLoss
│   ├── schedules.py         Constant / StepSchedule (curriculum) / ThresholdSwitch
│   ├── optimizers.py        decay_mask: weight decay off the latent-dynamics parameters
│   ├── loggers.py           Logger base class; Null / List / Print / Wandb; DelayedLogger;
│   │                        MultiLogger; EpochLogger
│   ├── checkpoint.py        Checkpointer base class; OrbaxCheckpointer (whole state)
│   └── evaluation.py        Evaluator(val_data, reference, t_split) on the model contract;
│                            val/mse_early | late by window start (design: validation-split);
│                            align_phases(model, reference, offset=) -> PhaseAlignment
│                            (sense, offset, residual, rms): the aligned phase error (M)
├── experiment/              configs -> a run (config parsing + instantiation only) ── ADR-0011
│   ├── compose.py           compose(*overrides): the train config as train.py sees it
│   ├── instantiate.py       `_target_` instantiation with YAML lists as tuples
│   ├── data.py              dataset_file / build_source (never generates) / generate_dataset /
│                            load_or_generate_source (notebooks); build_evaluator;
│   │                        build_window_source / Loaders.from_config (resume = slice);
│   │                        validation_t_split
│   ├── isochrons.py         IsochronConfig / CycleConfig / PhasesConfig, the schemas of
│   │                        configs/isochrons/ and configs/cycle/ (ADR-0011 §1), with
│   │                        wrapped_phases / result_digest / file; compute_isochrons (refuses an
│   │                        existing file) / load_isochrons (never computes; rebuilds the
│   │                        cycle): the reference isochrons (ADR-0017)
│   ├── model.py             build_model (`_target_` layers + per-layer keys + flip)
│   ├── training.py          build_loss / build_schedule / build_optimizer / build_trainer
│   ├── logging.py           build_logger: Print (+ Wandb online/offline) → Epoch → Delayed
│   └── run.py               Run (the directory: config.yaml, metadata.json, checkpoints/;
│                            create refuses an existing run, resume is explicit);
│                            Experiment.from_config(cfg, run_dir); train = Experiment.from_config().train();
│                            load_run / load_model (template= for hand-built models)
├── plotting/                matplotlib style (configure_rcparams), colored_curve;
│   ├── trajectories.py      plot_source_trajectories
│   └── isochrons.py         plot_phase_field (a field panel, isochrons as white curves);
│                            plot_phase_comparison (reference, learned, aligned residual);
│                            plot_isochron_match (reference colored by phase, learned
│                            dashed); plot_isochron_hairpins (one isochron into the
│                            phaseless point, plane and log-polar, matched to the model)
└── utils/                   shared helpers
    ├── numerics.py          inv_softplus, squashed_exp, polar ↔ cartesian, wrap_phase
    │                        (constraints, normal forms, analysis)
    └── provenance.py        git_state(), package_version() — for dataset and run metadata

isochron2pbvp/               the second project in this repository: Andrus Giraldo's AUTO-07p
                             isochrons of FitzHugh–Nagumo, under his conventions (its README);
                             the package's checks stop at its edge; the pipeline was built from
                             and compared with commit 3af99f2 (design/isochrons.md)

configs/train.yaml + {data,model,loss,schedule,optimizer,wandb}/   the composed experiment
configs/cycle/ (schema CycleConfig)                                a system's cycle and phase 0
configs/phase_field/ (schema PhaseFieldConfig)                     t_end of phase_grid, per system
configs/isochrons/ (schema IsochronConfig; composes data + cycle)  reference isochrons
configs/hydra/launcher/slurm.yaml                                  SLURM jobs (submitit)
scripts/                                        the command-line entry points (scripts/README.md)
├── train.py                                    Hydra entry point: one training run
├── generate_data.py (+ configs/data/)          Hydra entry point for data generation
├── compute_isochrons.py (+ configs/isochrons/) Hydra entry point for AUTO isochrons
├── install_auto.sh                             AUTO-07p's Fortran tree at the pinned commit
├── diagnostics/                                instruments for a run
│   ├── bench_dataloader.py                     data-pipeline vs training-step benchmark
│   │                                           (also --config <overrides>: a config's own run)
│   ├── find_nan.py                             first non-finite gradient of a run, with the latent extent
│   └── dump_auto_problem.py                    the AUTO files of an isochron config, into a directory, with the slot table
└── checks/                                     the deterministic checks (prek.toml; docs/coding-standards.md)
    ├── check_md_math.py                        GitHub-safe Markdown math check
    ├── check_spelling.py                       the -ise families (American English)
    ├── check_references.py                     every citation and relative link resolves
    ├── check_docstrings.py                     docstring headers; private functions documented; modules under 600 lines
    ├── check_containers.py                     the container rule: no Array in a frozen dataclass or a static field; NamedTuple only for *Params
    └── check_changes.py                        a Review notes row per Files bullet of a change document
.github/workflows/ci.yml                        lint · ty · math · spelling · citations · tests; slow + end-to-end run
```

## The invertible package

The package has two abstract classes and one rule that ties them together.

`AbstractBijection` is a diffeomorphism of `Rᵈ`: `__call__`, `inverse`, `jacobian`
(forward-mode by default), and two declared static properties, `dim` and `smoothness`.
The map is `C^k` with `k = smoothness`, `None` is `C^∞`, and containers take the minimum
(ADR-0003). Every concrete subclass is the identity map when freshly constructed and
orientation-preserving at every point of its parameter space; both are universal laws in
the test suite.

`AbstractScalarBijection[P]` is an `R → R` map with exactly one array leaf, the
unconstrained vector `raw`. Its constrained parameters, a `NamedTuple` `P`, are computed
on read by `constrain(raw)`, with `constrain(0)` the identity (ADR-0001, ADR-0002).
`constrain` is built from the primitives in `constraints.py`, plain Python objects that
are created inside it and never stored (ADR-0005), each carrying its own identity-at-zero
shift. An instance with `raw = None` is a *template*: static configuration only. The
class docstring is the implementation checklist, and `Affine` is the smallest example.

The rule that ties them together is `CouplingFlow`. It stores a scalar *template* and a
conditioner `x_const → raw`, by default an MLP with a zero-initialized last layer and a
`TruncatedFourier` for the polar variant, and writes the conditioner's output straight
into one scalar bijection per coupled coordinate through `template.from_unconstrained`.
Because `raw` is unconstrained, any conditioner output is valid; because `constrain(0)` is
the identity, a fresh layer is the identity. So every scalar bijection, whether analytic,
spline, chain or offset, is a coupling layer without further work, and `AffineCoupling`
and `ResidualCoupling` are three-line factories rather than classes.

`BiLipschitzLinear` is the only linear layer: `U diag(s) Vᵀ` with `U, V ∈ SO(d)` and
`s ∈ (1/L, L)`, so it is orientation-preserving and well-conditioned everywhere. The
rotations are Cayley transforms of skew generators (ADR-0010), because `expm`'s 16
`lax.cond`s per rotation were the training step's host-side cost on GPU. The
unconstrained `InvertibleLinear` was removed for failing both properties (change document
2026-10-02). An INN is a `SequentialINN` of alternating linear and coupling layers, the
notebook's `make_invertible_block`.

Smoothness matters here because the conjugacy must be at least `C¹` for the isochron
geometry (Jacobians) to be meaningful and `C²` for curvature-based losses (Phase E). The
cubic B-spline is the production candidate for that reason; its design is in ADR-0006 and
`docs/design/cubic-bspline.md`.

## Systems and normal forms

The systems package has two layers and one rule (ADR-0007). `AbstractODE` is any ODE in
the study: `rhs(t, u, args)` in the diffrax signature, `dim`, `params()` (the constrained
parameter values, for metadata), and the numerical
`flow(ts, u0, *, config: SolverConfig) -> diffrax.Solution`, whose `.ys` is the
trajectory and whose `.result` serves batched generation with `throw=False`.
`SolverConfig` holds solver, tolerances, `max_steps`, adjoint and `throw` as static
fields; it is leafless, so under `eqx.filter_vmap`/`filter_jit` it is static. `flow` is
written for one initial condition, and batching is
`eqx.filter_vmap(ode.flow, in_axes=(None, 0))`.

The observed systems declare their vector field once, as SymPy expressions
(`systems/symbolic.py`, ADR-0016), in the classmethod `_symbolic_template()` of
`AbstractSymbolicODE(AbstractODE)`. `rhs` is derived from the template: it is compiled
through `sympy2jax` once per class, in `__init_subclass__`, and stored as a class
attribute that is neither a field nor a pytree leaf. `params()` is derived the same way,
and so is the AUTO-07p Fortran `RHS` with its exact Jacobian, in
`analysis/ode/auto/codegen.py`. Parameter symbols are named after the attributes that
hold their values (fields, or the constrained-value properties of the normal forms), so
`getattr(self, name)` is the value and the template's `params` order is AUTO's
`PAR(1..n)`. `AbstractODE` itself stays free: a learned or pushforward field is an ODE
without being symbolic. The normal forms are `AbstractSymbolicODE`s too, with a polar
template from which the cartesian one is derived (next paragraph), so the code
generator takes an `AbstractSymbolicODE` for every system.

`AbstractNormalForm(AbstractSymbolicODE)` is the subset with closed-form phase–amplitude
structure, a conjugacy target or a reference with known isochrons. Such a form is planar
and rotationally symmetric, `ṙ = rρ(r)`, `θ̇ = ω(r)`, with a stable cycle at `r = 1`, on
a declared basin `basin_radii() = (r_in, r_out)` outside which the chart is `nan`
(ADR-0013). A subclass supplies its field in polar coordinates *as SymPy*,
`_symbolic_template_polar()` returning `(ṙ, θ̇)` in the states `(r, θ)`, from which the
rates `ρ = ṙ/r` and `ω = θ̇` and the JAX hooks are derived (ADR-0016), and two
closed-form integrals, `phase_shift` `h(r)` and `isostable` `ψ(r)`, as hand-derived JAX
that the tests verify symbolically. The base derives the rest: the cartesian template
(polynomial for the even family), the cartesian `rhs`, `rhs_polar`, `params()`,
`period`, `floquet_exponent` (`κ = ρ'(1)` by autodiff),
`phase` (`Θ = σ(θ + h(r))/2π` in `[0, 1)`, ADR-0020), `amplitude` (`Ψ`, `∂ᵣΨ(1) = 1`), `limit_cycle`, `isochron`,
and the chart maps `to_polar`/`from_polar` (polar) and
`to_phase_amplitude`/`from_phase_amplitude`. The latter goes through
`radius_from_isostable`, a differentiable root solve, unless the subclass overrides it
with the explicit inverse, as Hopf does. The mathematics and the implementation contract
are `docs/design/normal-forms.md` (§9). `AbstractEvenNormalForm` is the family whose
rates are functions of `s = r²`, declared by the base class and checked when a form is
created; its `_sq` hooks are compiled in `s`, and its field is smooth at the origin, with
`eigenvalues_origin` and the `r_squared` integration; Hopf and Bautin belong to it, and
`WinfreeNormalForm` (Langfield et al. 2025) is the first form outside it, with a
phaseless disk. The rule is that a class carries only what is available analytically.
Anything numerical, such as locating a limit cycle, the monodromy, or the asymptotic
phase of FitzHugh–Nagumo, is a function over `AbstractODE` in `analysis/ode` (Phase E;
`limit_cycle` landed as O1). Everything public is cartesian.

Integrations (`normal_forms/integration.py`) are objects in the diffrax style: the caller
passes an instance, or one of `"cartesian" | "polar" | "r_squared" | "closed_form"` for
the defaults, where `closed_form` evolves `(Θ, Ψ)` exactly and maps back with no solver.
They are leafless modules, hence static, so each traces separately and dispatch is free.
All return cartesian `.ys` in the same `Solution`, so `ConjugateLatentDynamics` never
converts charts.

`base.py` is the contract both approaches implement, `AbstractPhaseAmplitudeModel`:
`phase(x)` in `[0, 1)`, a fraction of the period (ADR-0020), `amplitude(x)` (isostable-like, with no fixed normalization),
`cycle_point(θ)`, `isochron_coordinate(x)` and `isochron(θ, r)` (the learned isochron sampled
along the model's own latent radial coordinate, so that a reference isochron can be
matched point by point), and `__call__(ts, x0) -> (xt, yt)`; `phase_gradient` and
`phase_sensitivity` are derived once for all. Training and evaluation code takes the
contract and never asks which model it has. For `ConjugateLatentDynamics` the contract is
the normal form's closed forms transported by the bijection (`Θ = Θ_NF ∘ H`,
`cycle_point = H⁻¹ ∘ limit_cycle`, `isochron = H⁻¹ ∘ isochron_NF`), and it alone has an
`equilibrium()`, `H⁻¹(0)`: the autoencoder's latent is the unit circle times a
deviation, with no point for the phaseless set.

`latent_dynamics.py` and `autoencoder.py` are the non-invertible baseline, the phase
autoencoder of Yawata et al. (Chaos 34, 063111, 2024), as mapped in
`docs/design/normal-forms.md` §5.3. `PhaseAmplitudeLatentDynamics(omega, kappa)` is the
closed-form flow on `R³`, with `(Y₁, Y₂)` rotating at `omega` and `Y₃` decaying at
`kappa < 0` (Eqs. 12–14). `PhaseAmplitudeAutoencoder` is an MLP encoder whose first two
outputs are normalized to the unit circle (Eqs. 15–16), an MLP decoder, and that flow;
`phase(x) = atan2(Y₂, Y₁)` (Eq. 19) and `phase_sensitivity(θ)` is `jax.grad` at the
decoded cycle point (Eq. 20). Its latent space is exactly the `(Θ, Ψ)` chart of a normal
form, `(cos Θ, sin Θ, Ψ)`, which is what `tests/test_baseline.py` checks. The loss
`PhaseAutoencoderLoss` (Eqs. 21–26) lives in `training/losses.py`, and the near-cycle
training distribution is the `OnCycleGaussian` sampler (Eqs. 27–28). The baseline does
not use `AbstractODE`.

## Data

A dataset is one netCDF4 file, HDF5 underneath (ADR-0008): dims `(trajectory, time,
dim)`, the time grid as the `time` coordinate, and a grouped `DatasetMetadata` with one
JSON-string attribute per group. The groups are `system` (name, constrained `params()`),
`sampling` (IC sampler, params, seed, `n_trajectories`), `grid` (`t0`, `t1`, `n`),
`solve` (solver, tolerances, `max_steps`, integration), `provenance` (created, git
SHA/dirty, package version, dtype), and `extra` (the resolved Hydra config).
`config_hash` is a property over the first four groups, so an unchanged config
regenerates to the same `<name>-<hash>.nc`.

`TimeSeriesDataSource` is a frozen dataclass (`ts`, `ys`, `metadata`) and a grain
random-access source over whole trajectories: `len` is the number of trajectories and
`source[i] = {"t": ts, "u": ys[i]}`. It carries no window bookkeeping. `split_time(idx)`
and `split_trajectories(frac, seed)` are `copy.replace` with sliced arrays; `save`/`load`
(with an optional dtype assertion) are the only I/O; `dataset` is the xarray view.

`generate(system, ic_sampler, ts, n, *, seed, config, integration, extra)` draws initial
conditions from an `AbstractICSampler` (`UniformBox`, `UniformAnnulus`,
`OnCycleGaussian`), vmaps `flow` with `throw=False`, and raises naming any failed
indices. `scripts/generate_data.py` is the Hydra entry point over
`configs/data/*.yaml`.

Training runs read batches of windows from a `WindowBatchSource`
(`data/windows/batched.py`, ADR-0008 Decision 3). Element `i` is the `i`-th batch of the
run, one vectorized gather, and an epoch is every window of every trajectory once, with
a fresh permutation per epoch and the remainder dropped, so the loader is finite and
`len = epochs · batches_per_epoch`. `window_batches(source, length, batch, seed=…,
epochs=… | num_steps=…, weight=… | start_range=…)` and
`mixed_window_batches(…, split_idx, weights)` build it; weighted and mixed starts are
draws with replacement, where `transient_weight(boost, tau)` oversamples the transient.
The per-element grain transforms `RandomWindow`/`WeightedWindow` behind
`windows()`/`mixed_windows()` remain the reference semantics, since the tests compare
marginals, and they serve `validation_windows` (`data/windows/elementwise.py`). Batches
are dicts `{"t": (B, L), "u": (B, L, dim)}`.

## From a config to a run

```
configs/train.yaml ──compose──► cfg          (Hydra groups; `_target_` leaves; tests use hydra.compose)
        │
        ├─ configure_jax: x64, resolve_device(cfg.device) → jax_default_device
        ├─ build_source(cfg.data): <out_dir>/<name>-<hash>.nc  (hash = generation_metadata; no generation)
        ├─ build_trainer(cfg): loss, schedule (Constant ← default_weights), optimizer
        ├─ build_model(cfg.model, key(seed)) → trainer.init  — or OrbaxCheckpointer(resume).restore
        ├─ Loaders.from_config(cfg, source, device, start_step): split_trajectories; WindowBatchSource
        │        (num_steps → epochs); [start_step:]; to_device; validation_windows().batch
        ├─ Run.from_config(cfg, run_dir, metadata): config.yaml, metadata.json  (refuses an existing run;
        │        Experiment.from_config(..., resume=True) → Run.from_config(resume=True): restore latest, record config_changes)
        ├─ build_evaluator(cfg, source, loaders)        ─┐ all of the above = Experiment.from_config(cfg, run_dir)
        ├─ build_logger (Print, wandb? or wandb_run=) → DelayedLogger   → Experiment
        └─ Trainer.train(state, loader, num_steps, logger, run.checkpointer()  [= Experiment.train()]
                 (OrbaxCheckpointer(run_dir/checkpoints, custom_metadata=metadata)), evaluator, eval_every)
                 └─ final step always checkpointed → load_run(run_dir) / load_model(run_dir)
                    notebook: exp = Experiment.from_config(cfg, dir); trainer.train(exp.state, exp.loaders.train, logger=mine,
                    checkpointer=exp.run.checkpointer(), evaluate=exp.evaluator) → load_run(dir) works
```

## Data flow of one training step

```
loader ──► batch {t:[B,L], u:[B,L,d]}  (grain; to_device() prefetches to the accelerator)
                 │
Trainer.train_step(state, batch)                                  ── jitted, pure
                 ├─ weights = schedule.weights(state.schedule_state, state.step)
                 ├─ (loss, terms) = loss_fn(model, batch, weights)
                 │     ConjugacyTrajectoryLoss:  y0 = H(x[:, 0]); y[t] = nf.flow(t, y0).ys;
                 │                               x̂ = H⁻¹(y); data = mse(x, x̂),
                 │                               latent = mse(H(x[t]), y[t])
                 │     PhaseAutoencoderLoss:     Y = f_enc(x); Ŷ = latent flow of Y[:, 0];
                 │                               recon, pha, amp (α_k-weighted), aux
                 ├─ grads = ∇_model loss ; schedule_state' = schedule.update(…, terms)
                 ▼
state' = state.take_step(grads, schedule_state')   optax update on is_trainable leaves
                 │
                 ├─ logger.log({loss, terms, w/*}, step)   DelayedLogger: forwarded one
                 │                                          step later (host sync then)
                 └─ every eval_every: evaluate(model) → {val/mse, period, κ, …}
                                      logger.log(…); checkpointer.save(step, state, …)
```

The model is an `AbstractPhaseAmplitudeModel`: `ConjugateLatentDynamics` (bijection,
normal form, `SolverConfig`) or `PhaseAmplitudeAutoencoder`. A loss is a weighted sum of
named terms whose weights the trainer's schedule supplies each step (ADR-0009), and the
conjugacy equation is what `data` and `latent` together enforce. Batches come from
`window_batches`/`mixed_window_batches` over a `TimeSeriesDataSource`, a finite loader
in which the data define the run, and the loader transfers them with `to_device`.
Validation is `validation_windows` over the held-out trajectories, finite and
deterministic, evaluated to exhaustion by an `Evaluator`, whose `val/mse` is what the
`OrbaxCheckpointer` keeps the best checkpoint by.

## Tests

`tests/registry.py` declares *what* is tested (scalar templates, vector builders,
`UNTESTED` with reasons), and `tests/test_registry.py` fails if a concrete bijection is
missing from it. `tests/test_bijections.py` holds the laws every bijection satisfies
(round trip, identity at init, orientation, finiteness, Jacobian consistency, pytree
hygiene, one optimizer step), run through Hypothesis draws from `tests/strategies.py`.
Per-family laws are in `test_splines.py`, `test_analytic.py`, `test_constraints.py` and
`test_linear.py`. The other modules each cover one layer: `test_normal_forms.py` pins the
normal forms' closed forms by autodiff and the integrations; `test_systems.py` the flow
machinery and the observed systems' facts (Langfield et al. 2014); `test_data.py` the
data layer end to end (source, metadata, disk, device, generation); `test_windows.py`
both windowing paths and their agreement; `test_analysis.py` the analysis functions
against the normal forms and Langfield et al.'s FitzHugh–Nagumo facts, with and without
measurement noise; `test_baseline.py` the phase autoencoder against the exact chart,
plus one `slow` training test on Hopf data; `test_models.py` the
`AbstractPhaseAmplitudeModel` contract on both models; `test_training.py` the losses,
schedules, loggers, trainer loop, evaluation and Orbax round trip; and
`test_experiment.py` the experiment layer, where every shipped config composes and tiny
runs train end to end, reopen and resume. Solver configurations used by tests are named
in `tests/helpers.SOLVERS` with their reasons, like `TOL`. Shape annotations are checked
at runtime by the jaxtyping/beartype import hook (`conftest.py`).

## Decision records

| ADR | Decision |
|---|---|
| [0001](decisions/0001-unconstrained-leaves.md) | one unconstrained leaf `raw`; constrained values computed on read |
| [0002](decisions/0002-identity-at-zero.md) | `constrain(0)` is the identity; the shift lives in the map |
| [0003](decisions/0003-declared-smoothness.md) | `smoothness` is declared per class; containers take the minimum |
| [0004](decisions/0004-abstractvar-fields.md) | `AbstractVar`s implemented as static `init=False` fields |
| [0005](decisions/0005-constraint-primitives.md) | constraint primitives are plain objects created inside `constrain` |
| [0006](decisions/0006-cubic-bspline-boundary-and-inverse.md) | `CubicBSpline`: Greville-pinned boundary; bracketed Newton inverse |
| [0007](decisions/0007-systems-hierarchy-and-flow-strategies.md) | `AbstractODE` / `AbstractNormalForm`; `SolverConfig`; flow integrations as objects |
| [0008](decisions/0008-dataset-format-and-sampling.md) | one netCDF4 file per dataset via xarray; weighted windows alongside `mix`; batched window source for training runs |
| [0009](decisions/0009-trainer-losses-schedules-checkpoints.md) | trainer: injected logging/checkpointing, losses as weighted terms, schedules in the state, whole-state checkpoints |
| [0010](decisions/0010-rotations-by-cayley-transform.md) | `BiLipschitzLinear` rotations by Cayley transform, not `expm` |
| [0011](decisions/0011-experiment-layer.md) | experiment layer: YAML `_target_` configs, builders, Hydra-owned runs, explicit generation, forced final checkpoint |
| [0012](decisions/0012-analysis-layout.md) | `analysis`: a data-driven half and an ODE half; E1 self-contained; the data half host-side NumPy, the ODE half's results `eqx.Module`s; one commit per function |
| [0013](decisions/0013-normal-form-basins.md) | normal forms: data in `r` on a declared basin; evenness is the `AbstractEvenNormalForm` subfamily; `nan` outside the basin; `generate(outside_basin=)` |
| [0014](decisions/0014-limit-cycle-by-shooting.md) | the limit cycle from the vector field: first-return map on a flow-normal section, multiple shooting with AD Jacobians and LM; `PoincareMap` as the abstraction; reverse time for repelling cycles |
| [0015](decisions/0015-floquet-by-seeded-periodic-schur.md) | Floquet multipliers by the periodic Schur chain on the shooting mesh, seeded with the flow direction and run in the attracting direction; monodromy and Liouville as checks; `LimitCycle` an `eqx.Module` carrying its ODE and configuration; `f(cycle)` signatures |
| [0016](decisions/0016-symbolic-systems.md) | systems declared once in SymPy; JAX `rhs` and normal-form hooks derived (compiled per class in `__init_subclass__`); AUTO-07p Fortran assembled by `INCLUDE` from generated `rhs.inc`/`indices.inc` |
| [0018](decisions/0018-docstrings-map-and-contract.md) | docstrings: the module docstring is the map, a function docstring the contract; `**Attributes:**` lists; one bibliography `references.md` with `(Key, locator)` citations |
| [0019](decisions/0019-adrs-describe-the-current-design.md) | ADRs describe the current design; Motivation with evidence; History at the end; links checked |
