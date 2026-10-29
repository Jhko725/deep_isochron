"""The training package (roadmap C1–C5, ADR-0009), in six groups.

The losses: the building blocks on known inputs, `AbstractLoss` weighting the named
terms in order, the two project losses exposing their terms, and the autoencoder loss
vanishing on the exact chart (as in ``test_baseline``) and reproducing the paper's
``α_k``. The schedules: `Constant`, `StepSchedule` following its function,
`ThresholdSwitch` flipping once and staying, a schedule of the wrong length rejected.
The loggers: `DelayedLogger` forwarding one call late and flushing on ``close``,
`ListLogger` and `PrintLogger` thinning. The trainer: a toy linear fit converging,
`TrainerState` deterministic in its key and respecting the trainable filter, the loop
stopping when the loader ends, evaluation every ``eval_every`` steps and at the end,
the schedule state advancing inside the jitted step, the float64 guard. The
evaluation: on exact models (the identity-bijection conjugacy and the exact-chart
autoencoder) every error is zero and the physics scalars are the normal form's, and
the bounding-box grid covers the data. The checkpointer: Orbax save then restore gives
``tree_equal`` on the whole state, the schedule state and the key included, the
best-by-metric and latest steps are kept, and a missing metric is a `KeyError`. The
fixtures are a linear model with its loss and batches, an exact autoencoder built
from a normal form's chart, and batches of windows on the Bautin form.
"""

import io
from contextlib import redirect_stdout

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from deep_isochron.data import (
    generate,
    OnCycleGaussian,
    validation_windows,
    windows,
)
from deep_isochron.model import (
    ConjugateLatentDynamics,
    PhaseAmplitudeAutoencoder,
    PhaseAmplitudeLatentDynamics,
)
from deep_isochron.model.invertible import BiLipschitzLinear
from deep_isochron.systems import BautinNormalForm, HopfNormalForm
from deep_isochron.training import (
    AbstractLoss,
    Checkpointer,
    collect_batches,
    ConjugacyTrajectoryLoss,
    Constant,
    decay_mask,
    DelayedLogger,
    Evaluator,
    ListLogger,
    OrbaxCheckpointer,
    PhaseAutoencoderLoss,
    PrintLogger,
    StepSchedule,
    ThresholdSwitch,
    Trainer,
)
from deep_isochron.training.evaluation import (
    align_phases,
    bounding_box_grid,
    circular_std,
    prediction_sums,
)
from deep_isochron.training.losses import (
    alpha_schedule,
    batch_center_of_mass,
    final_mse,
    step_weighted_consistency,
    trajectory_mse,
    YAWATA_STAGE_1,
    YAWATA_STAGE_2,
    YAWATA_SWITCH_AT,
)
from deep_isochron.training.trainer import check_batch_dtype
from deep_isochron.utils.numerics import wrap_phase

from tests.helpers import assert_close, TOL


# ----------------------------------------------------------------------- fixtures --
class LinearModel(eqx.Module):
    """``y = W x + b`` on the dict batches the trainer expects (``u`` is the input and
    ``t`` the target here, to reuse the batch contract)."""

    w: jax.Array
    b: jax.Array

    def __call__(self, x):
        """``W x + b``."""
        return self.w @ x + self.b


class LinearLoss(AbstractLoss):
    """The mean squared error of `LinearModel`'s output against ``batch["t"]``, as the
    one named term ``"mse"``."""

    weight_names: tuple[str, ...] = eqx.field(static=True, init=False, default=("mse",))
    default_weights: tuple[float, ...] = eqx.field(static=True, default=(1.0,))

    def terms(self, model, batch):
        """The ``"mse"`` term and no auxiliaries."""
        pred = jax.vmap(model)(batch["u"])
        return {"mse": jnp.mean((pred - batch["t"]) ** 2)}, {}


W_TRUE = jnp.array([[2.0, -1.0], [0.5, 3.0]])
B_TRUE = jnp.array([0.3, -0.7])


def linear_batches(n, batch=64, seed=0):
    """`n` batches of Gaussian inputs with targets ``W_TRUE x + B_TRUE``."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.normal(size=(batch, 2))
        yield {"u": x, "t": x @ np.asarray(W_TRUE).T + np.asarray(B_TRUE)}


def linear_model(key=jax.random.key(0)):
    """A `LinearModel` with small random weights."""
    kw, kb = jax.random.split(key)
    return LinearModel(
        0.1 * jax.random.normal(kw, (2, 2)), 0.1 * jax.random.normal(kb, (2,))
    )


def _exact_autoencoder(nf):
    """A `PhaseAmplitudeAutoencoder` whose encoder and decoder are `nf`'s exact
    phase–amplitude chart ``(cos 2πΘ, sin 2πΘ, Ψ)``, with the phase ``Θ`` in turns,
    and whose latent dynamics carry `nf`'s Floquet exponent and the rate of the encoded
    angle ``2πΘ``, which is ``2π/|T| = |ω₁|`` whichever way the form turns."""

    def encode(u):
        phase, psi = nf.to_phase_amplitude(u)
        angle = 2 * jnp.pi * phase
        return jnp.stack((jnp.cos(angle), jnp.sin(angle), psi))

    def decode(y):
        phase = jnp.arctan2(y[1], y[0]) / (2 * jnp.pi)
        return nf.from_phase_amplitude(jnp.stack((phase, y[2])))

    dyn = PhaseAmplitudeLatentDynamics(
        float(jnp.abs(nf.omega())), float(nf.floquet_exponent())
    )
    model = PhaseAmplitudeAutoencoder(2, dyn, mlp_width=2, key=jax.random.key(0))
    return eqx.tree_at(lambda m: (m.encoder, m.decoder), model, (encode, decode))


NF = BautinNormalForm(1.0, 0.5, 2.0, 0.7)


def _nf_batches(nf=NF, n_traj=24, length=9, batch=8, num=2):
    """`num` batches of whole-trajectory windows generated from `nf` near its cycle."""
    src = generate(
        nf,
        OnCycleGaussian.from_normal_form(nf, 100, 0.4),
        jnp.linspace(0, 1.5, length),
        n_traj,
        seed=0,
    )
    return collect_batches(windows(src, length, seed=0).batch(batch), num)


# ------------------------------------------------------------------------ 1. losses --
def test_loss_building_blocks():
    """`trajectory_mse`, `final_mse`, `step_weighted_consistency`, `alpha_schedule` and
    `batch_center_of_mass` give their hand-computed values on small arrays, including
    the ``L ≥ 1`` and ``L → 0`` limits of the schedule."""
    pred = jnp.array([[[1.0, 0.0], [0.0, 2.0]]])  # (1 batch, 2 time, 2 dim)
    target = jnp.zeros((1, 2, 2))
    assert_close(trajectory_mse(pred, target), (1 + 4) / 2)
    assert_close(final_mse(pred, target), 4.0)
    y = jnp.zeros((3, 3, 2))  # batch 3, time 3 (K = 2 steps), dim 2
    y_pred = y.at[:, 1, 0].set(1.0).at[:, 2, 1].set(2.0)
    per_component = step_weighted_consistency(y, y_pred, jnp.array([1.0, 0.5]))
    assert_close(per_component, jnp.array([1.0, 0.5 * 4.0]))
    assert_close(alpha_schedule(3, jnp.asarray(2.0)), 1 / jnp.arange(1, 4))  # L ≥ 1
    assert_close(alpha_schedule(3, jnp.asarray(0.0)), jnp.ones(3))  # L → 0
    assert_close(alpha_schedule(2, jnp.asarray(0.5)), jnp.array([1.0, 2**-0.5]))
    assert_close(batch_center_of_mass(jnp.array([[1.0, 0.0], [-1.0, 0.0]])), 0.0)
    assert_close(batch_center_of_mass(jnp.array([[1.0, 0.0], [1.0, 0.0]])), 1.0)


def test_abstract_loss_weights_named_terms_in_order():
    """The total of a one-term loss is the term at the default weight, three times the
    term at weight 3, and half of it at a default weight of 0.5."""
    loss = LinearLoss()
    batch = {"u": jnp.ones((4, 2)), "t": jnp.zeros((4, 2))}
    model = LinearModel(jnp.eye(2), jnp.zeros(2))
    total, terms = loss(model, batch)  # default weight 1
    assert_close(total, terms["mse"])
    total3, _ = loss(model, batch, jnp.array([3.0]))
    assert_close(total3, 3 * terms["mse"])
    assert LinearLoss(default_weights=(0.5,))(model, batch)[0] == 0.5 * terms["mse"]


def test_project_losses_expose_their_terms_and_vanish_on_exact_models():
    """Both losses are zero (up to the integrator) on the exact models; the terms are
    the named building blocks (the paper-specific semantics of the autoencoder loss are
    pinned in ``test_baseline``)."""
    batches = _nf_batches()
    identity = ConjugateLatentDynamics(
        NF, BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0))
    )
    total, terms = ConjugacyTrajectoryLoss()(identity, batches[0])
    assert set(terms) == {"data", "latent", "final"}
    assert_close(total, 0.0, atol=TOL["flow"])  # the normal form's own flow
    assert_close(terms["latent"], 0.0, atol=TOL["flow"])
    weighted, _ = ConjugacyTrajectoryLoss()(identity, batches[0], jnp.array([1.0, 1.0]))
    assert_close(weighted, terms["data"] + terms["latent"])

    total, terms = PhaseAutoencoderLoss()(_exact_autoencoder(NF), batches[0])
    assert {"recon", "pha", "amp", "aux", "omega", "kappa"} <= set(terms)
    assert_close(total, 2.0 * terms["aux"], rtol=TOL["closed_form"], atol=1e-10)


# --------------------------------------------------------------------- 2. schedules --
def test_schedules():
    """`Constant` returns its weights; `StepSchedule` follows its function of the step;
    `ThresholdSwitch` starts on stage 1, flips to stage 2 once ``pha`` falls below its
    threshold and stays flipped; a wrong-length schedule, and a `Trainer` whose
    schedule length disagrees with its loss, are rejected."""
    c = Constant((1.0, 2.0))
    assert_close(c.weights(c.init(), jnp.asarray(5)), jnp.array([1.0, 2.0]))

    s = StepSchedule(lambda step: jnp.stack([1.0, jnp.minimum(1.0, step / 10.0)]))
    assert_close(s.weights(s.init(), jnp.asarray(5)), jnp.array([1.0, 0.5]))
    assert_close(s.weights(s.init(), jnp.asarray(50)), jnp.array([1.0, 1.0]))

    sw = ThresholdSwitch(YAWATA_STAGE_1, YAWATA_STAGE_2, YAWATA_SWITCH_AT)
    state = sw.init()
    assert not bool(state)
    assert_close(sw.weights(state, jnp.asarray(0)), jnp.asarray(YAWATA_STAGE_1))
    high = {"pha": jnp.asarray(0.5), "aux": jnp.asarray(0.01)}
    state = sw.update(state, jnp.asarray(0), high)
    assert not bool(state)  # pha not yet below threshold
    low = {"pha": jnp.asarray(0.001), "aux": jnp.asarray(0.01)}
    state = sw.update(state, jnp.asarray(1), low)
    assert bool(state)
    assert_close(sw.weights(state, jnp.asarray(2)), jnp.asarray(YAWATA_STAGE_2))
    state = sw.update(state, jnp.asarray(2), high)  # once switched, stays switched
    assert bool(state)
    with pytest.raises(ValueError):
        ThresholdSwitch((1.0,), (1.0, 2.0), {})
    with pytest.raises(ValueError, match="weights"):
        Trainer(optax.sgd(0.1), LinearLoss(), Constant((1.0, 2.0)))


# ----------------------------------------------------------------------- 3. loggers --
def test_delayed_logger_is_one_step_late_and_flushes_on_close():
    """`DelayedLogger` forwards each record on the next `log` call, forwards the last
    one on `close`, and a second `close` forwards nothing."""
    inner = ListLogger()
    log = DelayedLogger(inner)
    log.log({"a": jnp.asarray(1.0)}, 1)
    assert inner.records == []
    log.log({"a": jnp.asarray(2.0)}, 2)
    assert inner.records == [(1, {"a": 1.0})]
    log.close()
    assert inner.records == [(1, {"a": 1.0}), (2, {"a": 2.0})]
    log.close()  # idempotent
    assert len(inner.records) == 2


def test_print_logger_thins_and_formats():
    """`PrintLogger(every=2, keys=("loss",))` prints steps 2 and 4 only, each as
    ``step N | loss: value``."""
    out = io.StringIO()
    with redirect_stdout(out):
        logger = PrintLogger(every=2, keys=("loss",))
        for step in range(1, 5):
            logger.log({"loss": jnp.asarray(step / 10), "other": 1.0}, step)
    lines = out.getvalue().strip().split("\n")
    assert lines == ["step 2 | loss: 0.2", "step 4 | loss: 0.4"]


# ----------------------------------------------------------------------- 4. trainer --
def test_toy_linear_fit_converges_and_logs_every_step():
    """400 Adam steps on the linear fixture recover ``W_TRUE`` and ``B_TRUE`` to 1e-2,
    with one logged record per step and the final loss below 1e-4."""
    trainer = Trainer(optax.adam(0.05), LinearLoss())
    log = ListLogger()
    state = trainer.train(
        linear_model(),
        linear_batches(400),
        num_steps=400,
        key=jax.random.key(0),
        logger=DelayedLogger(log),
    )
    assert int(state.step) == 400
    assert [s for s, _ in log.records] == list(range(1, 401))
    assert log.records[-1][1]["loss"] < 1e-4
    assert log.records[-1][1]["w/mse"] == 1.0
    assert_close(state.model.w, W_TRUE, atol=1e-2)
    assert_close(state.model.b, B_TRUE, atol=1e-2)


def test_trainer_state_is_deterministic_and_respects_the_trainable_filter():
    """Two states from the same key take the same step and advance their training
    key; with nothing trainable the model does not move while the step counts."""
    trainer = Trainer(optax.adam(0.05), LinearLoss())
    batch = {"u": jnp.ones((4, 2)), "t": jnp.zeros((4, 2))}
    s1 = trainer.init(linear_model(), key=jax.random.key(3))
    s2 = trainer.init(linear_model(), key=jax.random.key(3))
    a, _ = trainer.train_step(s1, batch)
    b, _ = trainer.train_step(s2, batch)
    assert eqx.tree_equal(eqx.filter(a, eqx.is_array), eqx.filter(b, eqx.is_array))
    assert not eqx.tree_equal(a.training_key, s1.training_key)

    only_b = Trainer(optax.adam(0.05), LinearLoss(), is_trainable=lambda x: False)
    frozen = only_b.init(linear_model(), key=jax.random.key(0))
    # nothing trainable: the model never moves
    after, _ = only_b.train_step(frozen, batch)
    assert eqx.tree_equal(after.model, frozen.model)
    assert int(after.step) == 1


def test_trainer_runs_to_loader_end_without_num_steps(recwarn):
    """Without `num_steps`, `train` stops when the 5-batch loader ends, without a
    warning."""
    trainer = Trainer(optax.adam(0.05), LinearLoss())
    state = trainer.train(linear_model(), linear_batches(5), key=jax.random.key(0))
    assert int(state.step) == 5
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_trainer_stops_on_exhausted_loader_and_evaluates_on_schedule():
    """A loader of 7 batches under ``num_steps=100`` ends the run at step 7 with a
    warning; `evaluate` is called at steps 3, 6 and the final 7; ``eval_every=0``
    and a missing key are rejected."""
    trainer = Trainer(optax.adam(0.05), LinearLoss())
    calls: list[int] = []

    def evaluate(model):
        calls.append(1)
        return {"val/mse": jnp.asarray(0.5)}

    log = ListLogger()
    with pytest.warns(UserWarning, match="ended after 7 of 100"):
        state = trainer.train(
            linear_model(),
            linear_batches(7),
            num_steps=100,
            key=jax.random.key(0),
            logger=log,
            evaluate=evaluate,
            eval_every=3,
        )
    assert int(state.step) == 7  # the loader ended first
    assert len(calls) == 3  # steps 3, 6 and the final step 7
    eval_steps = [s for s, m in log.records if "val/mse" in m]
    assert eval_steps == [3, 6, 7]
    with pytest.raises(ValueError, match="eval_every"):
        trainer.train(
            linear_model(),
            linear_batches(1),
            num_steps=1,
            key=jax.random.key(0),
            evaluate=evaluate,
            eval_every=0,
        )
    with pytest.raises(ValueError, match="key"):
        trainer.train(linear_model(), linear_batches(1), num_steps=1)


def test_threshold_switch_flips_inside_the_jitted_step():
    """The schedule state advances in ``train_step`` and the logged weights follow."""
    sw = ThresholdSwitch((1.0,), (5.0,), {"mse": 1e-3})
    trainer = Trainer(optax.adam(0.05), LinearLoss(), sw)
    log = ListLogger()
    state = trainer.train(
        linear_model(),
        linear_batches(400),
        num_steps=400,
        key=jax.random.key(0),
        logger=log,
    )
    assert bool(state.schedule_state)
    weights = [m["w/mse"] for _, m in log.records]
    assert weights[0] == 1.0 and weights[-1] == 5.0
    flip = weights.index(5.0)
    assert all(w == 5.0 for w in weights[flip:])  # once and for all


def test_float64_guard():
    """`check_batch_dtype` accepts float64 arrays with x64 on, rejects them with x64
    off naming the dtype, and accepts float32 with x64 off."""
    batch64 = {"u": np.zeros((2, 2)), "t": np.zeros(2)}
    check_batch_dtype(batch64, x64_enabled=True)
    with pytest.raises(TypeError, match="float64"):
        check_batch_dtype(batch64, x64_enabled=False)
    check_batch_dtype({"u": np.zeros((2, 2), np.float32)}, x64_enabled=False)


# -------------------------------------------------------------------- 5. evaluation --
def test_evaluation_on_exact_models_is_exact():
    """On the identity-bijection conjugacy and the exact-chart autoencoder the
    `Evaluator`'s errors are at the integrator's floor, the period, ``kappa`` and the
    normal-form parameters are the reference's, and the conjugacy-only metrics are
    absent for the autoencoder; a wrong frequency is detected, and an empty
    evaluator is rejected."""
    batches = _nf_batches()
    evaluator = Evaluator(batches, reference=NF)
    identity = ConjugateLatentDynamics(
        NF, BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0))
    )
    out = evaluator(identity)
    assert out["val/mse"] < TOL["flow"] and out["val/final_mse"] < TOL["flow"]
    assert out["period"] == pytest.approx(float(NF.period()))
    assert out["kappa"] == pytest.approx(float(NF.floquet_exponent()))
    assert out["nf/a"] == pytest.approx(1.0) and out["nf/b"] == pytest.approx(0.5)
    assert out["roundtrip/max"] < TOL["vector_roundtrip"]
    assert out["jac/sv_min"] == pytest.approx(1.0) == out["jac/sv_max"]
    assert out["phase/circ_std"] < 1e-6 and out["amplitude/corr"] > 1 - 1e-9

    out = evaluator(_exact_autoencoder(NF))
    assert out["val/mse"] < TOL["vector_roundtrip"]
    assert out["period"] == pytest.approx(float(NF.period()))
    assert out["phase/circ_std"] < 1e-6 and out["amplitude/corr"] > 1 - 1e-9
    assert "roundtrip/max" not in out  # conjugacy-only metrics

    # a wrong model is not exact
    wrong = ConjugateLatentDynamics(
        BautinNormalForm(1.0, 0.5, 1.0, 0.7),  # ω₁ = 1 instead of 2
        BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0)),
    )
    assert evaluator(wrong)["val/mse"] > 1e-2

    with pytest.raises(ValueError, match="empty"):
        Evaluator([])(identity)


def test_non_finite_validation_metric_stops_the_run_before_checkpointing():
    """A ``nan`` validation metric raises ``FloatingPointError`` at that evaluation
    and the state is not offered to the checkpointer, so the last checkpoint stays
    finite; ``val/mse_early`` / ``late`` being ``nan`` for an empty bucket does not stop
    a run."""
    batches = _nf_batches()
    model = ConjugateLatentDynamics(
        NF, BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0))
    )
    trainer = Trainer(optax.adam(1e-3), ConjugacyTrajectoryLoss())
    saved: list[int] = []

    class Recorder(Checkpointer):
        def save(self, step, state, metrics, *, force=False):
            saved.append(step)
            return True

    calls = {"n": 0}

    def evaluate(m):
        calls["n"] += 1
        return {"val/mse": float("nan") if calls["n"] >= 2 else 1.0}

    with pytest.raises(FloatingPointError, match="step 4"):
        trainer.train(
            model,
            batches * 4,
            key=jax.random.key(0),
            checkpointer=Recorder(),
            evaluate=evaluate,
            eval_every=2,
        )
    assert saved == [2]
    state = trainer.train(
        model,
        batches * 2,
        key=jax.random.key(0),
        evaluate=lambda m: {"val/mse": 1.0, "val/mse_early": float("nan")},
        eval_every=1,
    )
    assert int(state.step) == len(batches) * 2


def test_latent_extent_reports_where_the_data_are_sent():
    """``latent/r_min|r_max`` are the extremes of ``|H(x)|`` over every validation
    sample, ``latent/amp_min|amp_max`` those of the amplitude, ``latent/nonfinite`` /
    ``nonfinite_image`` the fractions of non-finite amplitudes / images: the detector
    for a map that flings points toward the latent origin or far outside the cycle."""
    batches = _nf_batches()
    starts = np.concatenate([np.asarray(b["u"]).reshape(-1, 2) for b in batches])
    radii = np.linalg.norm(starts, axis=-1)
    identity = ConjugateLatentDynamics(
        NF, BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0))
    )
    out = Evaluator(batches)(identity)
    assert out["latent/r_min"] == pytest.approx(radii.min(), rel=1e-6)
    assert out["latent/r_max"] == pytest.approx(radii.max(), rel=1e-6)
    assert out["latent/amp_min"] == pytest.approx(float(NF.isostable(radii.min())))
    assert out["latent/amp_max"] == pytest.approx(float(NF.isostable(radii.max())))
    assert out["latent/nonfinite"] == 0.0 and out["latent/nonfinite_image"] == 0.0
    # a map that contracts the plane (raw_s → −∞ puts s at its floor 1/L = 0.5) sends
    # the data toward the origin: r_min shrinks and the amplitude plunges
    # (Ψ ~ −r^{−2(1+b)}); the autoencoder path reports the amplitude extremes only
    scaled = eqx.tree_at(lambda m: m.bijection.raw_s, identity, -20.0 * jnp.ones(2))
    r_scaled = float(jnp.linalg.norm(scaled.bijection(jnp.asarray(starts[0]))))
    shrink = r_scaled / radii[0]
    out2 = Evaluator(batches)(scaled)
    assert shrink < 0.6
    assert out2["latent/r_min"] == pytest.approx(radii.min() * shrink, rel=1e-6)
    assert out2["latent/amp_min"] < out["latent/amp_min"]
    ae = Evaluator(batches)(_exact_autoencoder(NF))
    assert "latent/r_min" not in ae and ae["latent/nonfinite"] == 0.0

    # the intended source: a finite, deterministic dataset, re-iterated each call
    src = generate(
        NF,
        OnCycleGaussian.from_normal_form(NF, 100, 0.4),
        jnp.linspace(0, 1.5, 9),
        6,
        seed=0,
    )
    streamed = Evaluator(validation_windows(src, 5).batch(4), reference=NF)
    first, second = streamed(identity), streamed(identity)
    assert first == second and first["val/mse"] < TOL["flow"]
    assert "val/mse_early" not in first  # no t_split: no split


def test_evaluator_splits_the_prediction_error_by_window_start():
    """``t_split`` buckets windows by their start time: counts add up, early / late
    means recombine to the overall mean, an empty bucket is NaN, and (through
    ``prediction_sums``) a model wrong only on late windows is caught only there."""
    L = 4  # 13 times on [0, 1.5]: starts 0, 4, 8 → t = 0, 0.5, 1.0
    src = generate(
        NF,
        OnCycleGaussian.from_normal_form(NF, 100, 0.4),
        jnp.linspace(0, 1.5, 13),
        5,
        seed=0,
    )
    ys = np.asarray(src.ys)
    windows_ds = validation_windows(src, L).batch(5)
    identity = ConjugateLatentDynamics(
        NF, BiLipschitzLinear(dim=2, max_lipschitz=2.0, key=jax.random.key(0))
    )

    # the exact model through the Evaluator: counts, recombination, empty bucket
    split = Evaluator(windows_ds, t_split=0.75)(identity)
    assert split["val/n_early"] == 10 and split["val/n_late"] == 5  # starts 0, 4 | 8
    assert split["val/mse_early"] < TOL["flow"] and split["val/mse_late"] < TOL["flow"]
    none_early = Evaluator(windows_ds, t_split=-1.0)(identity)
    assert np.isnan(none_early["val/mse_early"]) and none_early["val/n_early"] == 0
    assert "val/mse_early" not in Evaluator(windows_ds)(identity)

    # the sums: a model that predicts zero on early windows and one on late ones
    def oracle(t, x0):
        return jnp.zeros((t.shape[0], 2)) + jnp.where(t[0] < 0.75, 0.0, 1.0), None

    total = None
    for batch in windows_ds:
        part = prediction_sums(oracle, batch, jnp.asarray(0.75))
        total = part if total is None else {k: total[k] + part[k] for k in part}
    assert total is not None and total["n"] == 15 and total["n_early"] == 10
    mean_sq = lambda sel: float(np.mean(np.sum(sel**2, -1)))  # noqa: E731
    early = np.stack([ys[:, s : s + L] for s in (0, 4)], 1).reshape(-1, L, 2)
    late = ys[:, 8 : 8 + L]
    assert_close(total["mse_early"] / total["n_early"], mean_sq(early), rtol=1e-9)
    late_mean = (total["mse"] - total["mse_early"]) / (total["n"] - total["n_early"])
    assert_close(late_mean, mean_sq(late - 1.0), rtol=1e-9)
    overall = (10 * mean_sq(early) + 5 * mean_sq(late - 1.0)) / 15
    assert_close(total["mse"] / total["n"], overall, rtol=1e-9)


def test_bounding_box_grid_and_circular_std():
    """`bounding_box_grid` with ``num=3`` spans the points' box with 9 grid points;
    `circular_std` is zero on a constant angle, invariant to ``2π``, and large on
    angles spread around the circle."""
    pts = jnp.array([[0.0, 0.0], [2.0, 4.0]])  # (lo, hi)
    grid = bounding_box_grid(pts, num=3, margin=0.0)
    assert grid.shape == (9, 2)
    assert_close(jnp.min(grid, axis=0), pts[0]) and assert_close(
        jnp.max(grid, 0), pts[1]
    )
    assert_close(circular_std(jnp.full(10, 0.3)), 0.0, atol=1e-7)
    assert_close(circular_std(jnp.full(10, 0.3) + 2 * jnp.pi), 0.0, atol=1e-7)
    assert circular_std(jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)) > 5.0


def test_align_phases_recovers_sense_and_offset_and_ignores_nan():
    """A model phase that is the reference reversed and shifted by 0.7 aligns with
    sense −1, offset 0.7 and a residual at round-off, held as NumPy, with ``nan``
    entries ignored; a given offset is kept and only the sense chosen."""
    reference = jnp.linspace(-3.0, 3.0, 50)
    model = wrap_phase(-reference + 0.7).at[3].set(jnp.nan)
    alignment = align_phases(model, reference)
    assert alignment.sense == -1.0
    assert abs(alignment.offset - 0.7) < 1e-9 and alignment.rms < 1e-9
    assert isinstance(alignment.residual, np.ndarray)
    assert bool(np.isnan(alignment.residual[3]))
    seeded = align_phases(model, reference, offset=0.7)
    assert seeded.sense == -1.0 and seeded.offset == 0.7


# -------------------------------------------------------------------- 6. checkpoint --
def test_orbax_checkpointer_round_trip(tmp_path):
    """Over a 20-step run with ``save_every=5`` the checkpointer keeps the best (10)
    and the latest (20) steps; restoring the latest gives `tree_equal` on the whole
    state and training resumes from step 20; saving without the metric raises
    `KeyError`."""
    sw = ThresholdSwitch((1.0,), (5.0,), {"mse": 1e-3})
    trainer = Trainer(optax.adam(0.05), LinearLoss(), sw)
    ck = OrbaxCheckpointer(tmp_path / "ckpt", save_every=5, metric="val/mse")
    val = iter([3.0, 1.0, 2.0, 2.5])  # best at the second save

    def evaluate(model):
        return {"val/mse": jnp.asarray(next(val))}

    state = trainer.train(
        linear_model(),
        linear_batches(20),
        num_steps=20,
        key=jax.random.key(0),
        checkpointer=ck,
        evaluate=evaluate,
        eval_every=5,
    )
    assert ck.steps == [10, 20]  # best (10) and latest (20)
    template = trainer.init(linear_model(jax.random.key(9)), key=jax.random.key(9))
    restored = ck.restore(template)  # latest
    assert int(restored.step) == 20
    assert eqx.tree_equal(
        eqx.filter(restored, eqx.is_array), eqx.filter(state, eqx.is_array)
    )
    best = ck.restore(template, step=10)
    assert int(best.step) == 10
    with pytest.raises(KeyError):
        ck.save(21, state, {"other": 1.0})
    ck.close()

    # resuming continues from the restored step
    more = trainer.train(restored, linear_batches(3), num_steps=3)
    assert int(more.step) == 23


def test_final_step_is_checkpointed_regardless_of_the_save_policy(tmp_path):
    """``save_every=2`` and a 3-step run: step 2 by policy, step 3 forced at the end;
    the final state is what ``restore()`` (latest) returns. A run whose last step the
    loop already saved is not saved twice."""
    trainer = Trainer(optax.adam(0.05), LinearLoss())
    ck = OrbaxCheckpointer(tmp_path / "a", save_every=2, metric="val/mse")
    evaluate = lambda model: {"val/mse": jnp.asarray(1.0)}  # noqa: E731
    state = trainer.train(
        linear_model(),
        linear_batches(3),
        key=jax.random.key(0),
        checkpointer=ck,
        evaluate=evaluate,
        eval_every=1,
    )
    assert ck.steps == [3] or ck.steps == [2, 3]  # latest kept; best may be 2
    template = trainer.init(linear_model(), key=jax.random.key(1))
    assert int(ck.restore(template).step) == 3
    ck.close()
    ck2 = OrbaxCheckpointer(tmp_path / "b", save_every=2, metric=None)
    trainer.train(
        linear_model(), linear_batches(4), key=jax.random.key(0), checkpointer=ck2
    )
    assert ck2.steps == [4]  # no evaluate: the final step only
    ck2.close()
    assert eqx.tree_equal(state.step, jnp.asarray(3))


def test_trainer_runs_the_phase_autoencoder_pipeline():
    """The Hopf baseline through the trainer (short; the long version is the ``slow``
    test in ``test_baseline``): schedule, evaluation with a reference and logging."""
    nf = HopfNormalForm(1.0, 2.0, 1.0)
    period = float(nf.period())
    src = generate(
        nf,
        OnCycleGaussian.from_normal_form(nf, 200, 0.5),
        jnp.arange(0.0, 2 * period, period / 20),
        32,
        seed=0,
    )
    train_src, val_src = src.split_trajectories(0.25, seed=0)
    evaluator = Evaluator(
        collect_batches(windows(val_src, 11, seed=1).batch(16), 2), reference=nf
    )
    model = PhaseAmplitudeAutoencoder(
        2, PhaseAmplitudeLatentDynamics(1.0, -0.5), mlp_width=16, key=jax.random.key(0)
    )
    trainer = Trainer(
        optax.adam(1e-3),
        PhaseAutoencoderLoss(),
        ThresholdSwitch(YAWATA_STAGE_1, YAWATA_STAGE_2, YAWATA_SWITCH_AT),
    )
    log = ListLogger()
    state = trainer.train(
        model,
        windows(train_src, 11, seed=0).batch(32),
        num_steps=20,
        key=jax.random.key(1),
        logger=DelayedLogger(log),
        evaluate=evaluator,
        eval_every=10,
    )
    assert int(state.step) == 20
    final = log.records[-1][1]
    assert {"val/mse", "period", "kappa", "phase/circ_std", "amplitude/corr"} <= set(
        final
    )
    assert all(np.isfinite(v) for v in final.values())


def test_decay_mask_keeps_weight_decay_off_the_latent_dynamics():
    """A zero-gradient ``adamw`` step shrinks the bijection's parameters by the decay
    and leaves the normal form's untouched; without the mask it shrinks both (and
    ``raw_a → 0`` would drift the Bautin form toward ``a = 1``, ``b = 0``)."""
    model = ConjugateLatentDynamics(
        BautinNormalForm(1.3, 0.4, 2.0, 0.5),
        BiLipschitzLinear(
            dim=2, max_lipschitz=2.0, key=jax.random.key(0), init="rotation"
        ),
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    zero = jax.tree.map(jnp.zeros_like, params)
    for masked in (True, False):
        opt = optax.adamw(1e-2, weight_decay=0.1, mask=decay_mask() if masked else None)
        updates, _ = opt.update(zero, opt.init(params), params)
        new = eqx.apply_updates(params, updates)
        dyn_moved = any(
            bool(jnp.any(a != b))
            for a, b in zip(
                jax.tree.leaves(params.latent_dynamics),
                jax.tree.leaves(new.latent_dynamics),
            )
        )
        bij_moved = bool(jnp.any(new.bijection.raw_U != params.bijection.raw_U))
        assert bij_moved and dyn_moved == (not masked)
    # the mask is a pytree of bools matching the parameters, False exactly under
    # latent_dynamics
    m = decay_mask()(params)
    assert not any(jax.tree.leaves(m.latent_dynamics))
    assert all(jax.tree.leaves(m.bijection))
