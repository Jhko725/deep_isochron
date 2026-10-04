"""The training package (roadmap C1–C5, ADR-0009).

1. losses      building blocks on known inputs; ``AbstractLoss`` weights the named terms
               in order; the two project losses expose their terms; the autoencoder loss
               vanishes on the exact chart (as in ``test_baseline``) and reproduces the
               paper's ``α_k``.
2. schedules   ``Constant``; ``StepSchedule`` follows its function; ``ThresholdSwitch``
               flips once and stays; a schedule of the wrong length is rejected.
3. loggers     ``DelayedLogger`` forwards one call late and flushes on ``close``;
               ``ListLogger``/``PrintLogger`` thinning.
4. trainer     a toy linear fit converges; ``TrainerState`` is deterministic in its key
               and respects the trainable filter; the loop stops on ``StopIteration``;
               evaluation is called every ``eval_every`` and at the end; the schedule
               state advances inside the jitted step; the float64 guard.
5. evaluation  on exact models (identity-bijection conjugacy; exact-chart autoencoder)
               every error is zero and the physics scalars are the normal form's; the
               bounding-box grid covers the data.
6. checkpoint  Orbax: save → restore gives ``tree_equal`` on the whole state (including
               the schedule state and the key); the best-by-metric and latest steps are
               kept; a missing metric is a ``KeyError``.

Run: ``uv run pytest tests/test_training.py``.
"""

import io
from contextlib import redirect_stdout

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from deep_isochron.data import generate, OnCycleGaussian, windows
from deep_isochron.model import (
    ConjugateLatentDynamics,
    PhaseAmplitudeAutoencoder,
    PhaseAmplitudeLatentDynamics,
)
from deep_isochron.model.invertible import BiLipschitzLinear
from deep_isochron.systems import BautinNormalForm, HopfNormalForm
from deep_isochron.training import (
    AbstractLoss,
    collect_batches,
    ConjugacyTrajectoryLoss,
    Constant,
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
from deep_isochron.training.evaluation import bounding_box_grid, circular_std
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

from tests.helpers import assert_close, TOL


# ----------------------------------------------------------------------- fixtures --
class LinearModel(eqx.Module):
    """``y = W x + b`` on the dict batches the trainer expects (``u`` is the input and
    ``t`` the target here, to reuse the batch contract)."""

    w: jax.Array
    b: jax.Array

    def __call__(self, x):
        return self.w @ x + self.b


class LinearLoss(AbstractLoss):
    weight_names: tuple[str, ...] = eqx.field(static=True, init=False, default=("mse",))
    default_weights: tuple[float, ...] = eqx.field(static=True, default=(1.0,))

    def terms(self, model, batch):
        pred = jax.vmap(model)(batch["u"])
        return {"mse": jnp.mean((pred - batch["t"]) ** 2)}


W_TRUE = jnp.array([[2.0, -1.0], [0.5, 3.0]])
B_TRUE = jnp.array([0.3, -0.7])


def linear_batches(n, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.normal(size=(batch, 2))
        yield {"u": x, "t": x @ np.asarray(W_TRUE).T + np.asarray(B_TRUE)}


def linear_model(key=jax.random.key(0)):
    kw, kb = jax.random.split(key)
    return LinearModel(
        0.1 * jax.random.normal(kw, (2, 2)), 0.1 * jax.random.normal(kb, (2,))
    )


def _exact_autoencoder(nf):
    def encode(u):
        theta, psi = nf.to_phase_amplitude(u)
        return jnp.stack((jnp.cos(theta), jnp.sin(theta), psi))

    def decode(y):
        return nf.from_phase_amplitude(jnp.stack((jnp.arctan2(y[1], y[0]), y[2])))

    dyn = PhaseAmplitudeLatentDynamics(float(nf.omega()), float(nf.floquet_exponent()))
    model = PhaseAmplitudeAutoencoder(2, dyn, mlp_width=2, key=jax.random.key(0))
    return eqx.tree_at(lambda m: (m.encoder, m.decoder), model, (encode, decode))


NF = BautinNormalForm(1.0, 0.5, 2.0, 0.7)


def _nf_batches(nf=NF, n_traj=24, length=9, batch=8, num=2):
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
    out = io.StringIO()
    with redirect_stdout(out):
        logger = PrintLogger(every=2, keys=("loss",))
        for step in range(1, 5):
            logger.log({"loss": jnp.asarray(step / 10), "other": 1.0}, step)
    lines = out.getvalue().strip().split("\n")
    assert lines == ["step 2 | loss: 0.2", "step 4 | loss: 0.4"]


# ----------------------------------------------------------------------- 4. trainer --
def test_toy_linear_fit_converges_and_logs_every_step():
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


def test_trainer_stops_on_exhausted_loader_and_evaluates_on_schedule():
    trainer = Trainer(optax.adam(0.05), LinearLoss())
    calls: list[int] = []

    def evaluate(model):
        calls.append(1)
        return {"val/mse": jnp.asarray(0.5)}

    log = ListLogger()
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
    with pytest.raises(ValueError, match="go together"):
        trainer.train(
            linear_model(),
            linear_batches(1),
            num_steps=1,
            key=jax.random.key(0),
            evaluate=evaluate,
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
    batch64 = {"u": np.zeros((2, 2)), "t": np.zeros(2)}
    check_batch_dtype(batch64, x64_enabled=True)
    with pytest.raises(TypeError, match="float64"):
        check_batch_dtype(batch64, x64_enabled=False)
    check_batch_dtype({"u": np.zeros((2, 2), np.float32)}, x64_enabled=False)


# -------------------------------------------------------------------- 5. evaluation --
def test_evaluation_on_exact_models_is_exact():
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

    with pytest.raises(ValueError):
        Evaluator([])


def test_bounding_box_grid_and_circular_std():
    pts = jnp.array([[0.0, 0.0], [2.0, 4.0]])
    grid = bounding_box_grid(pts, num=3, margin=0.0)
    assert grid.shape == (9, 2)
    assert_close(jnp.min(grid, axis=0), pts[0]) and assert_close(
        jnp.max(grid, 0), pts[1]
    )
    assert_close(circular_std(jnp.full(10, 0.3)), 0.0, atol=1e-7)
    assert_close(circular_std(jnp.full(10, 0.3) + 2 * jnp.pi), 0.0, atol=1e-7)
    assert circular_std(jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)) > 5.0


# -------------------------------------------------------------------- 6. checkpoint --
def test_orbax_checkpointer_round_trip(tmp_path):
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
