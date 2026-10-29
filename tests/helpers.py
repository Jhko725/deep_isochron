"""Shared test helpers: the named tolerances ``TOL``, solver configurations
``SOLVERS`` and isochron settings ``ISOCHRON_SETTINGS``, the coworker's stored
FitzHugh–Nagumo isochrons, a foliation of Hopf's cycle built by hand, jitted checkers
for the bijection laws, and an optimizer step along a random direction."""

import logging
import re
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from deep_isochron.analysis.ode import (
    find_limit_cycle,
    Isochron,
    IsochronFailure,
    IsochronFoliation,
    IsochronSettings,
    LimitCycle,
    reanchor,
)
from deep_isochron.systems import FitzhughNagumo, SolverConfig
from deep_isochron.systems.normal_forms import (
    BautinNormalForm,
    HopfNormalForm,
    WinfreeNormalForm,
)
from deep_isochron.utils.numerics import wrap_phase
from jaxtyping import Float


# Tolerances (float64 throughout). Every tolerance in the suite is an entry here, used
# by name, so that a tolerance is a documented decision rather than a per-test literal.
# Numbers that are *laws* (an asymptotic slope bound, a finite-difference step, a
# magnitude range for a strategy) are not tolerances and stay where the law is stated.
TOL = {
    # the same algebra evaluated under jit and eagerly: equal up to summation order
    "jit_eager": 1e-14,
    # exact algebra, round-off only: identity at zero raw parameters, a product
    # assembled from its factors, an oracle (scipy) evaluating the same spline
    "identity": 1e-12,
    # the CubicBSpline inverse at extreme parameters: Newton reaches float64 round trip
    "bspline_inverse": 1e-11,
    # two closed-form evaluations of one quantity: a primitive's inverse∘forward, a
    # parameter-swap inverse, f'(x_k) vs. the stored knot derivative, SVD vs. the
    # constrained singular values, expm(skew) orthogonality, round trip at knots
    "closed_form": 1e-9,
    # one analytic/spline map and its closed-form or 20-step Newton inverse
    "scalar_roundtrip": 1e-8,
    # a coupling/radial/linear layer: inverse plus a solve or a ratio
    "vector_roundtrip": 1e-7,
    # one-sided limits of f, f', f'' at the range endpoints (mismatch ~ h * f^(k+1))
    "join": 1e-7,
    # products of forward-mode Jacobians, 2x2; second derivatives of inverses
    "jacobian": 1e-6,
    # two numerical integrations of the same ODE at tight tolerance (rtol 1e-10), or an
    # integrated closed-form identity (phase, isostable) against the integrator's error
    "flow": 1e-6,
    # analysis (Phase E). ``Cycle`` point queries: a dense search plus one Newton step
    "cycle_query": 1e-6,
    # ``estimate_cycle`` on noise-free normal-form data: the period from interpolated
    # crossing times (relative), the curve and center (absolute), the phase against the
    # true one (peak-to-peak spread of the offset, in turns) — limited by the time grid
    # (100 steps per period) and the truncated density expansion, not by round-off
    "cycle_period": 1e-6,
    "cycle_curve": 2e-3,
    "cycle_phase": 3e-3,
    # ``analysis.ode`` (O1), the vector field integrated at rtol 1e-10 / atol 1e-12
    # (``SHOOTING_SOLVER_CONFIG``; always the solver, never a normal form's closed form,
    # which is the oracle): over one period the global error is ~1e-9, so the period and
    # the orbit against the exact values, and the level-set residual of a designed cycle
    # (Guckenheimer & Meloon's own accuracy measure), are asked to a decade above that;
    # the return map's point and time (event located by Newton at the solver's
    # tolerance) and the distance of a return-map derivative's eigenvalue from its exact
    # value likewise
    "limit_cycle": 1e-8,
    "return_map": 1e-8,
    "trivial_multiplier": 1e-8,
    "level_set": 1e-8,
    "floquet": 1e-8,  # O2: the exponent inherits O1's orbit error
    # O7: an AUTO isochron against the coworker's stored curve, as the distance from
    # points to the other curve's polyline; AUTO's own rerun reproduces the stored
    # curves to 3.6e-6, and the corrected bundle moves them by 8.1e-6
    # (docs/design/isochrons.md §4, decision D3)
    "isochron_curve": 2e-5,
    # O7: the adjoint bundle's branch point, where AUTO locates the multiplier 1 to
    # about 1e-11 (0.99999999998 on Hopf and Bautin)
    "branch_point": 1e-8,
    # O7: AUTO's orbit after stage 2 against O1's flow, and the Hopf route's orbit and
    # period against O1's; AUTO corrects to EPSU = 1e-10 and O1's solver runs at
    # rtol 1e-10, and the measured gaps are 3e-9 (FitzHugh–Nagumo) and 6e-11 (Hopf)
    "auto_orbit": 1e-8,
    # O7: the sine of the angle between the bundle's rotation Jw and O2's Floquet
    # vector, or between w and a closed-form gradient of the phase; measured 1e-11
    "bundle_direction": 1e-8,
    # O7: the spread of w·F along the cycle relative to its mean, constant in exact
    # arithmetic since d(w·F)/dt = 0 for the adjoint; measured 7e-10
    "bundle_scale": 1e-8,
    # O7: a normal form's closed-form phase along a traced isochron, against the
    # requested phase; stage 3 accepts a user point within 1e-8 T of its target
    # (``stages.py``), so the base point's phase is off by up to 1e-8 turns; beyond
    # that the end direction is the stable direction, so the error is O(eta²) plus the
    # collocation error (measured 6e-11 turns)
    "isochron_phase": 1e-8,
    # O7: a traced point flowed five periods, against the base point; a point off its
    # isochron by delta ends |grad Θ| |F| delta away, measured 4e-6 at most on
    # FitzHugh–Nagumo and 1e-9 for most points
    "isochron_forward": 2e-5,
    # O3: the asymptotic phase by forward integration and read-off, against a closed
    # form, the phase along the cycle or the flow law, in turns; the solve at rtol 1e-10
    # over t_end sets the floor, measured 2.4e-10 at most (also the discrepancy and
    # distance)
    "phase": 2e-9,
    # O3: the phase at the coworker's FitzHugh–Nagumo isochron points within
    # GOLDEN_NEAR_CYCLE of the cycle, against the file's phase, in turns; measured 7e-10
    "golden_phase": 1e-8,
}


# Solver configurations used by the tests, by name, with the reason — the counterpart of
# ``TOL`` for ODE solves. ``Tsit5`` is diffrax's recommended non-stiff solver ("now
# reckoned on being slightly more efficient overall" than Dopri5; docs, "How to choose a
# solver"); the Kvaerno family is its recommendation for stiff problems.
SOLVERS = {
    # reference solutions for closed-form comparisons: error far below TOL["flow"]
    "tight": SolverConfig(solver=dfx.Tsit5(), rtol=1e-10, atol=1e-12, max_steps=16384),
    # a second integrator at the same tolerance: the answer is solver-independent
    "tight_stiff": SolverConfig(
        solver=dfx.Kvaerno5(), rtol=1e-10, atol=1e-12, max_steps=16384
    ),
    # long FitzHugh–Nagumo runs (period measurement over ~18 cycles): tight, many steps
    "tight_long": SolverConfig(
        solver=dfx.Tsit5(), rtol=1e-10, atol=1e-12, max_steps=1 << 17
    ),
    # Hodgkin–Huxley spiking: stiff gating kinetics, an implicit solver
    "stiff": SolverConfig(
        solver=dfx.Kvaerno5(), rtol=1e-8, atol=1e-10, max_steps=65536
    ),
    # data-generation tests: tight enough that only the metadata is in question
    "data": SolverConfig(solver=dfx.Tsit5(), rtol=1e-9, atol=1e-11),
}


# AUTO isochron settings used by the tests, by name (decision G3). The step count of a
# half is set by how far its segment reaches, which grows with the number of periods,
# so the quick settings trace one period; a half that runs out of steps is flagged, not
# raised (D7). One FitzHugh–Nagumo phase then takes about half a minute on two cores.
# Every entry has one worker: pytest already runs one test per core, and a pool of
# its own per test oversubscribed the two cores of a session (8.4 min for the tier).
ISOCHRON_SETTINGS = {
    # every push: one period, a small box, a step cap
    "quick": IsochronSettings(
        n_periods=1, box_scale=2.0, max_steps=1500, max_workers=1
    ),
    # the full oracle suite (auto and slow): the coworker's settings, his box of
    # |x| = 10 included (box_scale 5 on FitzHugh–Nagumo), so that the two-way
    # comparison covers most of each stored curve; the default box of 2.5 covered
    # 1243 of the 7124 points of his file 100
    "full": IsochronSettings(box_scale=5.0, max_workers=1),
    # the normal forms against their closed forms: two periods, so that the halves
    # reach from near the phaseless set to the box (one period stays within 0.002 of
    # Winfree's cycle, whose multiplier is about 4e-4)
    "reach": IsochronSettings(
        n_periods=2, box_scale=2.0, max_steps=1500, max_workers=1
    ),
}


# The integration time of the asymptotic phase (O3) before its first reading, by
# system: the escape from the slowest point tested plus enough periods for the
# deviation from the cycle to shrink below rounding.
PHASE_T_END = {
    # the slowest contraction tested is Winfree's, kappa = -1.25: e^{-1.25 * 30} ~ 5e-17
    "normal_forms": 30.0,
    # as configs/isochrons/fhn.yaml: mu ~ 1.7e-12 per period of 11.2
    "fitzhugh_nagumo": 80.0,
}


# The coworker's FitzHugh–Nagumo isochrons (tests/golden/fncMultiSeg_Iso_*.csv), the
# oracle of O7 and O3: file k is the isochron at phase 1 - k/100, in turns, after his
# phase 0, the base point of file 100.
GOLDEN = Path(__file__).parent / "golden"
FHN_ANCHOR = jnp.array([-1.72170711, 0.5])
"""The coworker's phase 0: the base point of their file 100."""
GOLDEN_NEAR_CYCLE = 0.01
"""The distance from the cycle within which a stored isochron point is an oracle of
the phase: farther out, near the source, the stored points' position error of about
1e-5 times the phase gradient reaches a quarter of a turn (measured gaps, in turns:
7e-10 within 0.01, 7e-4 within 0.05, 0.29 within 0.1)."""


PLANAR_SYSTEMS = {
    "fitzhugh_nagumo": (FitzhughNagumo(), [1.0, 0.5], FHN_ANCHOR),
    "hopf": (HopfNormalForm(1.3, 1.5, 0.4), [1.2, 0.1], jnp.array([1.0, 0.0])),
    "bautin": (BautinNormalForm(1.0, 0.3, 1.5, 0.5), [1.1, 0.1], jnp.array([1.0, 0.0])),
    "winfree": (WinfreeNormalForm(-0.25, -0.5), [1.1, 0.1], jnp.array([1.0, 0.0])),
}
"""The planar systems the AUTO tests run on: the system, a point in the basin of its
cycle, and the anchor of phase 0 (the coworker's for FitzHugh–Nagumo, ``(1, 0)`` for
the normal forms, decision 14)."""


def anchored_cycle(name: str) -> LimitCycle:
    """The cycle of ``PLANAR_SYSTEMS[name]`` with phase 0 at its anchor."""
    ode, x0, anchor = PLANAR_SYSTEMS[name]
    return reanchor(find_limit_cycle(ode, jnp.asarray(x0), t_settle=50.0), anchor)


def golden_isochron(k: int) -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "n 2"]]:
    """The base point and the points of the coworker's file `k`."""
    data = np.loadtxt(GOLDEN / f"fncMultiSeg_Iso_{k}.csv")
    return data[0, :2], data[:, 2:]


def golden_phase(k: int) -> float:
    """The phase of the coworker's file `k`, in turns."""
    return float(wrap_phase(1 - k / 100))


class CompileCounter(logging.Handler):
    """Counts the staged programs XLA compiles while ``jax_log_compiles`` is on: the
    guard against a driver that compiles a stage again for every new value (a Python
    float reaching ``filter_jit`` as a static argument, a closure built per call).

    Use as a context manager; `count` is the number of programs compiled inside it.
    A program named after a ``jax.numpy`` or ``jax.lax`` function (``jit(subtract)``,
    ``jit(log)``) is an eager operation, compiled once per dtype, and is not counted:
    host-side arithmetic is ignored and a stage that compiles again is caught. The one
    blind spot is a stage named after such a function, which would be ignored, so stages
    are named otherwise (``settle``, ``polish``, ``stage``).
    ``equinox.debug.assert_max_traces`` and ``get_num_traces`` count the traces of one
    function and have to wrap it inside its jit at definition, and they do not see a
    jit inside diffrax; this counter sees every staged program the process compiles, and
    ``jax_explain_cache_misses`` or ``equinox.debug.announce_transform`` then names the
    argument that missed.
    """

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Count a compile record of jax, except that of an eager jax function."""
        message = record.getMessage()
        if "Compiling" not in message:
            return
        match = re.search(r"Compiling jit\(([^)]*)\)", message)
        if match is None or not (
            hasattr(jnp, match.group(1)) or hasattr(jax.lax, match.group(1))
        ):
            self.count += 1

    def __enter__(self) -> "CompileCounter":
        self.count = 0
        self._previous = jax.config.jax_log_compiles
        jax.config.update("jax_log_compiles", True)
        logging.getLogger("jax").addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        logging.getLogger("jax").removeHandler(self)
        jax.config.update("jax_log_compiles", self._previous)


@eqx.filter_jit
def roundtrip(f, x):
    """``(f(x), f⁻¹(f(x)), f(f⁻¹(x)))`` for a batch ``x``."""
    y = jax.vmap(f)(x)
    return y, jax.vmap(f.inverse)(y), jax.vmap(f)(jax.vmap(f.inverse)(x))


@eqx.filter_jit
def jacobian_dets(f, x):
    """``det Df(x)`` for a batch ``x``."""
    return jax.vmap(lambda p: jnp.linalg.det(f.jacobian(p)))(x)


@eqx.filter_jit
def inverse_jacobian_product(f, x):
    """``Df⁻¹(f(x)) · Df(x)`` for a batch ``x``, the identity for a bijection."""
    J = jax.vmap(f.jacobian)(x)
    Jinv = jax.vmap(lambda p: jax.jacfwd(f.inverse)(p))(jax.vmap(f)(x))
    return Jinv @ J


def assert_close(actual, desired, *, rtol=None, atol=None, msg=""):
    """``np.testing.assert_allclose`` with ``TOL["closed_form"]`` as the default."""
    rtol = TOL["closed_form"] if rtol is None else rtol
    atol = TOL["closed_form"] if atol is None else atol
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, err_msg=msg
    )


def adam_step[M: eqx.Module](module: M, key: jax.Array, lr: float = 1.0) -> M:
    """One Adam step on `module` along a random direction for every trainable leaf.

    A random direction is the stronger test for parameter validity: the constrained
    set must be preserved for any update an optimizer might take, not only for the
    gradient of one loss. ``test_true_gradient_step_decreases_loss`` covers the
    plumbing, that gradients flow through ``constrain``.

    **Arguments:**

    - `module`: The module to step.
    - `key`: The PRNG key for the direction.
    - `lr`: The Adam learning rate.

    **Returns:**

    The stepped module.
    """
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    grads = jax.tree.unflatten(
        treedef,
        [
            jax.random.normal(k, l.shape, l.dtype)
            for k, l in zip(jax.random.split(key, len(leaves)), leaves)
        ],
    )
    opt = optax.adam(lr)
    updates, _ = opt.update(grads, opt.init(params), params)
    return eqx.combine(eqx.apply_updates(params, updates), static)


HAND_FOLIATION_FORM = HopfNormalForm(1.3, 1.5, 1.5)
"""The normal form of `hand_foliation`: Hopf with ``w0 == w``, whose phase shift
``h(r) = (w - w0)/a · ln r`` vanishes, so its isochrons are the radial lines the
foliation is built from, and a model conjugate to it by the identity has the
foliation's phase exactly."""


def hand_foliation() -> IsochronFoliation:
    r"""A foliation of Hopf's cycle built by hand, without AUTO: two isochrons on
    radial lines (this form's isochrons are radial), one with its halves stopped for
    different reasons, and a failure. The cycle's phase 0 is at ``(1, 0)``, the normal
    form's own origin, so the isochrons' phases are the form's: the polar angles of the
    lines over $2\pi$, since the form turns counterclockwise."""
    cycle = find_limit_cycle(
        HAND_FOLIATION_FORM, jnp.array([1.2, 0.1]), t_settle=20.0, m=2
    )
    cycle = reanchor(cycle, jnp.array([1.0, 0.0]))  # phase 0 at (1, 0), as the form's
    isochrons = []
    for angle, n in ((0.0, 5), (2.0, 7)):
        r = np.linspace(0.5, 1.5, n)
        points = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
        isochrons.append(
            Isochron(
                phase=angle / (2 * np.pi),
                base_point=np.array([np.cos(angle), np.sin(angle)]),
                points=points,
                eta=r - 1.0,
                arclength=r - 1.0,
                stop=("eta", "box") if n == 5 else ("max_steps", "box"),
            )
        )
    failure = IsochronFailure(-1.0, "a planted failure", "last line", {"NMX": 10})
    return IsochronFoliation(
        cycle,
        tuple(isochrons),
        (failure,),
        IsochronSettings(n_periods=2),
        {"auto_commit": "abc"},
    )
