"""The AUTO-07p code generator (``deep_isochron.analysis.ode.auto.codegen``,
ADR-0016). Nothing here needs AUTO; the one test that needs a Fortran compiler is
``slow`` and skipped without ``gfortran``.

1. rhs.inc       the printed ``F(i)`` and ``A(i,j)`` are the template's expressions and
                 their exact Jacobian — parsed back from the Fortran text into SymPy
                 (no compiler) and compared at random points; every physical line fits
                 the width; the FitzHugh–Nagumo output equals the golden file (which is
                 the coworker's hand-written ``RHS`` up to term order)
2. indices       the block after the system's slots: period on ``PAR(11)`` for up to
                 ten parameters, contiguous, named; more than ten rejected
3. the generic   every ``PAR(`` reference in the generic file is a named slot that
   file          ``indices.inc`` declares; the two ``INCLUDE`` lines are present
4. write_problem lays down the four files, the ``PAR`` dict and slot lookup, a source
                 hash that is stable; non-planar systems are rejected
5. f2py (slow)   the generated ``RHS`` compiled with gfortran agrees with the JAX
                 ``rhs`` and ``jacfwd(rhs)``
"""

import re
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from deep_isochron.analysis.ode.auto import (
    auto_pars,
    codegen,
    generic_source,
    indices_block,
    internal_indices,
    parameter_map,
    rhs_block,
    write_problem,
)
from deep_isochron.systems import (
    BautinNormalForm,
    FitzhughNagumo,
    HodgekinHuxley,
    HopfNormalForm,
    WinfreeNormalForm,
)
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


GOLDEN = Path(__file__).parent / "golden"
PLANAR = [
    FitzhughNagumo(),
    HopfNormalForm(1.3, 1.5, 0.4),
    BautinNormalForm(1.0, 0.3, 1.5, 0.5),
    WinfreeNormalForm(-0.25, -0.5),
]


# ----------------------------------------------------------------- 1. rhs.inc ----
def _statements(block: str) -> dict[str, str]:
    """``{"F(1)": "<fortran expr>", "A(1,2)": ...}`` with continuations joined."""
    joined = re.sub(r" &\n\s*& ", "", block)
    out = {}
    for line in joined.splitlines():
        m = re.match(r"\s*([FA]\([\d,]+\)) = (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _fortran_to_sympy(expr: str, n_states: int, n_params: int) -> sp.Expr:
    """Parse the printed Fortran back: ``U(i)`` -> ``u_i``, ``PAR(j)`` -> ``p_j``,
    ``1.0d0`` -> ``1.0``; the arithmetic syntax is Python's."""
    text = re.sub(r"(\d)d([+-]?\d)", r"\1e\2", expr)
    text = re.sub(r"U\((\d+)\)", r"u_\1", text)
    text = re.sub(r"PAR\((\d+)\)", r"p_\1", text)
    names = {f"u_{i + 1}": sp.Symbol(f"u_{i + 1}") for i in range(n_states)}
    names |= {f"p_{j + 1}": sp.Symbol(f"p_{j + 1}") for j in range(n_params)}
    return sp.sympify(text, locals=names)


@pytest.mark.parametrize("ode", PLANAR, ids=repr)
@settings(max_examples=10)
@given(data=st.data())
def test_rhs_block_prints_the_field_and_its_jacobian(ode, data):
    sym = ode.symbolic()
    stmts = _statements(rhs_block(sym))
    n = sym.dim
    assert set(stmts) == {f"F({i})" for i in range(1, n + 1)} | {
        f"A({i},{j})" for i in range(1, n + 1) for j in range(1, n + 1)
    }
    point = {s: data.draw(st.floats(-2.0, 2.0)) for s in sym.states}
    values = {s: float(getattr(ode, name)) for name, s in sym.params.items()}
    back = {}
    for i, s in enumerate(sym.states):
        back[sp.Symbol(f"u_{i + 1}")] = point[s]
    for j, s in enumerate(sym.params.values()):
        back[sp.Symbol(f"p_{j + 1}")] = values[s]
    F = sp.Matrix(sym.rhs)
    A = F.jacobian(sym.states)
    for i in range(n):
        got = _fortran_to_sympy(stmts[f"F({i + 1})"], n, len(sym.params)).subs(back)
        assert_close(
            float(got),
            float(F[i].subs(point).subs(values)),
            rtol=TOL["identity"],
            atol=TOL["identity"],
        )
        for j in range(n):
            got = _fortran_to_sympy(
                stmts[f"A({i + 1},{j + 1})"], n, len(sym.params)
            ).subs(back)
            assert_close(
                float(got),
                float(A[i, j].subs(point).subs(values)),
                rtol=TOL["identity"],
                atol=TOL["identity"],
            )


@pytest.mark.parametrize("ode", PLANAR, ids=repr)
def test_rhs_block_lines_fit_the_width(ode):
    lines = rhs_block(ode.symbolic()).splitlines()
    assert all(len(line) <= codegen.LINE_WIDTH for line in lines)
    # a wrapped statement: trailing `&` on the line, leading `&` on the continuation
    for prev, line in zip(lines, lines[1:]):
        assert line.lstrip().startswith("&") == prev.rstrip().endswith("&")


def test_fitzhugh_nagumo_rhs_block_matches_the_golden_file():
    golden = (GOLDEN / "fitzhugh_nagumo_rhs.inc").read_text()
    assert rhs_block(FitzhughNagumo().symbolic()) == golden


# ----------------------------------------------------------------- 2. indices ----
@pytest.mark.parametrize("n", range(0, 11))
def test_internal_indices_follow_the_system_block(n):
    idx = internal_indices(n)
    assert idx.period == 11
    values = [
        idx.period, idx.seg1, idx.seg2, idx.copies, idx.mult, idx.norm, idx.eta,
        *idx.anchor, *idx.base, *idx.point, idx.stability,
    ]  # fmt: skip
    assert values == list(range(11, 11 + len(values)))
    assert idx.last == values[-1]
    constants = idx.fortran_constants()
    assert constants["IP_PERIOD"] == 11 and constants["IP_POINT_Y"] == idx.point[1]
    assert len(constants) == len(values)


def test_more_than_ten_parameters_is_not_supported_yet():
    with pytest.raises(NotImplementedError):
        internal_indices(11)


# ----------------------------------------------------------- 3. generic file -----
def test_generic_file_uses_only_named_slots_that_indices_inc_declares():
    src = generic_source()
    assert "INCLUDE 'rhs.inc'" in src and "INCLUDE 'indices.inc'" in src
    referenced = set(re.findall(r"PAR\((IP_[A-Z0-9_]+)", src))
    assert referenced, "the generic file refers to the pipeline's slots by name"
    assert not re.findall(r"PAR\(\d", src), "no literal slot numbers"
    declared = set(internal_indices(4).fortran_constants())
    assert referenced <= declared, referenced - declared
    # the names appear in indices.inc exactly as declared
    block = indices_block(internal_indices(4))
    for name in declared:
        assert re.search(rf"\b{name} = \d+", block)


# ----------------------------------------------------------- 4. write_problem ----
def test_write_problem_lays_down_the_files(tmp_path):
    ode = FitzhughNagumo()
    problem = write_problem(ode, tmp_path, name="fhn")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "c.fhn", "fhn.f90", "indices.inc", "rhs.inc"
    ]  # fmt: skip
    assert problem.pars == {1: 0.7, 2: 0.8, 3: 3.0, 4: -0.4} == auto_pars(ode)
    assert problem.parameters == parameter_map(ode.symbolic())
    assert problem.slot("c") == 3
    with pytest.raises(KeyError):
        problem.slot("nope")
    constants = (tmp_path / "c.fhn").read_text()
    assert "NDIM = 2" in constants and "NPAR = 36" in constants
    assert "THL =  {11: 0.0, 12: 0.0, 13: 0.0}" in constants
    again = write_problem(ode, tmp_path / "again", name="fhn")
    assert again.fortran_sha == problem.fortran_sha
    other = write_problem(BautinNormalForm(), tmp_path / "bautin")
    assert other.fortran_sha != problem.fortran_sha


def test_non_planar_systems_are_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match="planar"):
        write_problem(HodgekinHuxley(), tmp_path)


# --------------------------------------------------------------- 5. f2py (slow) ----
@pytest.mark.slow
@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
@pytest.mark.parametrize("ode", PLANAR, ids=repr)
def test_compiled_rhs_agrees_with_jax(ode, tmp_path):
    """Compile ``RHS`` alone (the generic file's other routines need AUTO's ``GETP``)
    with the generated include, through ``numpy.f2py``; compare ``F`` and ``A``."""
    import importlib
    import subprocess
    import sys

    problem = write_problem(ode, tmp_path)
    src = tmp_path / "rhs_only.f90"
    src.write_text(
        "      SUBROUTINE RHS(NDIM,U,PAR,F,JAC,A)\n"
        "        IMPLICIT NONE\n"
        "        INTEGER, INTENT(IN) :: NDIM\n"
        "        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(36)\n"
        "        LOGICAL, INTENT(IN) :: JAC\n"
        "        DOUBLE PRECISION, INTENT(OUT) :: F(NDIM), A(NDIM,NDIM)\n"
        "        INCLUDE 'rhs.inc'\n"
        "      END SUBROUTINE RHS\n"
    )
    subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "-c", "-m", "rhs_only", str(src)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("rhs_only")
    finally:
        sys.path.remove(str(tmp_path))
    par = np.zeros(36)
    for i, v in problem.pars.items():
        par[i - 1] = v
    u = np.array([0.5, -0.2])
    f, a = mod.rhs(u, par, True)  # f2py infers NDIM from U and makes F, A outputs
    assert_close(f, ode.rhs(0.0, jnp.asarray(u)))
    assert_close(a, jax.jacfwd(ode.rhs, argnums=1)(0.0, jnp.asarray(u)))
