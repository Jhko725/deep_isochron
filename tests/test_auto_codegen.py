"""The AUTO-07p code generator (``deep_isochron.analysis.ode.auto.codegen``,
ADR-0016). Nothing here needs AUTO; the one test that needs a Fortran compiler is
``slow`` and skipped without ``gfortran``.

Section 1 tests ``rhs.inc``: the printed ``F(i)`` and ``A(i,j)`` are the template's
expressions and their exact Jacobian, parsed back from the Fortran text into SymPy
and compared at random points; every physical line fits the width; the
FitzHugh–Nagumo output equals the golden file, which is the coworker's hand-written
``RHS`` up to term order. Section 2 tests the index block after the system's slots:
the period on ``PAR(11)`` for up to ten parameters, contiguous and named, more than
ten rejected. Section 3 checks that every ``PAR(`` reference in the generic file is a
named slot that ``indices.inc`` declares, and that the bundle integrates the adjoint
equation. Section 4 tests `write_problem`: the four files, the ``PAR`` dict and slot
lookup, a stable source hash, non-planar systems rejected. Section 5 (slow, needs
``gfortran``) compiles the generic file with each system's includes, and compiles a
driver around ``rhs.inc`` whose ``F`` and ``A`` are compared with the JAX ``rhs`` and
``jacfwd(rhs)``.
"""

import re
import shutil
import subprocess
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
    """``{"F(1)": "<fortran expr>", "A(1,2)": ...}`` with continuations joined by a
    blank, which is only correct if `rhs_block` never cuts inside a token."""
    joined = re.sub(r"\s*&\n\s*&\s*", " ", block)
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
    """`rhs_block` prints every ``F(i)`` and ``A(i,j)``, and each, parsed back into
    SymPy, equals the template's expression and Jacobian entry at a drawn point."""
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
    """Every line of `rhs_block` is at most ``codegen.LINE_WIDTH`` wide, a
    continuation line follows exactly the lines that end in ``&``, and every cut is
    at a blank: ``" &"`` ends the line and ``"& "`` starts the next."""
    lines = rhs_block(ode.symbolic()).splitlines()
    assert all(len(line) <= codegen.LINE_WIDTH for line in lines)
    for prev, line in zip(lines, lines[1:]):
        assert line.lstrip().startswith("&") == prev.rstrip().endswith("&")
        if prev.endswith("&"):
            assert prev.endswith(" &") and line.lstrip().startswith("& ")


def test_long_statements_are_cut_between_tokens():
    """A statement far longer than the width is cut only at blanks, so that joining
    the pieces with one blank gives the statement back."""
    x = sp.Symbol("x")
    expr = sum(sp.Symbol(f"PAR({i})") * x**i for i in range(1, 40))
    text = "        F(1) = " + codegen._printer()._print(expr)  # noqa: SLF001
    wrapped = codegen._wrap(text, "        ")  # noqa: SLF001
    assert len(wrapped.splitlines()) > 2
    assert re.sub(r" &\n\s*& ", " ", wrapped) == text


def test_indices_block_has_one_declaration_per_line():
    """`indices_block` declares one slot per line, so it stays far below Fortran's
    132-character limit."""
    lines = indices_block(internal_indices(4)).splitlines()
    assert len(lines) == len(internal_indices(4).fortran_constants())
    assert all(len(line) <= codegen.LINE_WIDTH for line in lines)


def test_fitzhugh_nagumo_rhs_block_matches_the_golden_file():
    """`rhs_block` for FitzHugh–Nagumo equals ``tests/golden/fitzhugh_nagumo_rhs.inc``
    byte for byte."""
    golden = (GOLDEN / "fitzhugh_nagumo_rhs.inc").read_text()
    assert rhs_block(FitzhughNagumo().symbolic()) == golden


# ----------------------------------------------------------------- 2. indices ----
@pytest.mark.parametrize("n", range(0, 11))
def test_internal_indices_follow_the_system_block(n):
    """For ``n`` system parameters the internal slots start at ``PAR(11)`` and are
    contiguous, ``last`` is the final one, and `fortran_constants` names each."""
    idx = internal_indices(n)
    assert idx.period == 11
    values = [
        idx.period, idx.seg1, idx.seg2, idx.copies, idx.mult, idx.norm, idx.eta,
        *idx.section_point, *idx.section_normal, *idx.base, *idx.point,
    ]  # fmt: skip
    assert values == list(range(11, 11 + len(values)))
    assert idx.last == values[-1]
    constants = idx.fortran_constants()
    assert constants["IP_PERIOD"] == 11 and constants["IP_POINT_Y"] == idx.point[1]
    assert constants["IP_SECTION_NORMAL_X"] == idx.section_normal[0]
    assert len(constants) == len(values)


def test_more_than_ten_parameters_is_not_supported_yet():
    """`internal_indices` raises `NotImplementedError` for eleven parameters."""
    with pytest.raises(NotImplementedError):
        internal_indices(11)


# ----------------------------------------------------------- 3. generic file -----
def test_generic_file_uses_only_named_slots_that_indices_inc_declares():
    """The generic file includes both ``.inc`` files, refers to ``PAR`` only through
    ``IP_*`` names, and every name is declared in `indices_block`."""
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


def test_bundle_integrates_the_adjoint_equation():
    """Both bundle segments are ``w' = -τ Aᵀ w``, the adjoint equation, whose
    multiplier-1 solution is the phase sensitivity (`docs/design/isochrons.md` §3.1);
    the coworker's file had the sign ``+``."""
    src = generic_source()
    assert "F(5:6)  = -PAR(IP_SEG1)*(MATMUL(TRANSPOSE(A),U(5:6)))" in src
    assert "F(7:8)  = -PAR(IP_SEG2)*(MATMUL(TRANSPOSE(B),U(7:8)))" in src


# ----------------------------------------------------------- 4. write_problem ----
def test_write_problem_lays_down_the_files(tmp_path):
    """`write_problem` writes the four files, returns the ``PAR`` dict, parameter map
    and slot lookup, puts ``NDIM``, ``NPAR`` and ``THL`` in the constants file, and
    gives a source hash that is stable across runs and differs between systems."""
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
    """`write_problem` raises `NotImplementedError` for Hodgkin–Huxley."""
    with pytest.raises(NotImplementedError, match="planar"):
        write_problem(HodgekinHuxley(), tmp_path)


# ------------------------------------------------------------ 5. gfortran (slow) ----
DRIVER = """\
      PROGRAM DRIVER
        IMPLICIT NONE
        DOUBLE PRECISION U(2), PAR(36), F(2), A(2,2)
        READ(*,*) U, PAR
        CALL RHS(2, U, PAR, F, .TRUE., A)
        WRITE(*,'(ES25.17E3)') F, A
      CONTAINS
        SUBROUTINE RHS(NDIM,U,PAR,F,JAC,A)
          INTEGER, INTENT(IN) :: NDIM
          DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
          LOGICAL, INTENT(IN) :: JAC
          DOUBLE PRECISION, INTENT(OUT) :: F(NDIM), A(NDIM,NDIM)
          INCLUDE 'rhs.inc'
        END SUBROUTINE RHS
      END PROGRAM DRIVER
"""
"""A program around ``rhs.inc`` that reads ``U`` and ``PAR`` and prints ``F`` and
``A`` (column-major) to 17 significant digits, which round-trips a double."""

needs_gfortran = pytest.mark.skipif(
    shutil.which("gfortran") is None, reason="needs gfortran"
)


@pytest.mark.slow
@needs_gfortran
@pytest.mark.parametrize("ode", PLANAR, ids=repr)
def test_generic_file_compiles_with_the_generated_includes(ode, tmp_path):
    """``gfortran -c`` compiles the generic file with the system's ``rhs.inc`` and
    ``indices.inc``; ``-c`` does not link, so AUTO's ``GETP`` is not needed."""
    write_problem(ode, tmp_path)
    subprocess.run(
        ["gfortran", "-c", "isochron.f90", "-o", "isochron.o"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


@pytest.mark.slow
@needs_gfortran
@pytest.mark.parametrize("ode", PLANAR, ids=repr)
def test_compiled_rhs_agrees_with_jax(ode, tmp_path):
    """The driver, compiled with ``gfortran -I`` against the generated ``rhs.inc``,
    gives ``F`` and ``A`` equal to ``rhs`` and ``jacfwd(rhs)`` at several points."""
    problem = write_problem(ode, tmp_path)
    (tmp_path / "driver.f90").write_text(DRIVER)
    subprocess.run(
        ["gfortran", "-I", str(tmp_path), "-o", "driver", "driver.f90"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    par = np.zeros(36)
    for i, v in problem.pars.items():
        par[i - 1] = v
    jac = jax.jacfwd(ode.rhs, argnums=1)
    for u in ([0.5, -0.2], [1.1, 0.3], [-0.7, 0.9], [0.05, -1.3]):
        stdin = " ".join(repr(float(v)) for v in (*u, *par))
        out = subprocess.run(
            [str(tmp_path / "driver")],
            input=stdin,
            text=True,
            check=True,
            capture_output=True,
        ).stdout.split()
        values = np.array([float(v) for v in out])
        f, a = values[:2], values[2:].reshape(2, 2, order="F")
        x = jnp.asarray(u)
        assert_close(f, ode.rhs(0.0, x), rtol=TOL["identity"], atol=TOL["identity"])
        assert_close(a, jac(0.0, x), rtol=TOL["identity"], atol=TOL["identity"])
