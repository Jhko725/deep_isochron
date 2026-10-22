r"""AUTO-07p problem files from a system's symbolic definition (ADR-0016).

The generic Fortran file ``fortran/isochron_multiseg.f90`` (the staged multi-segment
BVP; planar) is kept as plain, hand-maintained Fortran and ``INCLUDE``s two files
written here from ``ode.symbolic()``:

- ``rhs.inc`` — the body of ``SUBROUTINE RHS``: ``F(i)`` and, under ``IF(JAC)``, the
  Jacobian ``A(i,j)``, both printed by SymPy from the same expressions (the adjoint
  stages of ``FUNC`` need ``A`` explicitly, so an exact one matters);
- ``indices.inc`` — ``INTEGER, PARAMETER`` constants for the pipeline's own parameter
  slots, placed after the system's parameters ``PAR(1..n)``.

``write_problem`` lays the equation file, the two includes and a constants file
``c.<name>`` into a work directory and returns an ``AutoProblem`` with everything the
pipeline needs to talk to AUTO about them (the ``PAR`` dict, the named indices, a hash
of the generated source for provenance). AUTO is not needed to run this module; the
tests compare the output against a golden file and, when ``gfortran`` is present,
compile ``rhs.inc`` with ``f2py`` against the JAX ``rhs``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import sympy as sp
from sympy.printing.fortran import FCodePrinter

from ....systems.symbolic import SupportsSymbolic, SymbolicODE


GENERIC_SOURCE = "isochron_multiseg.f90"
"""The generic equation file, shipped as package data under ``fortran/``."""

RESERVED_LOW = 10
"""AUTO's own slots: ``PAR(11)`` is the period under ``IPS = 2`` and the manual
reserves ``PAR(11..)`` for internal use, so the pipeline's block never starts below
``PAR(11)`` even for a system with few parameters; a system with more than ten
parameters pushes the block up (the ``IPS = 2`` stage then needs its period copied —
not supported yet, ``internal_indices`` raises)."""

MIN_NPAR = 36
"""AUTO's default ``NPAR``; never go below it."""


class Idx(NamedTuple):
    """The pipeline's parameter slots, by name (the Fortran file sees them as
    ``IP_<NAME>`` through ``indices.inc``; the Python side uses the attributes).

    **Attributes:**

    - `period`: Period of the periodic orbit, ``IP_PERIOD``.
    - `seg1`, `seg2`: Integration times of the two segments, ``IP_SEG1``, ``IP_SEG2``.
    - `copies`: Integration time of the isochron segment (``n_periods`` periods),
        ``IP_COPIES``.
    - `mult`: Multiplier of the ``w' = Aᵀw`` bundle, ``IP_MULT`` (branch point at 1).
    - `norm`: Squared norm of ``w`` at the junction, ``IP_NORM``.
    - `eta`: Distance along the stable direction, the continuation parameter of an
        isochron, ``IP_ETA``.
    - `anchor`: ``(x, y)`` slots anchoring the junction, ``IP_ANCHOR_X/Y``.
    - `base`: ``(x, y)`` of the junction point on the cycle, ``IP_BASE_X/Y``.
    - `point`: ``(x, y)`` of the isochron point traced out, ``IP_POINT_X/Y``.
    - `stability`: AUTO's stability flag (``GETP('STA')``), ``IP_STABILITY``.
    - `last`: The highest slot used.
    """

    period: int
    seg1: int
    seg2: int
    copies: int
    mult: int
    norm: int
    eta: int
    anchor: tuple[int, int]
    base: tuple[int, int]
    point: tuple[int, int]
    stability: int
    last: int

    def fortran_constants(self) -> dict[str, int]:
        """``{"IP_PERIOD": 11, ...}``, what ``indices.inc`` declares."""
        out: dict[str, int] = {}
        for name, value in self._asdict().items():
            if name == "last":
                continue
            if isinstance(value, tuple):
                out[f"IP_{name.upper()}_X"], out[f"IP_{name.upper()}_Y"] = value
            else:
                out[f"IP_{name.upper()}"] = value
        return out


def internal_indices(n_params: int) -> Idx:
    """The slots for a system with `n_params` parameters: a contiguous block after
    ``max(n_params, RESERVED_LOW)``. For ``n_params <= 10`` the period lands on
    ``PAR(11)``, as AUTO's ``IPS = 2`` stage requires.

    **Raises:** `NotImplementedError` for more than ten parameters (the period would
    not be ``PAR(11)``; see `RESERVED_LOW`)."""
    if n_params > RESERVED_LOW:
        raise NotImplementedError(
            f"{n_params} parameters: the pipeline's period slot would not be PAR(11)."
        )
    b = max(n_params, RESERVED_LOW)
    return Idx(
        period=b + 1,
        seg1=b + 2,
        seg2=b + 3,
        copies=b + 4,
        mult=b + 5,
        norm=b + 6,
        eta=b + 7,
        anchor=(b + 8, b + 9),
        base=(b + 10, b + 11),
        point=(b + 12, b + 13),
        stability=b + 14,
        last=b + 14,
    )


def parameter_map(sym: SymbolicODE) -> dict[int, str]:
    """AUTO slot -> parameter name: ``PAR(1..n)`` in the template's order."""
    return {i + 1: name for i, name in enumerate(sym.params)}


def auto_pars(ode: SupportsSymbolic) -> dict[int, float]:
    """The ``PAR={...}`` dict for this instance, from its current parameter values."""
    return {
        i: float(getattr(ode, name))
        for i, name in parameter_map(ode.symbolic()).items()
    }


LINE_WIDTH = 100
"""Free-form Fortran allows 132 characters per line; expressions are wrapped below
that with a trailing ``&`` *and* a leading ``&`` on the continuation line, which makes
the split point irrelevant (a token may be cut). SymPy's own wrapper omits the leading
``&`` and may cut ``**`` into ``* *``, so it is not used."""


def _printer() -> FCodePrinter:
    return FCodePrinter({"source_format": "free", "standard": 95, "precision": 17})


def _wrap(text: str, indent: str) -> str:
    """Split one statement into continuation lines of at most `LINE_WIDTH`."""
    cont = indent + "   & "
    first, rest = LINE_WIDTH - 2, LINE_WIDTH - 2 - len(cont)  # room for the " &"
    chunks = []
    width = first
    while len(text) > width:
        chunks.append(text[:width])
        text, width = text[width:], rest
    chunks.append(text)
    return (" &\n" + cont).join(chunks)


def rhs_block(sym: SymbolicODE, indent: str = "        ") -> str:
    """The body of ``SUBROUTINE RHS``: ``F(i) = ...`` and, under ``IF(JAC)``, the exact
    Jacobian ``A(i,j) = ...``, with states as ``U(i)`` and parameters as ``PAR(j)``."""
    n = sym.dim
    to_auto = {s: sp.Symbol(f"U({i + 1})") for i, s in enumerate(sym.states)}
    to_auto |= {
        s: sp.Symbol(f"PAR({i})")
        for i, s in zip(parameter_map(sym), sym.params.values())
    }
    F = sp.Matrix(sym.rhs)
    A = F.jacobian(sym.states)
    printer = _printer()
    # `_print` gives the unwrapped expression; `doprint` would wrap it unsafely
    code = lambda e: printer._print(e.subs(to_auto))  # noqa: E731, SLF001
    lines = [_wrap(f"{indent}F({i + 1}) = {code(F[i])}", indent) for i in range(n)]
    lines.append(f"{indent}IF(JAC)THEN")
    for i in range(n):
        for j in range(n):
            lines.append(
                _wrap(f"{indent}   A({i + 1},{j + 1}) = {code(A[i, j])}", indent)
            )
    lines.append(f"{indent}ENDIF")
    return "\n".join(lines) + "\n"


def indices_block(idx: Idx, indent: str = "        ") -> str:
    """``indices.inc``: one ``INTEGER, PARAMETER`` per slot."""
    decls = ", ".join(f"{k} = {v}" for k, v in idx.fortran_constants().items())
    return f"{indent}INTEGER, PARAMETER :: {decls}\n"


def constants_file(sym: SymbolicODE, idx: Idx) -> str:
    """The base AUTO constants ``c.<name>``: the discretization and tolerances the
    coworker's reproduction used (NTST 300, NCOL 5, EPS 1e-10), ``NDIM`` from the
    system, ``NPAR`` covering the block, ``THL`` zero-weighting the time slots. Run
    specific constants (``IPS``, ``ICP``, ``UZSTOP`` ...) are passed per ``run``."""
    thl = ", ".join(f"{k}: 0.0" for k in (idx.period, idx.seg1, idx.seg2))
    return (
        "IPS  =   1, IRS =   0, ILP =   1\n"
        f"NDIM = {sym.dim}\n"
        "NTST = 300, NCOL=   5, IAD =   3, ISP =   2, ISW = 1, IPLT= 0, "
        "NBC= 0, NINT= 0\n"
        "NMX  = 1000, NPR= 100, MXBF=   0, IID =   2, ITMX= 10, ITNW= 5, "
        "NWTN= 3, JAC= 0\n"
        "EPSL = 1e-10, EPSU = 1e-10, EPSS = 1e-9\n"
        "DS   = 1.0, DSMIN= 1e-13, DSMAX= 1e-1, IADS=   1\n"
        f"NPAR = {max(idx.last, MIN_NPAR)}, THL =  {{{thl}}}, THU =  {{}}\n"
    )


def generic_source() -> str:
    """The text of the generic equation file (package data)."""
    return files(__package__).joinpath("fortran", GENERIC_SOURCE).read_text()


@dataclass(frozen=True)
class AutoProblem:
    """What `write_problem` laid down, and how the pipeline refers to it.

    **Attributes:**

    - `name`: The equation name (``run(e=name, c=name)``); files are ``<name>.f90``,
        ``c.<name>``, ``rhs.inc``, ``indices.inc`` in `workdir`.
    - `workdir`: Where the files are; AUTO must run with it as the working directory
        (``INCLUDE`` is resolved there).
    - `idx`: The named parameter slots.
    - `pars`: The ``PAR`` dict for the instance the problem was written for.
    - `parameters`: AUTO slot -> parameter name.
    - `fortran_sha`: SHA-256 of the three Fortran sources together (provenance).
    """

    name: str
    workdir: Path
    idx: Idx
    pars: dict[int, float]
    parameters: dict[int, str]
    fortran_sha: str

    def slot(self, parameter: str) -> int:
        """The AUTO slot of a system parameter by name (``slot("c")`` -> ``3``)."""
        for i, name in self.parameters.items():
            if name == parameter:
                return i
        raise KeyError(parameter)


def write_problem(
    ode: SupportsSymbolic, workdir: str | Path, name: str = "isochron"
) -> AutoProblem:
    """Write the AUTO problem for `ode` into `workdir` (created if needed).

    **Raises:** `NotImplementedError` for a system that is not planar (the generic
    file's segment slicing and the stable-direction trick assume ``dim == 2``), or
    has more than ten parameters."""
    sym = ode.symbolic().validate()
    if sym.dim != 2:
        raise NotImplementedError(
            f"the multi-segment isochron BVP is planar; {type(ode).__name__} has dim "
            f"{sym.dim}."
        )
    idx = internal_indices(len(sym.params))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    texts = {
        f"{name}.f90": generic_source(),
        "rhs.inc": rhs_block(sym),
        "indices.inc": indices_block(idx),
        f"c.{name}": constants_file(sym, idx),
    }
    for filename, text in texts.items():
        (workdir / filename).write_text(text)
    sha = hashlib.sha256()
    for key in (f"{name}.f90", "rhs.inc", "indices.inc"):
        sha.update(texts[key].encode())
    return AutoProblem(
        name=name,
        workdir=workdir,
        idx=idx,
        pars=auto_pars(ode),
        parameters=parameter_map(sym),
        fortran_sha=sha.hexdigest(),
    )


__all__ = [
    "AutoProblem",
    "GENERIC_SOURCE",
    "Idx",
    "auto_pars",
    "constants_file",
    "generic_source",
    "indices_block",
    "internal_indices",
    "parameter_map",
    "rhs_block",
    "write_problem",
]
