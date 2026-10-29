"""Finding and loading AUTO-07p (``deep_isochron.analysis.ode.auto._env``).

Section 1 tests how `auto_dir` resolves the tree from the environment and refuses a
missing one; it needs no AUTO. Section 2 (marker ``auto``, skipped without a built
tree) loads AUTO and runs the generated FitzHugh–Nagumo problem: the equilibrium
continuation in ``c`` must find the Hopf point where the Jacobian's trace vanishes,
which is computed here independently from the cubic for the equilibrium.
"""

from pathlib import Path

import numpy as np
import pytest
from deep_isochron.analysis.ode.auto import _env, write_problem
from deep_isochron.systems import FitzhughNagumo


# ----------------------------------------------------------- 1. resolving ----
def _fake_tree(root: Path) -> Path:
    """A directory that passes `auto_dir`'s check: ``cmds/cmds.make`` and an object
    file in ``lib/``."""
    (root / "cmds").mkdir(parents=True)
    (root / "cmds" / "cmds.make").write_text("")
    (root / "lib").mkdir()
    (root / "lib" / "main.o").write_bytes(b"")
    return root


def test_missing_tree_raises_and_names_the_install_script(tmp_path, monkeypatch):
    """`auto_dir` raises `AutoNotInstalled` for a directory without a build, and the
    message names ``scripts/install_auto.sh``."""
    monkeypatch.setenv("DEEP_ISOCHRON_AUTO_DIR", str(tmp_path))
    with pytest.raises(_env.AutoNotInstalled, match="scripts/install_auto.sh"):
        _env.auto_dir()


def test_variables_then_default(tmp_path, monkeypatch):
    """``DEEP_ISOCHRON_AUTO_DIR`` is consulted before ``AUTO_DIR``, and `DEFAULT_DIR`
    when neither is set."""
    ours, theirs = (_fake_tree(tmp_path / name) for name in ("ours", "theirs"))
    monkeypatch.setenv("DEEP_ISOCHRON_AUTO_DIR", str(ours))
    monkeypatch.setenv("AUTO_DIR", str(theirs))
    assert _env.auto_dir() == ours
    monkeypatch.delenv("DEEP_ISOCHRON_AUTO_DIR")
    assert _env.auto_dir() == theirs
    monkeypatch.delenv("AUTO_DIR")
    monkeypatch.setattr(_env, "DEFAULT_DIR", tmp_path / "nowhere")
    with pytest.raises(_env.AutoNotInstalled, match="nowhere"):
        _env.auto_dir()


def test_the_python_layer_is_pinned_by_commit():
    """The installed ``auto-07p`` comes from git at a 40-character commit, the one that
    ``scripts/install_auto.sh`` builds."""
    commit = _env.pinned_commit()
    assert commit is not None and len(commit) == 40


# ----------------------------------------------------------- 2. AUTO (auto) ----
def _fitzhugh_nagumo_hopf_c(a: float, b: float, z: float) -> float:
    """The ``c`` of FitzHugh–Nagumo's Hopf point. The equilibrium solves
    ``x - a + b(x³/3 - x - z) = 0``, independent of ``c``, and the Jacobian's trace
    ``c(1 - x²) - b/c`` vanishes at ``c² = b / (1 - x²)``."""
    roots = np.roots([b / 3, 0.0, 1 - b, -a - b * z])
    x = float(roots[np.argmin(abs(roots.imag))].real)
    return float(np.sqrt(b / (1 - x**2)))


@pytest.mark.auto
def test_fitzhugh_nagumo_problem_finds_its_hopf_point(auto, tmp_path, monkeypatch):
    """The generated FitzHugh–Nagumo problem, built and run by AUTO, continues the
    equilibrium in ``c`` from 3 and stops at a Hopf point where the trace of the
    Jacobian vanishes."""
    ode = FitzhughNagumo()
    problem = write_problem(ode, tmp_path, name="fhn")
    monkeypatch.chdir(tmp_path)
    c = problem.slot("c")
    branch = auto.commands.run(
        [0.9066, -0.2582],
        e="fhn",
        c="fhn",
        PAR=problem.pars,
        ICP=[c],
        DS=-0.01,
        NMX=200,
        SP=["HB1"],
        runner=auto.runner,
    )
    hopf = branch("HB1")[f"PAR({c})"]
    assert hopf == pytest.approx(_fitzhugh_nagumo_hopf_c(ode.a, ode.b, ode.z), rel=1e-6)
