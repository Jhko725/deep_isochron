"""Every concrete bijection must be registered for testing (or excluded with a reason)
."""

import importlib
import inspect
import pkgutil

import deep_isochron.model.invertible as pkg
from deep_isochron.model.invertible.base import (
    AbstractBijection,
    AbstractScalarBijection,
)

from tests.registry import SCALAR_TEMPLATES, UNTESTED, VECTOR_BUILDERS


def _import_all_submodules():
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(m.name)


def concrete_bijections() -> set[type]:
    _import_all_submodules()
    out, stack = set(), [AbstractBijection]
    while stack:
        for sub in stack.pop().__subclasses__():
            stack.append(sub)
            if not inspect.isabstract(sub):
                out.add(sub)
    return out


def test_every_bijection_is_registered(key):
    covered = {type(t) for t in SCALAR_TEMPLATES.values()}
    covered |= {type(b(key)) for b in VECTOR_BUILDERS.values()}
    missing = concrete_bijections() - covered - set(UNTESTED)
    assert not missing, f"""add a template/builder or an UNTESTED reason for:
        {sorted(c.__name__ for c in missing)}"""


def test_scalar_templates_are_scalar():
    bad = {
        n
        for n, t in SCALAR_TEMPLATES.items()
        if not isinstance(t, AbstractScalarBijection)
    }
    assert not bad, f"not AbstractScalarBijection: {bad}"


def test_untested_reasons_are_stale(key):
    """An UNTESTED entry that is now covered should be removed."""
    covered = {type(t) for t in SCALAR_TEMPLATES.values()} | {
        type(b(key)) for b in VECTOR_BUILDERS.values()
    }
    stale = set(UNTESTED) & covered
    assert not stale, f"remove from UNTESTED: {sorted(c.__name__ for c in stale)}"
