"""Analysis of planar limit-cycle systems (roadmap Phase E, ADR-0012).

Two halves share one object: ``Cycle`` (``cycle.py``), a closed curve parameterized by
the phase that advances uniformly in time. ``analysis.data`` estimates it — and the
settling time, the winding direction, the Floquet multiplier — from sampled, possibly
noisy trajectories; ``analysis.ode`` computes the same quantities, and the asymptotic
phase and isochrons, from a known vector field, as ground truth. Replanned 2026-10-15
(ADR-0012 amendment): the ODE half comes first, and ``Cycle`` is one object among
several rather than the organizing one — ``analysis.ode`` starts from ``Hyperplane``,
``PoincareMap`` and ``LimitCycle``.
"""

from .cycle import Cycle as Cycle
from .errors import NotConvergedError as NotConvergedError
