"""Analysis of planar limit-cycle systems (roadmap Phase E, ADR-0012).

``analysis.ode`` computes the reference objects of an oscillator from its vector
field: the limit cycle, its Floquet multipliers, and later the asymptotic phase and
isochrons. ``analysis.data`` estimates what it can from sampled, possibly noisy
trajectories: the cycle, the settling time, the winding direction. ``Cycle``
(``cycle.py``) is the closed curve both halves can produce; since the replanning of
2026-10-15 (ADR-0012 amendment) the ODE half comes first and starts from
``CrossSection``, ``PoincareMap`` and ``LimitCycle``, with ``Cycle`` one object among
several rather than the organizing one. ``analysis.model`` holds the analyses of a
trained model against a reference: ``phase_origin`` aligns the model's phase with a
cycle's.
"""

from .cycle import Cycle as Cycle
from .errors import NotConvergedError as NotConvergedError
from .model import phase_origin as phase_origin, phase_sense as phase_sense
