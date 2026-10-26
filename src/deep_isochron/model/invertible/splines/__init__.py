"""The monotone spline bijections: linear ($C^0$), rational quadratic ($C^1$) and cubic
B-spline ($C^2$), all the identity outside their range."""

from .base import AbstractSpline as AbstractSpline
from .cubic import CubicBSpline as CubicBSpline
from .linear import LinearSpline as LinearSpline
from .rational_quadratic import MonotonicRQSpline as MonotonicRQSpline
