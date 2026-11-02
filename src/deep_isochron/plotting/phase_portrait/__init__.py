"""Phase-portrait animations of a normal form for a talk. The subpackage has three
modules. `style.py` holds `PortraitStyle`, the drawing constants; `drawing.py` holds the
helpers that the animations share, which are private to the subpackage; `kick.py` holds
`animate_kick`.
"""

from .kick import animate_kick as animate_kick
from .style import PortraitStyle as PortraitStyle
