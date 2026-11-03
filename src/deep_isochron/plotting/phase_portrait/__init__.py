"""Phase-portrait animations of a normal form for a talk. The subpackage has five
modules. `style.py` holds `PortraitStyle`, the drawing constants; `drawing.py` holds the
helpers that the animations share, which are private to the subpackage; `labels.py`
holds the search that places text labels clear of the drawn curves; `kick.py` holds
`animate_kick`; `sync.py` holds `animate_isochron_sync`.
"""

from .kick import animate_kick as animate_kick
from .style import PortraitStyle as PortraitStyle
from .sync import animate_isochron_sync as animate_isochron_sync
