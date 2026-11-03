"""The label search of the phase-portrait animations: `_place_labels` puts each text
label near a landing point on the limit cycle, at the position whose white box is
farthest from the curves drawn in the foreground.

`sync.py` places the φ labels of `animate_isochron_sync` with it. The helpers
`_box_extent`, `_foreground_samples` and `_anchor` and the search constants are private
to the subpackage, so that a further animation can use the same search.

The design and its decisions are in `docs/changes/2026-11-02-ksiam-animations.md`.
"""

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from .drawing import _TEXT_BOX_PAD


_LABEL_DIRECTIONS = np.round(
    np.stack((np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)), 1),
    12,
)
"""The unit directions, every 45°, in which a φ label may sit from its landing point;
rounded so that the axis directions have exact zero components."""

_LABEL_OFFSETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
"""The distances, in plane units, from a landing point to its label's anchor; up to
1.2, since a 20-point label only clears the cycle of the default scene from about 1."""

_MIN_LABEL_CLEARANCE_PX = 8.0
"""The smallest gap, in pixels, between a label's box and a foreground curve that
reads as clear of the curve on a slide rather than touching it."""

_SAMPLE_SPACING = 1.0
"""The largest gap between two display samples of a foreground curve, in pixels, so
that a measured clearance is short of the true one by at most half a pixel."""


def _box_extent(text: Text, dpi: float) -> Bbox:
    """The display extent of `text` with the margin of its `_TEXT_BOX`, in pixels, in
    a figure of `dpi` dots per inch."""
    return text.get_window_extent().padded(_TEXT_BOX_PAD * dpi / 72)


def _foreground_samples(
    lines: list[Line2D], dpi: float
) -> tuple[Float[np.ndarray, "k 2"], Float[np.ndarray, " k"]]:
    """The display points that `lines` cover, each with the radius in pixels that its
    drawing covers around it: every drawn line sampled at most `_SAMPLE_SPACING` apart,
    with half its line width, and every marker with half its size, in a figure of
    `dpi` dots per inch."""
    px = dpi / 72  # pixels per point
    points, radii = [], []
    for line in lines:
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if line.get_linestyle() != "None" and len(xy) > 1:
            steps = np.diff(xy, axis=0)
            counts = np.maximum(
                np.ceil(np.hypot(*steps.T) / _SAMPLE_SPACING).astype(int), 1
            )
            segment = np.repeat(np.arange(len(steps)), counts)
            first = np.repeat(np.cumsum(counts) - counts, counts)
            fraction = (np.arange(counts.sum()) - first) / counts[segment]
            dense = np.concatenate(
                (xy[segment] + fraction[:, None] * steps[segment], xy[-1:])
            )
            points.append(dense)
            radii.append(np.full(len(dense), line.get_linewidth() * px / 2))
        if line.get_marker() not in ("None", "", " ", None):
            points.append(xy)
            radii.append(np.full(len(xy), line.get_markersize() * px / 2))
    return np.concatenate(points), np.concatenate(radii)


def _place_labels(
    plane: Axes,
    labels: list[Text],
    landings: Float[np.ndarray, "m 2"],
    foreground: list[Line2D],
    clock: Text,
    dpi: float,
) -> None:
    """Move each of `labels` near its group's landing point on the cycle, `landings`,
    to the position where its white box is farthest from the `foreground` curves.

    Each label is tried in every direction of `_LABEL_DIRECTIONS` at every distance of
    `_LABEL_OFFSETS` from its landing point, aligned by the edge or the corner of its
    text that faces the point. A position fits when the label's box lies inside `plane`
    and overlaps neither the clock's box nor a label placed before it. Its clearance is
    the smallest gap in pixels between the box and the foreground, whose lines and
    markers are measured with their drawn widths, and it is admissible when it also
    keeps a clearance of `_MIN_LABEL_CLEARANCE_PX`. The admissible position with the
    largest clearance wins, the smaller offset breaking a tie.

    No error is raised when no position is admissible. The label then takes the fitting
    position with the largest clearance, or the position with the largest clearance
    when none fits, and its box covers the curves below it.

    The figure is measured as it is drawn when the call is made, at `dpi`, the figure's
    own, so the call follows the drawing of the frame in which the labels are shown.
    """
    plane.apply_aspect()
    plane_box = plane.get_window_extent()
    points, radii = _foreground_samples(foreground, dpi)
    taken = [_box_extent(clock, dpi)]
    for label, landing in zip(labels, landings):
        candidates = []
        for k, direction in enumerate(_LABEL_DIRECTIONS):
            for offset in _LABEL_OFFSETS:
                _anchor(label, landing, direction, offset)
                box = _box_extent(label, dpi)
                fits = (
                    plane_box.x0 <= box.x0
                    and box.x1 <= plane_box.x1
                    and plane_box.y0 <= box.y0
                    and box.y1 <= plane_box.y1
                    and not any(box.overlaps(other) for other in taken)
                )
                dx = np.maximum(
                    np.maximum(box.x0 - points[:, 0], points[:, 0] - box.x1), 0
                )
                dy = np.maximum(
                    np.maximum(box.y0 - points[:, 1], points[:, 1] - box.y1), 0
                )
                clearance = float(np.min(np.hypot(dx, dy) - radii))
                candidates.append((fits, clearance, offset, k))
        admissible = [c for c in candidates if c[0] and c[1] >= _MIN_LABEL_CLEARANCE_PX]
        fitting = [c for c in candidates if c[0]]
        _, _, offset, k = min(
            admissible or fitting or candidates, key=lambda c: (-c[1], c[2])
        )
        _anchor(label, landing, _LABEL_DIRECTIONS[k], offset)
        taken.append(_box_extent(label, dpi))


def _anchor(
    label: Text,
    landing: Float[np.ndarray, " 2"],
    direction: Float[np.ndarray, " 2"],
    offset: float,
) -> None:
    """Put `label` at `offset` from `landing` along `direction`, aligned by the edge or
    the corner of its text that faces `landing`."""
    label.set_position(
        (
            float(landing[0] + offset * direction[0]),
            float(landing[1] + offset * direction[1]),
        )
    )
    label.set_horizontalalignment(
        ("right", "center", "left")[int(np.sign(direction[0])) + 1]
    )
    label.set_verticalalignment(
        ("top", "center", "bottom")[int(np.sign(direction[1])) + 1]
    )
