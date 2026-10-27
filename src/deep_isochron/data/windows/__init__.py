"""Windows: how training and validation windows are cut from the whole trajectories
of a ``TimeSeriesDataSource``, in two ways on purpose (ADR-0008, Decisions 2 and 3).

``elementwise`` cuts one window per grain element, through the ``RandomWindow`` and
``WeightedWindow`` transforms behind ``windows()`` and ``mixed_windows()``, and
``validation_windows()``. It is short, idiomatic grain and the reference semantics the
tests compare against, but at 0.1–0.2 ms per window, 50–90 ms per batch of 512, too
slow to feed a 30 ms step. ``batched`` is the training-run path: ``WindowBatchSource``
behind ``window_batches()`` and ``mixed_window_batches()``, a source whose element is a
whole batch by one vectorized gather, with epochs of every window once and a concrete
length, at about 1 ms per batch. ``common`` holds what both share. How either is
consumed next to JAX is ``data.device``.
"""

from .batched import (
    mixed_ranges as mixed_ranges,
    mixed_window_batches as mixed_window_batches,
    window_batches as window_batches,
    WindowBatchSource as WindowBatchSource,
)
from .common import (
    Batch as Batch,
    categorical as categorical,
    Element as Element,
    transient_weight as transient_weight,
    WeightFn as WeightFn,
)
from .elementwise import (
    mixed_windows as mixed_windows,
    RandomWindow as RandomWindow,
    validation_windows as validation_windows,
    WeightedWindow as WeightedWindow,
    windows as windows,
)
