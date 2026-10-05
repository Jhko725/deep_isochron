"""Windows: how training and validation windows are cut from the whole trajectories of a
``TimeSeriesDataSource``. Two ways, deliberately both (ADR-0008, Decisions 2 and 3):

``per_element``
    one window per grain element — ``RandomWindow`` / ``WeightedWindow`` transforms
    behind ``windows()`` and ``mixed_windows()``, and ``validation_windows()``. Short,
    idiomatic grain, the **reference semantics** the tests compare against; ≈ 0.1–0.2 ms
    per window, i.e. 50–90 ms per batch of 512 — too slow to feed a 30 ms step.
``batched``
    ``WindowBatchSource`` behind ``window_batches()`` / ``mixed_window_batches()`` — a
    source whose element is a whole batch, one vectorized gather, epochs of every window
    once, concrete length. ≈ 1 ms per batch. **The training-run path.**

``common`` holds what both share (types, range checks, weight functions, the categorical
draw); ``device`` holds what both are consumed through (``single_threaded``,
``to_device``, ``resolve_device``).
"""

from .batched import (
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
from .device import (
    resolve_device as resolve_device,
    single_threaded as single_threaded,
    to_device as to_device,
)
from .per_element import (
    mixed_windows as mixed_windows,
    RandomWindow as RandomWindow,
    validation_windows as validation_windows,
    WeightedWindow as WeightedWindow,
    windows as windows,
)
