"""The data layer: trajectory generation, datasets on disk, windows for training and
validation, and the transfer to the accelerator (ADR-0008).

``generate`` integrates a system from sampled initial conditions into a
``TimeSeriesDataSource``, one netCDF4 file per dataset with grouped
``DatasetMetadata``; ``windows`` cuts training and validation windows from the whole
trajectories as grain pipelines; ``device`` carries the batches to the accelerator.
"""

from .dataset import (
    DatasetMetadata as DatasetMetadata,
    GridSpec as GridSpec,
    Provenance as Provenance,
    SamplingSpec as SamplingSpec,
    SolveSpec as SolveSpec,
    SystemSpec as SystemSpec,
    TimeSeriesDataSource as TimeSeriesDataSource,
)
from .device import (
    resolve_device as resolve_device,
    single_threaded as single_threaded,
    to_device as to_device,
)
from .generate import (
    dataset_path as dataset_path,
    generate as generate,
    generation_metadata as generation_metadata,
)
from .initial_conditions import (
    AbstractICSampler as AbstractICSampler,
    OnCycleGaussian as OnCycleGaussian,
    UniformAnnulus as UniformAnnulus,
    UniformBox as UniformBox,
)
from .windows import (
    mixed_window_batches as mixed_window_batches,
    mixed_windows as mixed_windows,
    RandomWindow as RandomWindow,
    transient_weight as transient_weight,
    validation_windows as validation_windows,
    WeightedWindow as WeightedWindow,
    window_batches as window_batches,
    WindowBatchSource as WindowBatchSource,
    windows as windows,
)
