import abc
import math
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

from .dataset import TimeSeriesDataset


def get_trajectory_segments(dataset, sample_idx, time_idx, segment_length: int):
    t_segment = jax.lax.dynamic_index_in_dim(dataset.t, sample_idx, keepdims=False)
    t_segment = jax.lax.dynamic_slice_in_dim(t_segment, time_idx, segment_length)

    u_segment = jax.lax.dynamic_index_in_dim(dataset.u, sample_idx, keepdims=False)
    u_segment = jax.lax.dynamic_slice_in_dim(u_segment, time_idx, segment_length)
    return t_segment, u_segment


BatchState = Any


class AbstractBatching(eqx.Module):
    batch_size: eqx.AbstractVar[int | None]

    @abc.abstractmethod
    def init(self, num_total_data: int) -> BatchState: ...

    @abc.abstractmethod
    def generate_batch(
        self, batch_state: BatchState
    ) -> tuple[Int[Array, " {self.batch_size}"], BatchState]:
        pass


class FullBatching(AbstractBatching):
    @property
    def batch_size(self) -> None:
        return None

    def init(self, num_total_data: int) -> BatchState:
        return num_total_data

    def generate_batch(
        self, batch_state: BatchState
    ) -> tuple[Int[Array, " {self.batch_size}"], BatchState]:
        num_total_data = batch_state_next = batch_state
        batch_data_indices = np.arange(num_total_data)
        return batch_data_indices, batch_state_next


class RandomSampleBatching(AbstractBatching):
    batch_size: int = eqx.field(static=True)
    random_seed: int = eqx.field(static=True)

    def __init__(self, batch_size: int, *, random_seed: int = 0):
        self.batch_size = batch_size
        self.random_seed = random_seed

    def init(self, num_total_data: int) -> BatchState:
        return num_total_data, jax.random.key(self.random_seed)

    def generate_batch(
        self, batch_state: BatchState
    ) -> tuple[Int[Array, " {self.batch_size}"], BatchState]:
        num_total_data, key = batch_state
        key, key_next = jax.random.split(key)
        batch_data_indices = jax.random.randint(
            key, (self.batch_size,), 0, num_total_data
        )
        batch_state_next = (num_total_data, key_next)
        return batch_data_indices, batch_state_next


class MiniBatching(AbstractBatching):
    batch_size: int = eqx.field(static=True)
    permute_initial: bool = eqx.field(static=True)
    permute_every_epoch: bool = eqx.field(static=True)
    drop_last: bool = eqx.field(static=True)
    random_seed: int = eqx.field(static=True)

    def __init__(
        self,
        batch_size: int,
        permute_initial: bool = True,
        permute_every_epoch: bool = False,
        drop_last: bool = True,
        *,
        random_seed: int = 0,
    ):
        self.batch_size = batch_size
        self.permute_initial = permute_initial
        self.permute_every_epoch = permute_every_epoch
        self.drop_last = drop_last
        self.random_seed = random_seed

    def init(self, num_total_data: int) -> BatchState:
        key, key_permute = jax.random.split(jax.random.key(self.random_seed))
        data_inds = (
            jax.random.permutation(key_permute, num_total_data)
            if self.permute_initial
            else jnp.arange(num_total_data)
        )
        minibatch_ind = 0
        return data_inds, minibatch_ind, key

    def batches_per_epoch(self, num_data_total: int) -> Float[Array, ""]:
        round_fn = math.floor if self.drop_last else math.ceil
        return round_fn(num_data_total / self.batch_size)

    def generate_batch(
        self, batch_state: BatchState
    ) -> tuple[Int[Array, " {self.batch_size}"], BatchState]:
        data_inds, minibatch_ind, key = batch_state

        batch_data_indices = jax.lax.dynamic_slice_in_dim(
            data_inds,
            minibatch_ind,
            self.batch_size,
        )

        # Handle permutation and drop_last
        num_total_data = len(data_inds)
        if minibatch_ind + 1 == self.batches_per_epoch(num_total_data):
            if not self.drop_last:
                batch_data_indices = data_inds[minibatch_ind * self.batch_size :]
            minibatch_ind = 0
            if self.permute_every_epoch:
                key, key_permute = jax.random.split(key)
                data_inds = jax.random.permutation(key_permute, num_total_data)

        batch_state_next = (data_inds, minibatch_ind + 1, key)

        return batch_data_indices, batch_state_next


class SegmentLoader(eqx.Module):
    """
    Basic implementation of a SegmentLoader, which is a dedicated class that samples a
    given dataset of trajectories (of class TimeSeriesDataset) and returns a batch of
    trajectory segments with fixed length.

    This class inspired by DataLoader classes in many deep learning libraries, such as
    `torch.utils.data.DataLoader`.
    """

    dataset: TimeSeriesDataset
    segment_length: int
    batch_strategy: AbstractBatching
    aux_data: PyTree[Float[ArrayLike, "samples time *rest"]] | None = None

    def __check_init__(self):
        if self.segment_length < 2:
            raise ValueError("Minimum allowed segment length is 2.")

    @property
    def num_segments_per_traj(self) -> int:
        return self.dataset.trajectory_length - self.segment_length + 1

    @property
    def num_total_segments(self) -> int:
        return len(self.dataset) * self.num_segments_per_traj

    @property
    def batch_size(self) -> int:
        batch_size = self.batch_strategy.batch_size
        if batch_size is None:
            return self.num_total_segments
        else:
            return batch_size

    @property
    def num_batches(self) -> int | None:
        """Number of batches required to cover (on average) the entire dataset.

        This number is always rounded up to the nearest integer."""
        return math.ceil(self.num_total_segments / self.batch_size)

    def get_segments(self, traj_idx, time_idx):
        return eqx.filter_vmap(get_trajectory_segments, in_axes=(None, 0, 0, None))(
            self.dataset, traj_idx, time_idx, self.segment_length
        )

    def get_auxdata(self, traj_idx, time_idx):
        @partial(jax.vmap, in_axes=(None, 0, 0))
        def _get_aux_single(x, idx_traj, idx_time):
            x_ = jax.lax.dynamic_index_in_dim(x, idx_traj, axis=0, keepdims=False)
            return jax.lax.dynamic_index_in_dim(x_, idx_time, axis=0, keepdims=False)

        def _get_aux(x):
            return _get_aux_single(x, traj_idx, time_idx)

        return jax.tree.map(_get_aux, self.aux_data)

    def linear_to_sample_indices(
        self, linear_indices: Int[Array, " {self.batch_size}"]
    ) -> tuple[Int[Array, " {self.batch_size}"], Int[Array, " {self.batch_size}"]]:
        """
        Converts the 1D array of linear indices representing the starting position of
        the segments in the batch to a tuple of indices that can be used to locate the
        said position in `self.dataset.u`.
        """
        return jnp.divmod(linear_indices, self.num_segments_per_traj)

    def init(self) -> PyTree:
        """
        Returns the initial loader_state to be fed into the first call of
        `self.load_batch`.

        This is inspired by optax's optimizer.init function.
        """
        batch_state_init = self.batch_strategy.init(self.num_total_segments)
        return (batch_state_init,)

    def load_batch(self, loader_state: PyTree) -> tuple[PyTree[Array], ...]:
        """
        Main logic to load a single batch of time series data segments.

        loader_state contains any extra state necessary to generate a particular batch:
        For random sampling, this corresponds to the random key, and for minibatch
        sampling, this would correspond to the batch index (and the random key if the
        data is reshuffled each epoch.)
        """
        (batch_state,) = loader_state
        linear_indices, batch_state_next = self.batch_strategy.generate_batch(
            batch_state
        )
        sample_indices = self.linear_to_sample_indices(linear_indices)
        batch = self.get_segments(*sample_indices)
        if self.aux_data is not None:
            batch_aux = self.get_auxdata(*sample_indices)
            batch = (batch, batch_aux)
        loader_state_next = (batch_state_next,)
        return batch, loader_state_next
