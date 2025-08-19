# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import shapely
import torch
from shapely import Polygon
from torch import Generator
from torch.utils.data import Sampler

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice
from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips


class GeoSampler(Sampler[GeoSlice], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Polygon | None = None,
        toi: pd.Interval | None = None,
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.8
           The *toi* parameter.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        self.index = dataset.index
        self.res = dataset.res

        if roi:
            self.roi = roi
            self.index = self.index.clip(roi)
        else:
            x, y, t = dataset.bounds
            self.roi = shapely.box(x.start, y.start, x.stop, y.stop)

        if toi:
            self.toi = toi
            self.index = self.index.iloc[self.index.index.overlaps(toi)]
            tmin = np.maximum(self.index.index.left, np.datetime64(toi.left))
            tmax = np.minimum(self.index.index.right, np.datetime64(toi.right))
            self.index.index = pd.IntervalIndex.from_arrays(
                tmin, tmax, closed='both', name='datetime'
            )
        else:
            x, y, t = dataset.bounds
            self.toi = pd.Interval(t.start, t.stop)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[GeoSlice]:
        """Return the index of a dataset.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """


class SpatialSampler(abc.ABC):
    """Abstract base class for spatial sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Spatial samplers handle the geographic (x, y) dimension of sampling,
    independent of the temporal dimension.
    """

    def __init__(self, dataset: GeoDataset, roi: Polygon | None = None) -> None:
        """Initialize a new SpatialSampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        self.dataset = dataset
        self.index = dataset.index
        self.res = dataset.res

        if roi:
            self.roi = roi
            self.index = self.index.clip(roi)
        else:
            x, y, t = dataset.bounds
            self.roi = shapely.box(x.start, y.start, x.stop, y.stop)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Return spatial coordinates to index a dataset.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of spatial samples.

        Returns:
            number of spatial samples
        """


class TemporalSampler(abc.ABC):
    """Abstract base class for temporal sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Temporal samplers handle the time dimension of sampling,
    independent of the spatial dimensions.
    """

    def __init__(self, dataset: GeoDataset, toi: pd.Interval | None = None) -> None:
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: dataset to index from
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        self.dataset = dataset
        self.index = dataset.index

        if toi:
            self.toi = toi
            self.index = self.index.iloc[self.index.index.overlaps(toi)]
            tmin = np.maximum(self.index.index.left, np.datetime64(toi.left))
            tmax = np.minimum(self.index.index.right, np.datetime64(toi.right))
            self.index.index = pd.IntervalIndex.from_arrays(
                tmin, tmax, closed='both', name='datetime'
            )
        else:
            x, y, t = dataset.bounds
            self.toi = pd.Interval(t.start, t.stop)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[slice]:
        """Return temporal coordinates to index a dataset.

        Yields:
            [tmin:tmax] coordinates to index a dataset.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of temporal samples.

        Returns:
            number of temporal samples
        """


class RandomSpatialSampler(SpatialSampler):
    """Samples spatial locations from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random spatial :term:`chips <chip>` as possible.
    Note that randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new RandomSpatialSampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each spatial :term:`patch`
            length: number of random spatial samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            generator: pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.generator = generator
        self.length = 0
        self.bounds = []
        areas = []
        for hit in range(len(self.index)):
            bounds = self.index.geometry.iloc[hit].bounds
            xmin, ymin, xmax, ymax = bounds
            if xmax - xmin >= self.size[1] and ymax - ymin >= self.size[0]:
                if xmax > xmin and ymax > ymin:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.bounds.append(bounds)
                areas.append((xmax - xmin) * (ymax - ymin))

        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Return spatial coordinates to index a dataset.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            bounds = self.bounds[idx]

            # Choose a random spatial index within that tile
            bounding_box = get_random_bounding_box(
                bounds, self.size, self.res, self.generator
            )

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of spatial samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class GridSpatialSampler(SpatialSampler):
    """Samples spatial locations in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new GridSpatialSampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each spatial :term:`patch`
            stride: distance to skip between each patch (defaults to *size*)
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        if stride is not None:
            self.stride = _to_tuple(stride)
        else:
            self.stride = self.size

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])
            self.stride = (self.stride[0] * self.res[1], self.stride[1] * self.res[0])

        self.length = 0
        self.valid_tiles = []
        for i in range(len(self.index)):
            bounds = self.index.geometry.iloc[i].bounds
            xmin, ymin, xmax, ymax = bounds
            if xmax - xmin < self.size[1] or ymax - ymin < self.size[0]:
                continue
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols
            self.valid_tiles.append((i, bounds, rows, cols))

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Return spatial coordinates to index a dataset.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        # For each valid tile...
        for i, bounds, rows, cols in self.valid_tiles:
            # For each row...
            for row in range(rows):
                ymin = bounds[1] + row * self.stride[0]
                ymax = ymin + self.size[0]

                # For each column...
                for col in range(cols):
                    xmin = bounds[0] + col * self.stride[1]
                    xmax = xmin + self.size[1]

                    yield slice(xmin, xmax), slice(ymin, ymax)

    def __len__(self) -> int:
        """Return the number of spatial samples over the ROI.

        Returns:
            number of spatial patches that will be sampled
        """
        return self.length


class PreChippedSpatialSampler(SpatialSampler):
    """Samples entire spatial extents at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into spatial :term:`chips <chip>`.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Polygon | None = None,
        shuffle: bool = False,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new PreChippedSpatialSampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
            generator: pseudo-random number generator (PRNG) used in combination with
                shuffle.
        """
        super().__init__(dataset, roi)
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Return spatial coordinates to index a dataset.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = partial(torch.randperm, generator=self.generator)

        for idx in generator(len(self)):
            i = int(idx)
            xmin, ymin, xmax, ymax = self.index.geometry.iloc[i].bounds
            yield slice(xmin, xmax), slice(ymin, ymax)

    def __len__(self) -> int:
        """Return the number of spatial samples over the ROI.

        Returns:
            number of spatial patches that will be sampled
        """
        return len(self.index)


class WindowTemporalSampler(TemporalSampler):
    """Samples temporal windows of a fixed duration.

    This sampler is useful for time series analysis where you want to sample
    windows of a specific duration (e.g., monthly, seasonal, yearly windows).
    Only generates windows that contain actual data.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        window_size: timedelta,
        stride: timedelta | None = None,
        toi: pd.Interval | None = None,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new WindowTemporalSampler instance.

        Args:
            dataset: dataset to index from
            window_size: size of the temporal window to sample
            stride: temporal stride between windows (defaults to window_size)
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            generator: pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, toi)
        self.window_size = window_size
        self.stride = stride or window_size
        self.generator = generator

        # Calculate all possible windows that contain data
        self.windows = []
        if not self.index.empty:
            # Get the overall temporal bounds from the filtered index
            global_tmin = self.index.index.left.min()
            global_tmax = self.index.index.right.max()

            # Generate all possible windows
            current_start = pd.Timestamp(global_tmin)
            while current_start + window_size <= pd.Timestamp(global_tmax):
                current_end = current_start + window_size

                # Check if this window overlaps with any data in the filtered index
                window_interval = pd.Interval(current_start, current_end)
                if self.index.index.overlaps(window_interval).any():
                    self.windows.append((current_start, current_end))

                current_start += self.stride

    def __iter__(self) -> Iterator[slice]:
        """Return temporal coordinates to index a dataset.

        Yields:
            [tmin:tmax] coordinates to index a dataset.
        """
        for tmin, tmax in self.windows:
            yield slice(tmin, tmax)

    def __len__(self) -> int:
        """Return the number of temporal windows.

        Returns:
            number of temporal windows
        """
        return len(self.windows)


class FixedLengthTemporalSampler(TemporalSampler):
    """Samples temporal sequences of a fixed number of timestamps.

    This sampler is useful when you want to ensure a consistent number of
    timestamps in each sample, regardless of their temporal spacing.
    Only generates samples from available timestamps in the dataset.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        length: int,
        stride: int = 1,
        toi: pd.Interval | None = None,
        mode: Literal['consecutive', 'random'] = 'consecutive',
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new FixedLengthTemporalSampler instance.

        Args:
            dataset: dataset to index from
            length: number of timestamps to include in each sample
            stride: temporal stride between consecutive samples
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            mode: sampling mode - 'consecutive' for consecutive timestamps,
                'random' for random selection within available timestamps
            generator: pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, toi)
        self.length = length
        self.stride = stride
        self.mode = mode
        self.generator = generator

        # Get unique timestamps sorted by time from the filtered index
        if not self.index.empty:
            # Collect all unique timestamp boundaries from the actual data
            all_timestamps = set()
            for interval in self.index.index:
                all_timestamps.add(interval.left)
                all_timestamps.add(interval.right)
            self.timestamps = sorted(all_timestamps)
        else:
            self.timestamps = []

        # Calculate number of possible samples
        if self.mode == 'consecutive':
            self.num_samples = max(0, len(self.timestamps) - length + 1)
        else:  # random mode
            self.num_samples = (
                len(self.timestamps) if len(self.timestamps) >= length else 0
            )

    def __iter__(self) -> Iterator[slice]:
        """Return temporal coordinates to index a dataset.

        Yields:
            [tmin:tmax] coordinates to index a dataset.
        """
        if self.mode == 'consecutive':
            for i in range(0, self.num_samples, self.stride):
                start_idx = i
                end_idx = i + self.length - 1
                tmin = self.timestamps[start_idx]
                tmax = self.timestamps[end_idx]
                yield slice(tmin, tmax)
        else:  # random mode
            for _ in range(self.num_samples):
                # Randomly select timestamps
                indices = torch.randperm(
                    len(self.timestamps), generator=self.generator
                )[: self.length]
                selected_timestamps = [self.timestamps[i] for i in sorted(indices)]
                tmin = selected_timestamps[0]
                tmax = selected_timestamps[-1]
                yield slice(tmin, tmax)

    def __len__(self) -> int:
        """Return the number of temporal samples.

        Returns:
            number of temporal samples
        """
        return (
            max(0, (self.num_samples - 1) // self.stride + 1)
            if self.stride > 0
            else self.num_samples
        )


class FullTemporalSampler(TemporalSampler):
    """Samples the full temporal extent of the dataset.

    This sampler returns the entire temporal range of the dataset,
    which is useful when you want to include all available timestamps.
    """

    def __init__(self, dataset: GeoDataset, toi: pd.Interval | None = None) -> None:
        """Initialize a new FullTemporalSampler instance.

        Args:
            dataset: dataset to index from
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        super().__init__(dataset, toi)

    def __iter__(self) -> Iterator[slice]:
        """Return temporal coordinates to index a dataset.

        Yields:
            [tmin:tmax] coordinates to index a dataset.
        """
        if not self.index.empty:
            tmin = self.index.index.left.min()
            tmax = self.index.index.right.max()
            yield slice(tmin, tmax)

    def __len__(self) -> int:
        """Return the number of temporal samples.

        Returns:
            number of temporal samples (always 1 for full temporal sampling)
        """
        return 1 if not self.index.empty else 0


class SpatioTemporalGeoSampler(GeoSampler):
    """Combines spatial and temporal sampling strategies.

    This sampler allows you to independently configure spatial and temporal
    sampling strategies and combine them to create spatiotemporal samples.
    It ensures that all generated samples have actual data available in the
    dataset index.
    """

    def __init__(
        self,
        spatial_sampler: SpatialSampler,
        temporal_sampler: type[FullTemporalSampler] | TemporalSampler = FullTemporalSampler,
        mode: Literal['product', 'zip'] = 'product',
        max_retries: int = 100,
    ) -> None:
        """Initialize a new SpatioTemporalGeoSampler instance.

        Common Usage Examples:

        1. Agricultural Monitoring (Regular Grid + Seasonal Windows):
            ```python
            # Sample 256x256 pixel patches in a grid with 50% overlap
            spatial = GridSpatialSampler(dataset, size=256, stride=128)
            # Sample 3-month seasonal windows
            temporal = WindowTemporalSampler(dataset, window_size=timedelta(days=90))
            sampler = SpatioTemporalGeoSampler(spatial, temporal, mode='product')
            ```

        2. Change Detection (Random Patches + Fixed Timestamps):
            ```python
            # Random 512x512 patches
            spatial = RandomSpatialSampler(dataset, size=512, length=1000)
            # Sample sequences of 2 timestamps
            temporal = FixedLengthTemporalSampler(dataset, length=2)
            sampler = SpatioTemporalGeoSampler(spatial, temporal, mode='zip')
            ```

        3. Time Series Analysis (Pre-chipped + Full Temporal):
            ```python
            # Use pre-defined spatial chips
            spatial = PreChippedSpatialSampler(dataset)
            # Get full temporal extent
            temporal = FullTemporalSampler(dataset)
            sampler = SpatioTemporalGeoSampler(spatial, temporal, mode='product')
            ```

        4. Crop Monitoring (Grid + Monthly Windows):
            ```python
            # Regular grid of 1km x 1km patches
            spatial = GridSpatialSampler(dataset, size=1000, units=Units.CRS)
            # Monthly windows with 15-day stride
            temporal = WindowTemporalSampler(dataset,
                                                     window_size=timedelta(days=30),
                                                     stride=timedelta(days=15))
            sampler = SpatioTemporalGeoSampler(spatial, temporal, mode='product')
            ```

        Args:
             spatial_sampler: spatial sampling strategy
             temporal_sampler: temporal sampling strategy. Defaults to
                  :class:`~torchgeo.samplers.FullTemporalSampler` which samples the full
                  temporal extent of the dataset (making it a spatial-only sampler).
             mode: combination mode - 'product' tries all combinations between spatial and temporal
                  samples to form the Cartesian product, 'zip' for pairing spatial and temporal samples
             max_retries: maximum number of retries to find valid samples when
                  generating random combinations
        """
        self.spatial_sampler = spatial_sampler
        self.temporal_sampler = temporal_sampler
        self.mode = mode
        self.max_retries = max_retries

        # Both samplers should be working with the same dataset
        assert spatial_sampler.dataset is temporal_sampler.dataset, (
            'Spatial and temporal samplers must use the same dataset'
        )

        self.dataset = spatial_sampler.dataset

        # Use the original dataset index for validation, not the filtered ones
        self.index = self.dataset.index
        self.res = self.dataset.res

        # Pre-compute valid spatiotemporal combinations if mode is 'product'
        if self.mode == 'product':
            self._precompute_valid_samples()

    def _precompute_valid_samples(self) -> None:
        """Pre-compute all valid spatiotemporal combinations.

        This is used for 'product' mode to ensure all yielded samples
        have data available in the dataset.
        """
        self.valid_samples = []

        for t_slice in self.temporal_sampler:
            # Create temporal interval for filtering
            temporal_interval = pd.Interval(t_slice.start, t_slice.stop)

            # Filter original dataset index by temporal constraint
            temporal_index = self.index.iloc[
                self.index.index.overlaps(temporal_interval)
            ]

            if temporal_index.empty:
                continue

            for x_slice, y_slice in self.spatial_sampler:
                # Filter by spatial constraint
                spatiotemporal_index = temporal_index.cx[
                    x_slice.start : x_slice.stop, y_slice.start : y_slice.stop
                ]

                # Only add if there's actual data at this spatiotemporal location
                if not spatiotemporal_index.empty:
                    self.valid_samples.append((x_slice, y_slice, t_slice))
        # Create geodataframe with valid samples
        if self.valid_samples:
            self.samples = pd.DataFrame(
                self.valid_samples, columns=['x_slice', 'y_slice', 't_slice']
            )
            self.samples['geometry'] = self.samples.apply(
                lambda row: shapely.box(
                    row['x_slice'].start,
                    row['y_slice'].start,
                    row['x_slice'].stop,
                    row['y_slice'].stop,
                ),
                axis=1,
            )
            self.samples = self.samples.set_geometry('geometry')
            self.samples.set_crs(self.dataset.crs, inplace=True)

    def _has_data_at_query(
        self, x_slice: slice, y_slice: slice, t_slice: slice
    ) -> bool:
        """Check if dataset has data at the given spatiotemporal query.

        Args:
            x_slice: spatial x slice
            y_slice: spatial y slice
            t_slice: temporal slice

        Returns:
            True if data exists at this spatiotemporal location
        """
        # Create temporal interval for filtering
        temporal_interval = pd.Interval(t_slice.start, t_slice.stop)

        # Filter original dataset index by temporal constraint
        temporal_index = self.index.iloc[self.index.index.overlaps(temporal_interval)]

        if temporal_index.empty:
            return False

        # Filter by spatial constraint
        spatiotemporal_index = temporal_index.cx[
            x_slice.start : x_slice.stop, y_slice.start : y_slice.stop
        ]

        return not spatiotemporal_index.empty

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Return spatiotemporal coordinates to index a dataset.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        if self.mode == 'product':
            # Use pre-computed valid samples
            for sample in self.valid_samples:
                yield sample
        else:  # zip mode
            # Pair spatial and temporal samples, validating each combination
            spatial_iter = iter(self.spatial_sampler)
            temporal_iter = iter(self.temporal_sampler)

            try:
                while True:
                    # For zip mode, we need to find valid combinations
                    # by potentially skipping invalid ones
                    retries = 0
                    while retries < self.max_retries:
                        try:
                            x_slice, y_slice = next(spatial_iter)
                            t_slice = next(temporal_iter)
                        except StopIteration:
                            return

                        if self._has_data_at_query(x_slice, y_slice, t_slice):
                            yield x_slice, y_slice, t_slice
                            break
                        else:
                            retries += 1

                    if retries >= self.max_retries:
                        # If we can't find valid samples, stop iteration
                        return

            except StopIteration:
                pass

    def __len__(self) -> int:
        """Return the number of spatiotemporal samples.

        Returns:
            number of samples
        """
        if self.mode == 'product':
            return len(self.valid_samples)
        else:  # zip mode
            # For zip mode, we can't easily pre-compute the length since
            # we filter out invalid combinations during iteration
            # Return the minimum as an upper bound estimate
            return min(len(self.spatial_sampler), len(self.temporal_sampler))


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: Polygon | None = None,
        toi: pd.Interval | None = None,
        units: Units = Units.PIXELS,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        .. versionadded:: 0.7
           The *generator* parameter.

        .. versionadded:: 0.8
           The *toi* parameter.

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            generator: pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, roi, toi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.generator = generator
        self.length = 0
        self.bounds = []
        self.intervals = []
        areas = []
        for hit in range(len(self.index)):
            bounds = self.index.geometry.iloc[hit].bounds
            xmin, ymin, xmax, ymax = bounds
            tmin, tmax = self.index.index[hit].left, self.index.index[hit].right
            if xmax - xmin >= self.size[1] and ymax - ymin >= self.size[0]:
                if xmax > xmin and ymax > ymin:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.bounds.append(bounds)
                self.intervals.append(pd.Interval(tmin, tmax))
                areas.append((xmax - xmin) * (ymax - ymin))

        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Return the index of a dataset.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            bounds = self.bounds[idx]
            interval = self.intervals[idx]

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(
                bounds, self.size, self.res, self.generator
            )

            yield *bounding_box, slice(interval.left, interval.right)

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float | None = None,
        roi: Polygon | None = None,
        toi: pd.Interval | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionadded:: 0.8
           The *toi* parameter.

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch (defaults to *size*)
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi, toi)
        self.size = _to_tuple(size)
        if stride is not None:
            self.stride = _to_tuple(stride)
        else:
            self.stride = self.size

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])
            self.stride = (self.stride[0] * self.res[1], self.stride[1] * self.res[0])

        self.length = 0
        for i in range(len(self.index)):
            bounds = self.index.geometry.iloc[i].bounds
            xmin, ymin, xmax, ymax = bounds
            if xmax - xmin < self.size[1] or ymax - ymin < self.size[0]:
                continue
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Return the index of a dataset.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        # For each tile...
        for i in range(len(self.index)):
            bounds = self.index.geometry.iloc[i].bounds
            xmin, ymin, xmax, ymax = bounds
            if xmax - xmin < self.size[1] or ymax - ymin < self.size[0]:
                continue
            tmin, tmax = self.index.index[i].left, self.index.index[i].right
            rows, cols = tile_to_chips(bounds, self.size, self.stride)

            # For each row...
            for i in range(rows):
                ymin = bounds[1] + i * self.stride[0]
                ymax = ymin + self.size[0]

                # For each column...
                for j in range(cols):
                    xmin = bounds[0] + j * self.stride[1]
                    xmax = xmin + self.size[1]

                    yield slice(xmin, xmax), slice(ymin, ymax), slice(tmin, tmax)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


class PreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.NonGeoDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Polygon | None = None,
        toi: pd.Interval | None = None,
        shuffle: bool = False,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.3

        .. versionadded:: 0.7
           The *generator* parameter.

        .. versionadded:: 0.8
           The *toi* parameter.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
            generator: pseudo-random number generator (PRNG) used in combination with
                shuffle.
        """
        super().__init__(dataset, roi, toi)
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Return the index of a dataset.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = partial(torch.randperm, generator=self.generator)

        for idx in generator(len(self)):
            i = int(idx)
            xmin, ymin, xmax, ymax = self.index.geometry.iloc[i].bounds
            tmin, tmax = self.index.index[i].left, self.index.index[i].right
            yield slice(xmin, xmax), slice(ymin, ymax), slice(tmin, tmax)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.index)
