# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo sampler base classes."""

import abc
import warnings
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import Literal

import numpy as np
import shapely
import shapely.plotting
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pandas import Interval, IntervalIndex
from shapely import MultiPolygon, Polygon
from torch.utils.data import Sampler

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice
from .utils import prism


class GeoSampler(Sampler[GeoSlice], ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns a GeoSlice that can uniquely index any :class:`~torchgeo.datasets.GeoDataset`.
    """

    _length: int

    @abc.abstractmethod
    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """

    def __len__(self) -> int:
        """Length of each epoch.

        Returns:
            The sampler length.
        """
        if not hasattr(self, '_length'):
            # Use brute force to calculate length if not already cached
            self._length = sum(1 for _ in self)

        return self._length


class SpatialSampler(GeoSampler):
    """Abstract base class for all spatial sampling strategies.

    .. versionadded:: 0.10
    """

    @property
    @abc.abstractmethod
    def strategy(self) -> Literal['random', 'sequential']:
        """Sampling strategy.

        All sampling strategies can be categorized as either being random or sequential.
        This distinction only matters when combining samplers via
        :class:`SpatioTemporalSampler`, where either a zip (random) or product
        (sequential) of all sample locations is taken during each epoch.

        Returns:
            One of 'random' or 'sequential'.
        """

    def __init__(self, dataset: GeoDataset, *, roi: Polygon | None = None) -> None:
        """Initialize a new SpatialSampler instance.

        Args:
            dataset: Dataset to sample from.
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
        """
        # Create one single MultiPolygon of all objects
        # Allows all locations to be equally weighted, regardless of # time stamps
        self.geometry = dataset.index.geometry.union_all()
        self.bounds = self.geometry.bounds
        self.res = dataset.res

        if roi is not None:
            self.geometry &= roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """

    def __matmul__(self, other: 'TemporalSampler') -> 'SpatioTemporalSampler':
        """Compute the product of two samplers.

        Args:
            other: A temporal sampling strategy.

        Returns:
            A single spatial and temporal sampler.
        """
        return SpatioTemporalSampler(self, other)

    def plot(self) -> FuncAnimation:
        """Plot a visualization of the sampling strategy.

        Returns:
            An animation visualizing the sampling strategy.

        Raises:
            AssertionError: If *self.geometry* is not a Polygon.
        """
        geometry = self.geometry
        assert isinstance(geometry, Polygon | MultiPolygon)
        xmin, ymin, xmax, ymax = geometry.bounds

        fig, ax = plt.subplots()
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')

        def init_func() -> Iterable[Artist]:
            """Plot the static dataset."""
            return shapely.plotting.plot_polygon(geometry, ax=ax)

        def func(index: tuple[slice, slice]) -> Iterable[Rectangle]:
            """Plot the dynamic samples."""
            x, y = index
            xy = (x.start, y.start)
            width = x.stop - x.start
            height = y.stop - y.start
            patch = Rectangle(xy, width, height, color='tab:orange', alpha=0.3)
            ax.add_patch(patch)
            return [patch]

        return FuncAnimation(fig, func=func, frames=self, init_func=init_func)


class TemporalSampler(GeoSampler):
    """Abstract base class for all temporal sampling strategies.

    .. versionadded:: 0.10
    """

    @property
    @abc.abstractmethod
    def strategy(self) -> Literal['random', 'sequential']:
        """Sampling strategy.

        All sampling strategies can be categorized as either being random or sequential.
        This distinction only matters when combining samplers via
        :class:`SpatioTemporalSampler`, where either a zip (random) or product
        (sequential) of all sample locations is taken during each epoch.

        Returns:
            One of 'random' or 'sequential'.
        """

    def __init__(self, dataset: GeoDataset, *, toi: Interval | None = None) -> None:
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: Dataset to sample from.
            toi: Time of interest to sample from
                (defaults to the bounds of ``dataset.index``).
        """
        self.index = dataset.index

        if toi is not None:
            tmin = np.maximum(toi.left.to_datetime64(), self.index.index.left)
            tmax = np.minimum(toi.right.to_datetime64(), self.index.index.right)
            valid = tmax >= tmin
            tmin = tmin[valid]
            tmax = tmax[valid]
            self.index = self.index[valid]
            self.index.index = IntervalIndex.from_arrays(
                tmin, tmax, closed='both', name='datetime'
            )

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        yield from self._iter_subset()

    def _init_subset(
        self, location: tuple[slice, slice] = (slice(None), slice(None))
    ) -> IntervalIndex:
        """Narrow down index to a specific location.

        Args:
            location: A specific location.

        Returns:
            A subset of *self.index* at *location*.
        """
        index = self.index

        # Since this only occurs in combination with a SpatialSampler, x and y are
        # guaranteed to have start and stop, and t is guaranteed to be empty
        x, y = location
        index = index.cx[x.start : x.stop, y.start : y.stop]

        return index.index  # ty: ignore[invalid-return-type]

    @abc.abstractmethod
    def _iter_subset(
        self, location: tuple[slice, slice] = (slice(None), slice(None))
    ) -> Iterator[tuple[slice, slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Args:
            location: Region of interest to sample from.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """

    def plot(self) -> FuncAnimation:
        """Plot a visualization of the sampling strategy.

        Returns:
            An animation visualizing the sampling strategy.
        """
        tmin = self.index.index.left.min()
        tmax = self.index.index.right.max()

        fig, ax = plt.subplots()
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('t')
        ax.set_xlim(tmin, tmax)
        ax.yaxis.set_visible(False)
        ax.spines[['left', 'top', 'right']].set_visible(False)
        fig.autofmt_xdate()

        def init_func() -> Iterable[Artist]:
            """Plot the static dataset."""
            patches = []
            for interval in self.index.index:
                patch = ax.axvspan(
                    interval.left, interval.right, color='tab:blue', alpha=0.3
                )
                patches.append(patch)
            return patches

        def func(index: tuple[slice, slice, slice]) -> Iterable[Rectangle]:
            """Plot the dynamic samples."""
            _, _, t = index
            patch = ax.axvspan(t.start, t.stop, color='tab:orange', alpha=0.3)
            ax.add_patch(patch)
            return [patch]

        return FuncAnimation(fig, func=func, frames=self, init_func=init_func)


class SpatioTemporalSampler(GeoSampler):
    """Product of a spatial and a temporal sampler.

    .. versionadded:: 0.10
    """

    def __init__(
        self, spatial_sampler: SpatialSampler, temporal_sampler: TemporalSampler
    ) -> None:
        """Initialize a new SpatioTemporalSampler instance.

        Args:
            spatial_sampler: A spatial sampling strategy.
            temporal_sampler: A temporal sampling strategy.
        """
        self.spatial_sampler = spatial_sampler
        self.temporal_sampler = temporal_sampler

        match self.spatial_sampler.strategy, self.temporal_sampler.strategy:
            case 'random', 'sequential':
                msg = 'random_sampler @ sequential_sampler may result in a different '
                msg += 'number of samples per epoch if different random locations have '
                msg += 'a different number of timestamps'
                warnings.warn(msg, UserWarning)

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        match self.spatial_sampler.strategy, self.temporal_sampler.strategy:
            case 'random', 'random':
                spatial_iter = iter(self.spatial_sampler)
                for _ in range(len(self.spatial_sampler)):
                    location = next(spatial_iter)
                    yield next(self.temporal_sampler._iter_subset(location))
            case 'sequential', 'sequential':
                for location in self.spatial_sampler:
                    for index in self.temporal_sampler._iter_subset(location):
                        yield index
            case 'random', 'sequential':
                spatial_iter = iter(self.spatial_sampler)
                for _ in range(len(self.spatial_sampler)):
                    location = next(spatial_iter)
                    for index in self.temporal_sampler._iter_subset(location):
                        yield index
            case 'sequential', 'random':
                for location in self.spatial_sampler:
                    for _ in range(len(self.temporal_sampler)):
                        yield next(self.temporal_sampler._iter_subset(location))

    def plot(self) -> FuncAnimation:
        """Plot a visualization of the sampling strategy.

        Returns:
            An animation visualizing the sampling strategy.
        """
        spatial = self.spatial_sampler
        temporal = self.temporal_sampler

        xmin, ymin, xmax, ymax = spatial.geometry.bounds
        tmin = temporal.index.index.left.min().timestamp()  # Timestamp not supported
        tmax = temporal.index.index.right.max().timestamp()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f'{spatial.__class__.__name__} @ {temporal.__class__.__name__}')
        ax.set(xlabel='x', ylabel='y', zlabel='t')
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[tmin, tmax])
        ax.set_aspect('equalxy')

        def init_func() -> Iterable[Artist]:
            """Plot the static dataset."""
            verts = []
            for index, data in temporal.index.iterrows():
                x, y = data['geometry'].exterior.coords.xy
                tmin = index.left.timestamp()
                tmax = index.right.timestamp()
                t = np.array([tmin, tmax])
                verts.extend(prism(x, y, t))
            poly = Poly3DCollection(verts, color='tab:blue', alpha=0.3)
            return ax.add_collection3d(poly)

        def func(index: tuple[slice, slice, slice]) -> Iterable[Artist]:
            """Plot the dynamic samples."""
            x = np.array(
                [
                    index[0].start,
                    index[0].start,
                    index[0].stop,
                    index[0].stop,
                    index[0].start,
                ]
            )
            y = np.array(
                [
                    index[1].start,
                    index[1].stop,
                    index[1].stop,
                    index[1].start,
                    index[1].start,
                ]
            )
            t = np.array([index[2].start.timestamp(), index[2].stop.timestamp()])
            verts = prism(x, y, t)
            poly = Poly3DCollection(verts, color='tab:orange', alpha=0.3)
            return ax.add_collection3d(poly)

        return FuncAnimation(fig, func=func, frames=self, init_func=init_func)
