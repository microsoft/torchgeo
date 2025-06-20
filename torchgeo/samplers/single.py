# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
from collections.abc import Callable, Iterable, Iterator
from functools import partial

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
