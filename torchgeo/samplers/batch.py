# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo batch samplers."""

import abc
from collections.abc import Iterator

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


class BatchGeoSampler(Sampler[list[GeoSlice]], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.BatchSampler`, :class:`BatchGeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: Polygon | None = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        self.index = dataset.index
        self.res = dataset.res

        if roi:
            self.roi = roi
            self.index = self.index.clip(roi)
        else:
            bounds = dataset.bounds
            self.roi = shapely.box(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[list[GeoSlice]]:
        """Return a batch of indices of a dataset.

        Yields:
            Batch of [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """


class RandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        batch_size: int,
        length: int | None = None,
        roi: Polygon | None = None,
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

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
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
        self.generator = generator

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.batch_size = batch_size
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

    def __iter__(self) -> Iterator[list[tuple[slice, slice, slice]]]:  # type: ignore[override]
        """Return the indices of a dataset.

        Yields:
            Batch of [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            bounds = self.bounds[idx]
            interval = self.intervals[idx]

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):
                bounding_box = get_random_bounding_box(
                    bounds, self.size, self.res, self.generator
                )
                batch.append((*bounding_box, slice(interval.left, interval.right)))

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size
