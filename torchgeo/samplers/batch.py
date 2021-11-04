# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo batch samplers."""

import abc
import random
from typing import Iterator, List, Optional, Tuple, Union

from torch.utils.data import Sampler

from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox

from .utils import _to_tuple, get_random_bounding_box

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


class BatchGeoSampler(Sampler[List[BoundingBox]], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.BatchSampler`, :class:`BatchGeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.

    When sampling from :class:`~torchgeo.datasets.ZipDataset`, the ``dataset`` should be
    a tile-based dataset if possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch` in units of CRS
            batch_size: number of samples per batch
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        self.index = dataset.index
        self.res = dataset.res
        self.size = _to_tuple(size)
        self.batch_size = batch_size
        self.length = length
        if roi is None:
            roi = BoundingBox(*self.index.bounds)
        self.roi = roi
        self.hits = list(self.index.intersection(roi, objects=True))

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile
            hit = random.choice(self.hits)
            bounds = BoundingBox(*hit.bounds)

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):

                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                batch.append(bounding_box)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size
