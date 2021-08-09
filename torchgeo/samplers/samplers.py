"""TorchGeo samplers."""

import abc
import random
from typing import Any, Iterator, Tuple, Union

from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


def _to_tuple(value: Union[Tuple[Any, Any], Any]) -> Tuple[Any, Any]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (float, int)):
        return (value, value)
    else:
        return value


class GeoSampler(Sampler[Tuple[Any, ...]], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self, roi: BoundingBox, size: Union[Tuple[float, float], float], length: int
    ) -> None:
        """Initialize a new RandomGeoSampler.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
            size: dimensions of each :term:`patch` in units of CRS
            length: number of random samples to draw per epoch
        """
        self.roi = roi
        self.size = _to_tuple(size)
        self.length = length

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            minx = random.uniform(self.roi.minx, self.roi.maxx - self.size[1])
            maxx = minx + self.size[1]

            miny = random.uniform(self.roi.miny, self.roi.maxy - self.size[0])
            maxy = miny + self.size[0]

            # TODO: figure out how to handle time
            mint = self.roi.mint
            maxt = self.roi.maxt

            yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

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
        roi: BoundingBox,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
    ) -> None:
        """Initialize a new RandomGeoSampler.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
            size: dimensions of each :term:`patch` in units of CRS
            stride: distance to skip between each patch
        """
        self.roi = roi
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.rows = int((roi.maxy - roi.miny - self.size[0]) // self.stride[0]) + 1
        self.cols = int((roi.maxx - roi.minx - self.size[1]) // self.stride[1]) + 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for i in range(self.rows):
            miny = self.roi.miny + i * self.stride[0]
            maxy = miny + self.size[0]
            for j in range(self.cols):
                minx = self.roi.minx + j * self.stride[1]
                maxx = minx + self.size[1]

                # TODO: figure out how to handle time
                mint = self.roi.mint
                maxt = self.roi.maxt

                yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.rows * self.cols
