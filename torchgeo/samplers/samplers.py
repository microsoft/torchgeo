import abc
import random
from typing import Any, Iterator, Tuple

from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


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

    def __init__(self, roi: BoundingBox, size: int, length: int) -> None:
        """Initialize a new RandomGeoSampler.

        Parameters:
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
            size: dimensions of each :term:`patch` to return
            length: number of random samples to draw per epoch
        """
        self.roi = roi
        self.size = size
        self.length = length

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            minx = random.randint(int(self.roi.minx), int(self.roi.maxx) - self.size)
            maxx = minx + self.size

            miny = random.randint(int(self.roi.miny), int(self.roi.maxy) - self.size)
            maxy = miny + self.size

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
