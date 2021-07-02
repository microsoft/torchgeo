import abc
from typing import Any, Iterator, Tuple

from torch.utils.data import Sampler


# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


class GeoSampler(Sampler[Tuple[Any, ...]], abc.ABC):
    """Abstract base class for sampling from :class:`GeoDataset
    <torchgeo.datasets.GeoDataset>`.

    Unlike PyTorch's :class:`Sampler <torch.utils.data.Sampler>`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any :class:`GeoDataset
    <torchgeo.datasets.GeoDataset>`. This includes things like latitude, longitude,
    height, width, projection, coordinate system, and time.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        """Return the index of a dataset.

        Returns:
            tuple of latitude, longitude, height, width, projection, coordinate system,
            and time
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """


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
