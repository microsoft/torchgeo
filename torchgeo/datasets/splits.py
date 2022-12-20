# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset splitting utilities."""

from copy import deepcopy
from typing import Any, List, Optional, Sequence, Union

from rtree.index import Index, Property
from torch import Generator, default_generator, randint
from torch.utils.data import Subset, TensorDataset, random_split

from ..datasets import GeoDataset, NonGeoDataset
from .utils import BoundingBox


def random_nongeo_split(
    dataset: Union[TensorDataset, NonGeoDataset],
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> List[Subset[Any]]:
    """Randomly split a NonGeoDataset into non-overlapping new NonGeoDatasets.

    Args:
        dataset: dataset to be split
        lengths: lengths or fractions of splits to be produced
        generator: (optional) generator used for the random permutation

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.4
    """
    return random_split(dataset, lengths, generator)


def random_bbox_splitting(
    dataset: GeoDataset,
    fractions: Sequence[float],
    generator: Optional[Generator] = default_generator,
) -> List[GeoDataset]:
    """Randomly split a GeoDataset by splitting its index's BoundingBoxes.

    This function will go through each BoundingBox in the GeoDataset's index and
    split it in a random direction.

    Args:
        dataset: dataset to be split
        fractions: fractions of splits to be produced
        generator: (optional) generator used for the random permutation

    Returns
        A list of the subset datasets.

    .. versionadded:: 0.4
    """
    assert sum(fractions) == 1, "fractions must add up to 1"

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in fractions
    ]

    for i, hit in enumerate(
        dataset.index.intersection(dataset.index.bounds, objects=True)
    ):
        box = BoundingBox(*hit.bounds)
        fraction_left = 1.0

        for j, frac in enumerate(fractions):
            horizontal, flip = randint(0, 2, (2,), generator=generator)

            if fraction_left == frac:
                new_box = box
            elif flip:
                box, new_box = box.split((1 - frac) / fraction_left, horizontal)
            else:
                new_box, box = box.split(frac / fraction_left, horizontal)

            new_indexes[j].insert(i, tuple(new_box))
            fraction_left -= frac

    def new_geodataset_like(dataset: GeoDataset, index: Index) -> GeoDataset:
        new_dataset = deepcopy(dataset)
        new_dataset.index = index
        return new_dataset

    return [new_geodataset_like(dataset, index) for index in new_indexes]
