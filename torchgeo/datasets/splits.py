# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset splitting utilities."""

from copy import deepcopy
from math import floor
from typing import Any, List, Optional, Sequence, Union

from rtree.index import Index, Property
from torch import Generator, default_generator, randint, randperm
from torch.utils.data import Subset, TensorDataset, random_split

from ..datasets import GeoDataset, NonGeoDataset
from .utils import BoundingBox

__all__ = (
    "random_nongeo_split",
    "random_bbox_assignment",
    "random_bbox_splitting",
    "roi_split",
)


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


def _create_geodataset_like(dataset: GeoDataset, index: Index) -> GeoDataset:
    """Utility to create a new GeoDataset from an existing one with a different index.

    Args:
        dataset: dataset to copy
        index: new index

    Returns:
        A new GeoDataset.

    .. versionadded:: 0.4
    """
    new_dataset = deepcopy(dataset)
    new_dataset.index = index
    return new_dataset


def random_bbox_assignment(
    dataset: GeoDataset,
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> List[GeoDataset]:
    """Split a GeoDataset randomly assigning its index's BoundingBoxes.

    Args:
        dataset: dataset to be split
        lengths: lengths or fractions of splits to be produced
        generator: (optional) generator used for the random permutation

    Returns
        A list of the subset datasets.

    .. versionadded:: 0.4
    """
    if sum(lengths) != 1 or sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths must equal 1 or the length of dataset's index."
        )

    if any(n <= 0 for n in lengths):
        raise ValueError("All items in input lengths must be greater than 0.")

    if sum(lengths) == 1:
        lengths = [floor(frac * len(dataset)) for frac in lengths]
        remainder = int(len(dataset) - sum(lengths))
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(lengths)
            lengths[idx_to_add_at] += 1

    hits = list(dataset.index.intersection(dataset.index.bounds, objects=True))

    hits = [
        hits[i]
        for i in randperm(sum(lengths), generator=generator)  # type: ignore[arg-type]
    ]

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in lengths
    ]

    for i, length in enumerate(lengths):
        for j in range(length):  # type: ignore[arg-type]
            new_indexes[i].insert(j, hits.pop().bounds)

    return [_create_geodataset_like(dataset, index) for index in new_indexes]


def random_bbox_splitting(
    dataset: GeoDataset,
    fractions: Sequence[float],
    generator: Optional[Generator] = default_generator,
) -> List[GeoDataset]:
    """Split a GeoDataset randomly splitting its index's BoundingBoxes.

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
    if sum(fractions) != 1:
        raise ValueError("Sum of input fractions must equal 1.")

    if any(n <= 0 for n in fractions):
        raise ValueError("All items in input fractions must be greater than 0.")

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

    return [_create_geodataset_like(dataset, index) for index in new_indexes]


def roi_split(dataset: GeoDataset, rois: Sequence[BoundingBox]) -> List[GeoDataset]:
    """Split a GeoDataset intersecting it with a ROI for each desired new GeoDataset.

    Args:
        dataset: dataset to be split
        rois: regions of interest of splits to be produced

    Returns
        A list of the subset datasets.

    .. versionadded:: 0.4
    """
    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in rois
    ]

    for i, roi in enumerate(rois):
        for j, hit in enumerate(dataset.index.intersection(tuple(roi), objects=True)):
            box = BoundingBox(*hit.bounds)
            new_indexes[i].insert(j, tuple(box & roi))

    return [_create_geodataset_like(dataset, index) for index in new_indexes]
