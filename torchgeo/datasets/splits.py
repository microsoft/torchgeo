# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset splitting utilities."""

from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from rtree.index import Index, Property
from torch.utils.data import Subset, TensorDataset, random_split

from ..datasets import GeoDataset, NonGeoDataset
from .utils import BoundingBox


def random_nongeo_split(
    dataset: Union[TensorDataset, NonGeoDataset],
    val_pct: float,
    test_pct: Optional[float] = None,
) -> List[Subset[Any]]:
    """Split a torch Dataset into train/val/test sets.

    If ``test_pct`` is not set then only train and validation splits are returned.

    Args:
        dataset: dataset to be split into train/val or train/val/test subsets
        val_pct: percentage of samples to be in validation set
        test_pct: (Optional) percentage of samples to be in test set

    Returns:
        a list of the subset datasets. Either [train, val] or [train, val, test]
    """
    if test_pct is None:
        val_length = round(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        return random_split(dataset, [train_length, val_length])
    else:
        val_length = round(len(dataset) * val_pct)
        test_length = round(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(dataset, [train_length, val_length, test_length])


def random_bbox_splitting(
    dataset: GeoDataset, test_size: float = 0.25, random_seed: Optional[int] = None
) -> Tuple[GeoDataset, GeoDataset]:
    """Splits a dataset into train and test.

    This function will go through each BoundingBox saved in the GeoDataset's index and
    split it in a random direction by the proportion specified in test_size.

    Args:
        dataset: GeoDataset to split
        test_size: proportion of GeoDataset to use for test, in range [0,1]
        random_seed: random seed for reproducibility

    Returns
        A tuple with the resulting GeoDatasets in order (train, test)

    .. versionadded:: 0.4
    """
    assert 0 < test_size < 1, "test_size must be between 0 and 1"

    if random_seed:
        np.random.seed(random_seed)

    index_train = Index(interleaved=False, properties=Property(dimension=3))
    index_test = Index(interleaved=False, properties=Property(dimension=3))

    for i, hit in enumerate(
        dataset.index.intersection(dataset.index.bounds, objects=True)
    ):
        box = BoundingBox(*hit.bounds)
        horizontal, flip = np.random.randint(2, size=2)
        if flip:
            box_train, box_test = box.split(1 - test_size, horizontal)
        else:
            box_test, box_train = box.split(test_size, horizontal)
        index_train.insert(i, tuple(box_train))
        index_test.insert(i, tuple(box_test))

    dataset_train = deepcopy(dataset)
    dataset_train.index = index_train
    dataset_test = deepcopy(dataset)
    dataset_test.index = index_test

    return dataset_train, dataset_test
