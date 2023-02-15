# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common datamodule utilities."""

from typing import Any, List, Optional, Union

from torch import Generator
from torch.utils.data import Subset, TensorDataset, random_split

from ..datasets import NonGeoDataset


# Based on lightning_lite.utilities.exceptions
class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


def dataset_split(
    dataset: Union[TensorDataset, NonGeoDataset],
    val_pct: float,
    test_pct: Optional[float] = None,
) -> List[Subset[Any]]:
    """Split a torch Dataset into train/val/test sets.

    If ``test_pct`` is not set then only train and validation splits are returned.

    .. deprecated:: 0.4
       Use :func:`torch.utils.data.random_split` instead, ``random_split``
       now supports percentages as of PyTorch 1.13.

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
        return random_split(
            dataset, [train_length, val_length], generator=Generator().manual_seed(0)
        )
    else:
        val_length = round(len(dataset) * val_pct)
        test_length = round(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(
            dataset,
            [train_length, val_length, test_length],
            generator=Generator().manual_seed(0),
        )
