# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common datamodule utilities."""

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from ..datasets.utils import Sample


# Based on lightning_lite.utilities.exceptions
class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


def collate_fn_detection(batch: list[Sample]) -> Sample:
    """Custom collate fn for object detection and instance segmentation.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output

    .. versionadded:: 0.6
    """
    output: Sample = {}
    output['image'] = torch.stack([sample['image'] for sample in batch])
    output['bbox_xyxy'] = [sample['bbox_xyxy'].float() for sample in batch]
    if 'label' in batch[0].keys():
        output['label'] = [sample['label'] for sample in batch]
    else:
        output['label'] = [
            torch.tensor([1] * len(sample['bbox_xyxy'])) for sample in batch
        ]

    if 'mask' in batch[0]:
        output['mask'] = [sample['mask'] for sample in batch]
    return output


def group_shuffle_split(
    groups: Iterable[Any],
    train_size: float | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[list[int], list[int]]:
    """Method for performing a single group-wise shuffle split of data.

    Loosely based off of :class:`sklearn.model_selection.GroupShuffleSplit`.

    Args:
        groups: a sequence of group values used to split. Should be in the same order as
            the data you want to split.
        train_size: the proportion of groups to include in the train split. If None,
            then it is set to complement `test_size`.
        test_size: the proportion of groups to include in the test split (rounded up).
            If None, then it is set to complement `train_size`.
        random_state: controls the random splits (passed a seed to a
            numpy.random.Generator), set for reproducible splits.

    Returns:
        train_indices, test_indices

    Raises:
        ValueError if `train_size` and `test_size` do not sum to 1, aren't in the range
            (0,1), or are both None.
        ValueError if the number of training or testing groups turns out to be 0.
    """
    if train_size is None and test_size is None:
        raise ValueError('You must specify `train_size`, `test_size`, or both.')
    if (train_size is not None and test_size is not None) and (
        not math.isclose(train_size + test_size, 1)
    ):
        raise ValueError('`train_size` and `test_size` must sum to 1.')

    if train_size is None and test_size is not None:
        train_size = 1 - test_size
    if test_size is None and train_size is not None:
        test_size = 1 - train_size

    assert train_size is not None and test_size is not None

    if train_size <= 0 or train_size >= 1 or test_size <= 0 or test_size >= 1:
        raise ValueError('`train_size` and `test_size` must be in the range (0,1).')

    group_vals = sorted(set(groups))
    n_groups = len(group_vals)
    n_test_groups = round(n_groups * test_size)
    n_train_groups = n_groups - n_test_groups

    if n_train_groups == 0 or n_test_groups == 0:
        raise ValueError(
            f'{n_groups} groups were found, however the current settings of '
            + '`train_size` and `test_size` result in 0 training or testing groups.'
        )

    generator = np.random.default_rng(seed=random_state)
    train_group_vals = set(
        generator.choice(group_vals, size=n_train_groups, replace=False)
    )

    train_idxs = []
    test_idxs = []
    for i, group_val in enumerate(groups):
        if group_val in train_group_vals:
            train_idxs.append(i)
        else:
            test_idxs.append(i)

    return train_idxs, test_idxs
