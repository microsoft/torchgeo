# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common datamodule utilities."""

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Module


# Based on lightning_lite.utilities.exceptions
class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


class AugPipe(Module):
    """Pipeline for applying augmentations sequentially on select data keys.

    .. versionadded:: 0.6
    """

    def __init__(
        self, augs: Callable[[dict[str, Any]], dict[str, Any]], batch_size: int
    ) -> None:
        """Initialize a new AugPipe instance.

        Args:
            augs: Augmentations to apply.
            batch_size: Batch size
        """
        super().__init__()
        self.augs = augs
        self.batch_size = batch_size

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply the augmentation.

        Args:
            batch: Input batch.

        Returns:
            Augmented batch.
        """
        batch_len = len(batch['image'])
        for bs in range(batch_len):
            batch_dict = {
                'image': batch['image'][bs],
                'labels': batch['labels'][bs],
                'boxes': batch['boxes'][bs],
            }

            if 'masks' in batch:
                batch_dict['masks'] = batch['masks'][bs]

            batch_dict = self.augs(batch_dict)

            batch['image'][bs] = batch_dict['image']
            batch['labels'][bs] = batch_dict['labels']
            batch['boxes'][bs] = batch_dict['boxes']

            if 'masks' in batch:
                batch['masks'][bs] = batch_dict['masks']

        # Stack images
        batch['image'] = rearrange(batch['image'], 'b () c h w -> b c h w')

        return batch


def collate_fn_detection(batch: list[dict[str, Tensor]]) -> dict[str, Any]:
    """Custom collate fn for object detection and instance segmentation.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output

    .. versionadded:: 0.6
    """
    output: dict[str, Any] = {}
    output['image'] = [sample['image'] for sample in batch]
    output['boxes'] = [sample['boxes'].float() for sample in batch]
    if 'labels' in batch[0]:
        output['labels'] = [sample['labels'] for sample in batch]
    else:
        output['labels'] = [
            torch.tensor([1] * len(sample['boxes'])) for sample in batch
        ]

    if 'masks' in batch[0]:
        output['masks'] = [sample['masks'] for sample in batch]
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


def get_prefixed_kwargs(prefix: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
    """Get kwargs with a specific prefix.

    Args:
        prefix: Prefix to filter kwargs by.
        **kwargs: Keyword arguments to filter.

    Returns:
        Dictionary of kwargs with the specified prefix.
    """
    return {k.replace(prefix, ''): v for k, v in kwargs.items() if k.startswith(prefix)}
