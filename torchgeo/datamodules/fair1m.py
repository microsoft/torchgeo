# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FAIR1M datamodule."""

from typing import Any

import torch

from ..datasets import FAIR1M, Sample
from .geo import NonGeoDataModule


def collate_fn(batch: list[Sample]) -> Sample:
    """Custom object detection collate fn to handle variable boxes.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output

    .. versionadded:: 0.5
    """
    output: Sample = {}
    output['image'] = torch.stack([sample['image'] for sample in batch])

    if 'boxes' in batch[0]:
        output['boxes'] = [sample['boxes'] for sample in batch]
    if 'label' in batch[0]:
        output['label'] = [sample['label'] for sample in batch]

    return output


class FAIR1MDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FAIR1M dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new FAIR1MDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FAIR1M`.

        .. versionchanged:: 0.5
           Removed *val_split_pct* and *test_split_pct* parameters.
        """
        super().__init__(FAIR1M, batch_size, num_workers, **kwargs)
        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = FAIR1M(split='train', **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = FAIR1M(split='val', **self.kwargs)
        if stage in ['predict']:
            # Test set labels are not publicly available
            self.predict_dataset = FAIR1M(split='test', **self.kwargs)
