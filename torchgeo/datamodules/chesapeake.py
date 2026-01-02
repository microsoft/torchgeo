# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datamodule."""

from typing import Any

import kornia.augmentation as K
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, default_collate

from ..datasets import ChesapeakeCVPRTileDataset
from ..samplers import GridTileSampler, RandomTileSampler
from .geo import BaseDataModule


class ChesapeakeCVPRDataModule(BaseDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets. Uses tile-based sampling for efficient patch extraction without
    geospatial reprojection overhead.
    """

    def __init__(
        self,
        train_splits: list[str],
        val_splits: list[str],
        test_splits: list[str],
        batch_size: int = 64,
        patch_size: int = 256,
        length: int | None = None,
        num_workers: int = 0,
        class_set: int = 7,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a new ChesapeakeCVPRDataModule instance.

        Args:
            train_splits: Splits used to train the model, e.g., ["ny-train"].
            val_splits: Splits used to validate the model, e.g., ["ny-val"].
            test_splits: Splits used to test the model, e.g., ["ny-test"].
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            class_set: The high-resolution land cover class set to use (5 or 7).
            use_prior_labels: Flag for using a prior over high-resolution classes
                instead of the high-resolution labels themselves.
            prior_smoothing_constant: Additive smoothing to add when using prior labels.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ChesapeakeCVPRTileDataset`.

        Raises:
            ValueError: If ``use_prior_labels=True`` is used with ``class_set=7``.
        """
        super().__init__(ChesapeakeCVPRTileDataset, batch_size, num_workers, **kwargs)

        assert class_set in [5, 7]
        if use_prior_labels and class_set == 7:
            raise ValueError(
                'The pre-generated prior labels are only valid for the 5'
                + ' class set of labels'
            )

        self.patch_size = patch_size
        self.length = length
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.class_set = class_set
        self.use_prior_labels = use_prior_labels
        self.prior_smoothing_constant = prior_smoothing_constant

        if self.use_prior_labels:
            self.layers = [
                'naip-new',
                'prior_from_cooccurrences_101_31_no_osm_no_buildings',
            ]
        else:
            self.layers = ['naip-new', 'lc']

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )

        self.collate_fn = default_collate

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = ChesapeakeCVPRTileDataset(
                splits=self.train_splits, layers=self.layers, **self.kwargs
            )
            self.train_sampler = RandomTileSampler(
                self.train_dataset,
                self.patch_size,
                self.length or len(self.train_dataset) * 100,
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = ChesapeakeCVPRTileDataset(
                splits=self.val_splits, layers=self.layers, **self.kwargs
            )
            self.val_sampler = GridTileSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ['test']:
            self.test_dataset = ChesapeakeCVPRTileDataset(
                splits=self.test_splits, layers=self.layers, **self.kwargs
            )
            self.test_sampler = GridTileSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Create a DataLoader for the specified split.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A DataLoader for the specified split.
        """
        dataset = self._valid_attribute(f'{split}_dataset', 'dataset')
        sampler = self._valid_attribute(f'{split}_sampler', 'sampler')
        batch_size = self._valid_attribute(f'{split}_batch_size', 'batch_size')

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == 'train',
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for training.

        Returns:
            A DataLoader for training.
        """
        return self._dataloader_factory('train')

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for validation.

        Returns:
            A DataLoader for validation.
        """
        return self._dataloader_factory('val')

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for testing.

        Returns:
            A DataLoader for testing.
        """
        return self._dataloader_factory('test')

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.use_prior_labels:
            batch['mask'] = F.normalize(batch['mask'].float(), p=1, dim=1)
            batch['mask'] = F.normalize(
                batch['mask'] + self.prior_smoothing_constant, p=1, dim=1
            ).long()
        else:
            if self.class_set == 5:
                batch['mask'][batch['mask'] == 5] = 4
                batch['mask'][batch['mask'] == 6] = 4

        return super().on_after_batch_transfer(batch, dataloader_idx)
