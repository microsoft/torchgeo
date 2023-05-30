# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L8 Biome datamodule."""

from typing import Any, Optional, Union

from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader

from ..datasets import L8Biome, random_bbox_assignment, TileDataset
from ..samplers import GridGeoSampler, RandomBatchGeoSampler, RandomTileGeoSampler, GridTileGeoSampler
from .geo import GeoDataModule


class L8BiomeDataModule(GeoDataModule):
    """LightningDataModule implementation for the L8 Biome dataset.

    .. versionadded:: 0.5
    """

    mean = torch.tensor(0)
    std = torch.tensor(10000)

    def __init__(
        self,
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 32,
        length: Optional[int] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new L8BiomeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.L8Biome`.
        """
        super().__init__(
            L8Biome,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = L8Biome(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_assignment(dataset, [0.6, 0.2, 0.2], generator)

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

class L8BiomeTileDataModule(LightningDataModule):

    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"] / 255.0

        mask_mapping = {64: 1, 128: 2, 192: 3, 255: 4}
        if "mask" in sample:
            mask = sample["mask"].squeeze()
            for k, v in mask_mapping.items():
                mask[mask == k] = v
            sample["mask"] = mask
        return sample

    def _get_all_the_fns(self, root):
        import os
        areas = L8Biome.filenames_to_md5.keys()
        image_fns = []
        mask_fns = []
        for area in areas:
            for scene_idx in os.listdir(os.path.join(root,area)):
                image_fns.append(os.path.join(root,area,scene_idx,f"{scene_idx}.TIF"))
                mask_fns.append(os.path.join(root,area,scene_idx,f"{scene_idx}_fixedmask.TIF"))
        return image_fns, mask_fns

    def __init__(self, root, batch_size=1, patch_size=32, train_batches_per_epoch=None, val_batches_per_epoch=None, num_workers=0, seed=0):
        super().__init__()
        self.image_fns, self.mask_fns = self._get_all_the_fns(root)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        self.num_workers = num_workers

        generator = torch.Generator().manual_seed(seed)

        idxs = torch.randperm(len(self.image_fns), generator=generator)
        train_idxs = idxs[:int(len(idxs)*0.6)]
        val_idxs = idxs[int(len(idxs)*0.6):int(len(idxs)*0.8)]
        test_idxs = idxs[int(len(idxs)*0.8):]

        self.train_image_fns = [self.image_fns[i] for i in train_idxs]
        self.train_mask_fns = [self.mask_fns[i] for i in train_idxs]
        self.val_image_fns = [self.image_fns[i] for i in val_idxs]
        self.val_mask_fns = [self.mask_fns[i] for i in val_idxs]
        self.test_image_fns = [self.image_fns[i] for i in test_idxs]
        self.test_mask_fns = [self.mask_fns[i] for i in test_idxs]

    def setup(self, stage):
        self.train_dataset = TileDataset(self.train_image_fns, self.train_mask_fns, transforms=L8BiomeTileDataModule.preprocess)
        self.val_dataset = TileDataset(self.val_image_fns, self.val_mask_fns, transforms=L8BiomeTileDataModule.preprocess)
        self.test_dataset = TileDataset(self.test_image_fns, self.test_mask_fns, transforms=L8BiomeTileDataModule.preprocess)

    # def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    #     return super().on_after_batch_transfer(batch, dataloader_idx)

    def train_dataloader(self):
        sampler = RandomTileGeoSampler(self.train_dataset, self.patch_size, self.batch_size * self.train_batches_per_epoch)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        sampler = RandomTileGeoSampler(self.val_dataset, self.patch_size, self.batch_size * self.val_batches_per_epoch)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        sampler = GridTileGeoSampler(self.test_dataset, self.patch_size, self.patch_size)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def plot(self, sample):
        import matplotlib.pyplot as plt
        image = sample["image"].permute(1,2,0).numpy()
        mask = sample["mask"].numpy().squeeze()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image[:,:,[2,1,0]])
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=4)
        axs[1].axis("off")
        return fig