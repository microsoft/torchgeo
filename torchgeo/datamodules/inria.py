from typing import Any, Dict, List, Optional

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from torchgeo.datamodules.utils import dataset_split
from torchgeo.datasets import InriaAerialImageLabeling

DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class InriaAerialImageLabelingDataModule(pl.LightningDataModule):
    """Inria DataModule"""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        patch_size: int = 512,
        num_patches_per_tile: int = 32,
        augmentations=DEFAULT_AUGS,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.patch_size = patch_size
        self.num_patches_per_tile = num_patches_per_tile
        self.augmentations = augmentations
        self.random_crop = K.AugmentationSequential(
            K.RandomCrop((self.patch_size, self.patch_size), p=1.0, keepdim=False),
            data_keys=["input", "mask"],
        )

    def preprocess(self, sample):
        # RGB is int32 so divide by 255
        # TODO: why int32? cant it be uint8
        sample["image"] = sample["image"] / 255.0
        sample["image"] = torch.clip(sample["image"], min=0.0, max=1.0)

        # This is pointless since it will get squeezed out anyway
        if "mask" in sample:
            sample["mask"] = rearrange(sample["mask"], "h w -> () h w")

        return sample

    def crop(self, sample):
        sample["mask"] = sample["mask"].to(torch.float)
        sample["image"], sample["mask"] = self.random_crop(
            sample["image"], sample["mask"]
        )
        sample["mask"] = sample["mask"].to(torch.long)
        sample["image"] = rearrange(sample["image"], "() c h w -> c h w")
        sample["mask"] = rearrange(sample["mask"], "() c h w -> c h w")
        return sample

    def n_random_crop(self, sample):
        images, masks = [], []
        for _ in range(self.num_patches_per_tile):
            image, mask = sample["image"], sample["mask"]
            # RandomCrop needs image and mask to be in float
            mask = mask.to(torch.float)
            image, mask = self.random_crop(image, mask)
            images.append(image.squeeze())
            masks.append(mask.squeeze(0).long())
        sample["image"] = torch.stack(images)  # (t,c,h,w)
        sample["mask"] = torch.stack(masks)  # (t, 1, h, w)
        return sample

    def setup(self, stage=None):
        # transforms = T.Compose([self.preprocess, self.crop])
        train_transforms = T.Compose([self.preprocess, self.n_random_crop])
        val_transforms = T.Compose([self.preprocess, self.crop])
        test_transforms = T.Compose([self.preprocess])

        train_dataset = InriaAerialImageLabeling(
            self.root_dir, split="train", transforms=train_transforms
        )

        if self.val_split_pct > 0.0:
            val_dataset = InriaAerialImageLabeling(
                self.root_dir, split="train", transforms=val_transforms
            )
            self.train_dataset, self.val_dataset = dataset_split(
                train_dataset, val_pct=self.val_split_pct
            )
            self.val_dataset.dataset = val_dataset
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = InriaAerialImageLabeling(
            self.root_dir, "test", transforms=test_transforms
        )

        # self.predict_dataset = InriaAerialImageLabeling(
        #     self.root_dir, self.predict_on, transforms=test_transforms
        # )

    def train_dataloader(self):
        """Return a DataLoader for training."""

        def collate_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            r_batch: Dict[str, Any] = default_collate(  # type: ignore[no-untyped-call]
                batch
            )
            r_batch["image"] = torch.flatten(  # type: ignore[attr-defined]
                r_batch["image"], 0, 1
            )
            r_batch["mask"] = torch.flatten(  # type: ignore[attr-defined]
                r_batch["mask"], 0, 1
            )
            return r_batch

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    # def predict_dataloader(self):
    #     return DataLoader(
    #         self.predict_dataset,
    #         batch_size=1,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.augmentations is not None:
            batch["mask"] = batch["mask"].to(torch.float)
            batch["image"], batch["mask"] = self.augmentations(
                batch["image"], batch["mask"]
            )
            batch["mask"] = batch["mask"].to(torch.long)

        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch
