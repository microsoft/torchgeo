# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Levircd datamodule."""

from ..datasets import LEVIRCDPlus
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.utils import dataset_split
from torchvision.transforms import Compose
import kornia.augmentation as K

from .geo import NonGeoDataModule

class LEVIRCDPlusDataModule(NonGeoDataModule):
    def __init__(
            self, 
            batch_size=8, 
            num_workers=0, 
            patch_size=256, 
            val_split_pct=0.2, 
            **kwargs
            ):
        super().__init__()
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
    
    def preprocess(self, sample):
        sample["image1"] = (sample["image1"] / 255.0).float()
        sample["image2"] = (sample["image2"] / 255.0).float()
        # Kornia adds batch dimension which we need to remove
        sample["image1"] = K.Normalize(mean=self.mean, std=self.std)(sample["image1"]).squeeze(0)
        sample["image2"] = K.Normalize(mean=self.mean, std=self.std)(sample["image2"]).squeeze(0)
        sample["mask"] = sample["mask"].long()
        return sample

    def train_augmentations(self, batch):
        augmentations = AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomCrop(self.patch_size),
            K.RandomSharpness(p=0.5),
            data_keys=["image1", "image2", "mask"],
        )
        return augmentations(batch)

    def on_after_batch_transfer(self, batch, batch_idx):
        if self.trainer and self.trainer.training:
            batch["mask"] = batch["mask"].float().unsqueeze(1)
            batch = self.train_augmentations(batch)
            batch["mask"] = batch["mask"].squeeze(1).long()
        return batch

    def prepare_data(self):
        LEVIRCDPlus(split="train", **self.kwargs)
        LEVIRCDPlus(split="test", **self.kwargs)

    def setup(self, stage=None):
        train_transforms = Compose([self.preprocess])
        test_transforms = Compose([self.preprocess])

        train_dataset = LEVIRCDPlus(split="train", transforms=train_transforms, **self.kwargs)

        if self.val_split_pct > 0.0:
            self.train_dataset, self.val_dataset, _ = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = LEVIRCDPlus(
            split="test", transforms=test_transforms, **self.kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )