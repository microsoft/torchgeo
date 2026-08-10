# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Biomass-S3 datamodule."""

from collections.abc import Callable
from typing import Any, cast

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample

from ...datasets import CopernicusBenchBiomassS3
from ..geo import NonGeoDataModule

# Multiplicative scale factors from
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI#bands
SCALE = {
    'Oa01_radiance': 0.0139465,
    'Oa02_radiance': 0.0133873,
    'Oa03_radiance': 0.0121481,
    'Oa04_radiance': 0.0115198,
    'Oa05_radiance': 0.0100953,
    'Oa06_radiance': 0.0123538,
    'Oa07_radiance': 0.00879161,
    'Oa08_radiance': 0.00876539,
    'Oa09_radiance': 0.0095103,
    'Oa10_radiance': 0.00773378,
    'Oa11_radiance': 0.00675523,
    'Oa12_radiance': 0.0071996,
    'Oa13_radiance': 0.00749684,
    'Oa14_radiance': 0.0086512,
    'Oa15_radiance': 0.00526779,
    'Oa16_radiance': 0.00530267,
    'Oa17_radiance': 0.00493004,
    'Oa18_radiance': 0.00549962,
    'Oa19_radiance': 0.00502847,
    'Oa20_radiance': 0.00326378,
    'Oa21_radiance': 0.00324118,
}

TARGET_SIZE = (282, 282)
# Dataset-wide biomass statistics after resizing masks to TARGET_SIZE.
TARGET_MEAN = torch.tensor(93.197777079690, dtype=torch.float32)
TARGET_STD = torch.tensor(119.004185235754, dtype=torch.float32)


def _collate_time_series_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a time-series batch by padding temporal dimensions to a common length."""
    lengths = [sample['image'].shape[0] for sample in batch]
    max_length = max(lengths)
    batch_size = len(batch)

    collated: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        value = values[0]

        is_time_series_tensor = value.ndim > 0 and all(
            item.shape[0] == length for item, length in zip(values, lengths)
        )
        if is_time_series_tensor:
            padded = value.new_zeros((batch_size, max_length, *value.shape[1:]))
            for i, item in enumerate(values):
                padded[i, : item.shape[0]] = item
            collated[key] = padded
        else:
            collated[key] = torch.stack(values)

    return collated


class CopernicusBenchBiomassS3DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Copernicus Biomass-S3 dataset.

    Uses the train/val/test splits provided with the benchmark.

    .. versionadded:: 0.10
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new CopernicusBenchBiomassS3DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CopernicusBenchBiomassS3`.
        """
        bands = kwargs.get('bands', CopernicusBenchBiomassS3.all_bands)
        mode = kwargs.get('mode', 'static')
        scale_factors = torch.tensor([SCALE[b] for b in bands], dtype=torch.float32)

        self.mean = torch.zeros(len(bands), dtype=torch.float32)
        self.std = torch.reciprocal(scale_factors)
        self.target_mean = TARGET_MEAN.clone()
        self.target_std = TARGET_STD.clone()

        resize_transform = K.AugmentationSequential(
            K.Resize(size=TARGET_SIZE, resample=Resample.BILINEAR, align_corners=False),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {'resample': Resample.BILINEAR, 'align_corners': None}
            },
        )
        existing_transform = cast(
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None,
            kwargs.get('transforms'),
        )

        if existing_transform is not None:
            transform = existing_transform

            def composed(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                resized = resize_transform(sample)
                return transform(resized)

            kwargs['transforms'] = composed
        else:
            kwargs['transforms'] = resize_transform

        super().__init__(CopernicusBenchBiomassS3, batch_size, num_workers, **kwargs)

        normalizer = K.Normalize(mean=self.mean, std=self.std)
        if mode == 'time-series':
            self.collate_fn = _collate_time_series_batch
            self.aug = K.AugmentationSequential(
                K.VideoSequential(normalizer),
                data_keys=None,
                keepdim=True,
                same_on_batch=True,
            )
        else:
            self.aug = K.AugmentationSequential(
                normalizer, data_keys=None, keepdim=True
            )

    def transfer_batch_to_device(
        self, batch: dict[str, torch.Tensor], device: torch.device, dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Transfer batch and statistics to device.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.
        """
        self.target_mean = self.target_mean.to(device)
        self.target_std = self.target_std.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_after_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Normalize imagery and biomass targets.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch with normalized imagery and target masks.
        """
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        if 'mask' in batch:
            batch['mask'] = (batch['mask'] - self.target_mean) / self.target_std
        return batch
