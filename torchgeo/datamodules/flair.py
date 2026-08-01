# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""FLAIRHUB datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from kornia.geometry.transform import resize

from ..datasets import FLAIRHUB, FLAIRHUBToy
from ..datasets.utils import Sample
from .geo import NonGeoDataModule

# From the research paper : https://arxiv.org/abs/2506.07080 p16
# AERIAL_RGBI: R, G, B, NIR bands
AERIAL_RGBI_MEAN = torch.tensor([105.35, 111.31, 102.09, 106.12])
AERIAL_RGBI_STD = torch.tensor([52.36, 45.63, 44.39, 40.01])

# SPOT_RGBI: R, G, B, NIR bands
SPOT_RGBI_MEAN = torch.tensor([432.03, 507.34, 466.75, 1126.26])
SPOT_RGBI_STD = torch.tensor([324.70, 300.28, 239.36, 535.28])

# DEM_ELEV: DSM, DTM bands
DEM_ELEV_MEAN = torch.tensor([326.43, 322.42])
DEM_ELEV_STD = torch.tensor([535.41, 535.75])

# AERIAL_RLT_PAN: Panchromatic band
AERIAL_RLT_PAN_MEAN = torch.tensor([125.92])
AERIAL_RLT_PAN_STD = torch.tensor([38.69])


class FLAIRHUBDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FLAIRHUB dataset.

    Uses official splits from the research paper. You can choose between
    ``split_1``, ``split_2``, ``split_3``, ``split_4``, ``split_5``, and
    ``split_flairchallenge``.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        concatenate_modalities: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FLAIRHUBDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            concatenate_modalities: If True, concatenate all mono-temporal modalities
                (AERIAL_RGBI, SPOT_RGBI, DEM_ELEV, AERIAL-RLT_PAN) into an 'image' key.
                Modalities will be resized to the maximum resolution before concatenation.
                If False, modalities remain separate. Defaults to False.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FLAIRHUB`.
        """
        super().__init__(FLAIRHUB, batch_size, num_workers, **kwargs)

        self.concatenate_modalities = concatenate_modalities

        # Create K.Normalize instances for each modality
        self.normalizers = {
            'AERIAL_RGBI': K.Normalize(
                mean=AERIAL_RGBI_MEAN, std=AERIAL_RGBI_STD, keepdim=True
            ),
            'SPOT_RGBI': K.Normalize(
                mean=SPOT_RGBI_MEAN, std=SPOT_RGBI_STD, keepdim=True
            ),
            'DEM_ELEV': K.Normalize(mean=DEM_ELEV_MEAN, std=DEM_ELEV_STD, keepdim=True),
            'AERIAL-RLT_PAN': K.Normalize(
                mean=AERIAL_RLT_PAN_MEAN, std=AERIAL_RLT_PAN_STD, keepdim=True
            ),
        }

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Apply normalization to specific modalities in the batch.

        Optionally concatenate all mono-temporal modalities into an 'image' key.

        Args:
            batch: A batch of data that needs to be normalized.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data with normalized modalities, and optionally concatenated
            mono-temporal modalities into an 'image' key.
        """
        # Map band names to batch keys using the dataset's modality_key_map
        key_map = self.dataset_class.modality_key_map

        # Apply modality-specific normalization
        for band_name, normalizer in self.normalizers.items():
            batch_key = key_map.get(band_name, band_name)
            if batch_key in batch:
                batch[batch_key] = normalizer(batch[batch_key])

        # Concatenate mono-temporal modalities if enabled
        if self.concatenate_modalities:
            mono_temporal_modalities = [
                'AERIAL_RGBI',
                'SPOT_RGBI',
                'DEM_ELEV',
                'AERIAL-RLT_PAN',
            ]
            present_keys = [
                key_map.get(mod, mod)
                for mod in mono_temporal_modalities
                if key_map.get(mod, mod) in batch
            ]
            if present_keys:
                max_resolution = max(batch[k].shape[-1] for k in present_keys)

                # Resize and concatenate modalities for each sample in batch
                concatenated_modalities = []
                for batch_key in present_keys:
                    tensor = batch[batch_key]  # [B, C, H, W]
                    if tensor.shape[-1] != max_resolution:
                        tensor = resize(
                            tensor,
                            size=(max_resolution, max_resolution),
                            interpolation='bilinear',
                            align_corners=False,
                        )
                    concatenated_modalities.append(tensor)

                batch['image'] = torch.cat(concatenated_modalities, dim=1)

        return batch


class FLAIRHUBToyDataModule(FLAIRHUBDataModule):
    """LightningDataModule implementation for the FLAIRHUBToy dataset.

    Uses official splits from the ``split_toy`` column.

    .. versionadded:: 0.10.0
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        concatenate_modalities: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FLAIRHUBToyDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            concatenate_modalities: If True, concatenate all mono-temporal modalities
                (aerial_rgbi, spot_rgbi, dem_elev, aerial_rlt_pan) into an 'image' key.
                Modalities will be resized to the maximum resolution before concatenation.
                If False, modalities remain separate. Defaults to False.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FLAIRHUBToy`.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            concatenate_modalities=concatenate_modalities,
            **kwargs,
        )
        # Override dataset class to use FLAIRHUBToy
        self.dataset_class = FLAIRHUBToy
