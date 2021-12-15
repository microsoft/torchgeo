# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo indices transforms.

For more information about indices see the following references:
- https://www.indexdatabase.de/db/i.php
- https://github.com/davemlz/awesome-spectral-indices
"""

from typing import Dict

import torch
from torch import Tensor
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


_EPSILON = 1e-10


def _compute_index(band_a: Tensor, band_b: Tensor) -> Tensor:
    """Compute common difference-based indices
    
    Args:
        band_a: tensor containing the reference band
        band_b: tensor containing the difference band

    Returns:
        tensor which contains the computer index values
    """
    return (band_a - band_b) / ((band_a + band_b) + _EPSILON)


class AppendIndex(Module):
    """
    Normalized Difference Built-up Index (NDBI): (swir - nir) / (swir + nir)
    Normalized Difference Snow Index (NDSI): (green - swir) / (green + swir)
    Normalized Difference Vegetation Index (NDVI): (red - nir) / (red + nir)
    Normalized Difference Water Index: (green - nir) / (green + nir)

    """
    _INDICES = {'ndbi', 'ndsi', 'ndvi', 'ndwi'}

    def __init__(self, index: str, band_a: int, band_b: int) -> None:
        super().__init__()
        self.dim = -3
        self.band_a = band_a
        self.band_b = band_b


    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Create a band for the computed index and append to image channels.

        Args:
            sample: a single data sample

        Returns:
            a sample where the image has an additional channel representing the computed index
        """
        if "image" in sample:
            index = _compute_index(
                band_a=sample["image"][:, self.band_a],
                band_b=sample["image"][:, self.band_b],
            )
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat(  # type: ignore[attr-defined]
                [sample["image"], index], dim=self.dim
            )

        return sample