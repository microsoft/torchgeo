# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo indices transforms."""

from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def ndvi(red: Tensor, nir: Tensor) -> Tensor:
    """Compute Normalized Different Vegetation Index (NDVI).

    Args:
        red: tensor containing red band
        nir: tensor containing nir band

    Returns:
        tensor containing computed NDVI values
    """
    return (nir - red) / (nir + red)


class NDVI(Module):  # type: ignore[misc,name-defined]
    """Normalized Difference Vegetation Index (NDVI).

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.05713
    """

    def __init__(self, index_red: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_nir = index_nir

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Create a band for NDVI and append to image channels.

        Args:
            sample: a single data sample

        Returns:
            a sample where the image has an additional channel representing NDVI
        """
        if "image" in sample:
            index = ndvi(
                red=sample["image"][self.index_red], nir=sample["image"][self.index_nir]
            )
            index = index.unsqueeze(0)
            sample["image"] = torch.cat([sample["image"], index], dim=0)  # type: ignore[attr-defined]  # noqa: E501

        return sample
