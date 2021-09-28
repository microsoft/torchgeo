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
from torch.nn import Module  # type: ignore[attr-defined]

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


_EPSILON = 1e-10


def ndbi(swir: Tensor, nir: Tensor) -> Tensor:
    """Compute Normalized Different Built-up Index (NDBI).

    Args:
        swir: tensor containing swir band
        nir: tensor containing nir band

    Returns:
        tensor containing computed NDBI values
    """
    return (swir - nir) / ((swir + nir) + _EPSILON)


def ndsi(green: Tensor, swir: Tensor) -> Tensor:
    """Compute Normalized Different Snow Index (NDSI).

    Args:
        green: tensor containing green band
        swir: tensor containing swir band

    Returns:
        tensor containing computed NDSI values
    """
    return (green - swir) / ((green + swir) + _EPSILON)


def ndvi(red: Tensor, nir: Tensor) -> Tensor:
    """Compute Normalized Different Vegetation Index (NDVI).

    Args:
        red: tensor containing red band
        nir: tensor containing nir band

    Returns:
        tensor containing computed NDVI values
    """
    return (nir - red) / ((nir + red) + _EPSILON)


def ndwi(green: Tensor, nir: Tensor) -> Tensor:
    """Compute Normalized Different Water Index (NDWI).

    Args:
        green: tensor containing green band
        nir: tensor containing nir band

    Returns:
        tensor containing computed NDWI values
    """
    return (green - nir) / ((green + nir) + _EPSILON)


class AppendNDBI(Module):  # type: ignore[misc,name-defined]
    """Normalized Difference Built-up Index (NDBI).

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1080/01431160304987
    """

    def __init__(self, index_swir: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.dim = -3
        self.index_nir = index_nir
        self.index_swir = index_swir

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Create a band for NDBI and append to image channels.

        Args:
            sample: a single data sample

        Returns:
            a sample where the image has an additional channel representing NDBI
        """
        if "image" in sample:
            index = ndbi(
                swir=sample["image"][:, self.index_swir],
                nir=sample["image"][:, self.index_nir],
            )
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat(  # type: ignore[attr-defined]
                [sample["image"], index], dim=self.dim
            )

        return sample


class AppendNDSI(Module):  # type: ignore[misc,name-defined]
    """Normalized Difference Snow Index (NDSI).

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.1994.399618
    """

    def __init__(self, index_green: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.dim = -3
        self.index_green = index_green
        self.index_swir = index_swir

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Create a band for NDSI and append to image channels.

        Args:
            sample: a single data sample

        Returns:
            a sample where the image has an additional channel representing NDSI
        """
        if "image" in sample:
            index = ndsi(
                green=sample["image"][:, self.index_green],
                swir=sample["image"][:, self.index_swir],
            )
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat(  # type: ignore[attr-defined]
                [sample["image"], index], dim=self.dim
            )

        return sample


class AppendNDVI(Module):  # type: ignore[misc,name-defined]
    """Normalized Difference Vegetation Index (NDVI).

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(79)90013-0
    """

    def __init__(self, index_red: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.dim = -3
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
                red=sample["image"][:, self.index_red],
                nir=sample["image"][:, self.index_nir],
            )
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat(  # type: ignore[attr-defined]
                [sample["image"], index], dim=self.dim
            )

        return sample


class AppendNDWI(Module):  # type: ignore[misc,name-defined]
    """Normalized Difference Water Index (NDWI).

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1080/01431169608948714
    """

    def __init__(self, index_green: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.dim = -3
        self.index_green = index_green
        self.index_nir = index_nir

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Create a band for NDWI and append to image channels.

        Args:
            sample: a single data sample

        Returns:
            a sample where the image has an additional channel representing NDWI
        """
        if "image" in sample:
            index = ndwi(
                green=sample["image"][:, self.index_green],
                nir=sample["image"][:, self.index_nir],
            )
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat(  # type: ignore[attr-defined]
                [sample["image"], index], dim=self.dim
            )

        return sample
