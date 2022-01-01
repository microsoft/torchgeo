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


class AppendNormalizedDifferenceIndex(Module):
    """Append normalized difference index as channel to image tensor.

    .. versionadded:: 0.2
    """

    def __init__(self, index_a: int, index_b: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b

    def _compute_index(self, band_a: Tensor, band_b: Tensor) -> Tensor:
        """Compute normalized difference index.

        Args:
            band_a: reference band tensor
            band_b: difference band tensor

        Returns:
            the index
        """
        return (band_a - band_b) / ((band_a + band_b) + _EPSILON)

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute and append normalized difference index to image.

        Args:
            sample: a sample or batch dict

        Returns:
            the transformed sample
        """
        if "image" in sample:
            index = self._compute_index(
                band_a=sample["image"][..., self.index_a, :, :],
                band_b=sample["image"][..., self.index_b, :, :],
            )
            index = index.unsqueeze(self.dim)

            sample["image"] = torch.cat(  # type: ignore[attr-defined]
                [sample["image"], index], dim=self.dim
            )

        return sample


class AppendNBR(AppendNormalizedDifferenceIndex):
    """Normalized Burn Ratio (NBR).

    If you use this index in your research, please cite the following paper:

    * https://www.sciencebase.gov/catalog/item/4f4e4b20e4b07f02db6abb36

    .. versionadded:: 0.2.0
    """

    def __init__(self, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__(index_a=index_nir, index_b=index_swir)


class AppendNDBI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Built-up Index (NDBI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431160304987
    """

    def __init__(self, index_swir: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__(index_a=index_swir, index_b=index_nir)


class AppendNDSI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Snow Index (NDSI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.1994.399618
    """

    def __init__(self, index_green: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__(index_a=index_green, index_b=index_swir)


class AppendNDVI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Vegetation Index (NDVI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(79)90013-0
    """

    def __init__(self, index_red: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__(index_a=index_red, index_b=index_nir)


class AppendNDWI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Water Index (NDWI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431169608948714
    """

    def __init__(self, index_green: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__(index_a=index_green, index_b=index_nir)
