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
    r"""Append normalized difference index as channel to image tensor.

    Computes the following index:

    .. math::

       \text{NDI} = \frac{A - B}{A + B}

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

            sample["image"] = torch.cat([sample["image"], index], dim=self.dim)

        return sample


class AppendNBR(AppendNormalizedDifferenceIndex):
    r"""Normalized Burn Ratio (NBR).

    Computes the following index:

    .. math::

       \text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}

    If you use this index in your research, please cite the following paper:

    * https://www.sciencebase.gov/catalog/item/4f4e4b20e4b07f02db6abb36

    .. versionadded:: 0.2
    """

    def __init__(self, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__(index_a=index_nir, index_b=index_swir)


class AppendNDBI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Built-up Index (NDBI).

    Computes the following index:

    .. math::

       \text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}

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
    r"""Normalized Difference Snow Index (NDSI).

    Computes the following index:

    .. math::

       \text{NDSI} = \frac{\text{G} - \text{SWIR}}{\text{G} + \text{SWIR}}

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
    r"""Normalized Difference Vegetation Index (NDVI).

    Computes the following index:

    .. math::

       \text{NDVI} = \frac{\text{NIR} - \text{R}}{\text{NIR} + \text{R}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(79)90013-0
    """

    def __init__(self, index_nir: int, index_red: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_red: index of the Red band in the image
        """
        super().__init__(index_a=index_nir, index_b=index_red)


class AppendNDWI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Water Index (NDWI).

    Computes the following index:

    .. math::

       \text{NDWI} = \frac{\text{G} - \text{NIR}}{\text{G} + \text{NIR}}

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


class AppendSWI(AppendNormalizedDifferenceIndex):
    r"""Standardized Water-Level Index (SWI).

    Computes the following index:

    .. math::

       \text{SWI} = \frac{\text{VRE1} - \text{SWIR2}}{\text{VRE1} + \text{SWIR2}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.3390/w13121647

    .. versionadded:: 0.3
    """

    def __init__(self, index_vre1: int, index_swir2: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_vre1: index of the VRE1 band, e.g. B5 in Sentinel 2 imagery
            index_swir2: index of the SWIR2 band, e.g. B11 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_vre1, index_b=index_swir2)


class AppendGNDVI(AppendNormalizedDifferenceIndex):
    r"""Green Normalized Difference Vegetation Index (GNDVI).

    Computes the following index:

    .. math::

       \text{GNDVI} = \frac{\text{NIR} - \text{G}}{\text{NIR} + \text{G}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.2134/agronj2001.933583x

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_green: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_green: index of the Green band, e.g. B3 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_green)


class AppendBNDVI(AppendNormalizedDifferenceIndex):
    r"""Blue Normalized Difference Vegetation Index (BNDVI).

    Computes the following index:

    .. math::

       \text{BNDVI} = \frac{\text{NIR} - \text{B}}{\text{NIR} + \text{B}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_blue: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_blue: index of the Blue band, e.g. B2 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_blue)


class AppendNDRE(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Red Edge Vegetation Index (NDRE).

    Computes the following index:

    .. math::

       \text{NDRE} = \frac{\text{NIR} - \text{VRE1}}{\text{NIR} + \text{VRE1}}

    If you use this index in your research, please cite the following paper:

    * https://agris.fao.org/agris-search/search.do?recordID=US201300795763

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_vre1: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_vre1: index of the Red Edge band, B5 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_vre1)


class AppendTriBandNormalizedDifferenceIndex(Module):
    r"""Append normalized difference index involving 3 bands as channel to image tensor.

    Computes the following index:

    .. math::

       \text{NDI} = \frac{A - (B + C)}{A + (B + C)}

    .. versionadded:: 0.3
    """

    def __init__(self, index_a: int, index_b: int, index_c: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index of component 1
            index_c: difference band channel index of component 2
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b
        self.index_c = index_c

    def _compute_index(self, band_a: Tensor, band_b: Tensor, band_c: Tensor) -> Tensor:
        """Compute tri-band normalized difference index.

        Args:
            band_a: reference band tensor
            band_b: difference band tensor component 1
            band_c: difference band tensor component 2

        Returns:
            the index
        """
        return (band_a - (band_b + band_c)) / ((band_a + band_b + band_c) + _EPSILON)

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute and append tri-band normalized difference index to image.

        Args:
            sample: a sample or batch dict

        Returns:
            the transformed sample
        """
        if "image" in sample:
            index = self._compute_index(
                band_a=sample["image"][..., self.index_a, :, :],
                band_b=sample["image"][..., self.index_b, :, :],
                band_c=sample["image"][..., self.index_c, :, :],
            )
            index = index.unsqueeze(self.dim)

            sample["image"] = torch.cat([sample["image"], index], dim=self.dim)

        return sample


class AppendGRNDVI(AppendTriBandNormalizedDifferenceIndex):
    r"""Green-Red Normalized Difference Vegetation Index (GRNDVI).

    Computes the following index:

    .. math::

       \text{GRNDVI} =
           \frac{\text{NIR} - (\text{G} + \text{R})}{\text{NIR} + (\text{G} + \text{R})}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_green: int, index_red: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_green: index of the Green band, B3 in Sentinel 2 imagery
            index_red: index of the Red band, B4 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_green, index_c=index_red)


class AppendGBNDVI(AppendTriBandNormalizedDifferenceIndex):
    r"""Green-Blue Normalized Difference Vegetation Index (GBNDVI).

    Computes the following index:

    .. math::

       \text{GBNDVI} =
           \frac{\text{NIR} - (\text{G} + \text{B})}{\text{NIR} + (\text{G} + \text{B})}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_green: int, index_blue: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_green: index of the Green band, B3 in Sentinel 2 imagery
            index_blue: index of the Blue band, B2 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_green, index_c=index_blue)


class AppendRBNDVI(AppendTriBandNormalizedDifferenceIndex):
    r"""Red-Blue Normalized Difference Vegetation Index (RBNDVI).

    Computes the following index:

    .. math::

       \text{RBNDVI} =
           \frac{\text{NIR} - (\text{R} + \text{B})}{\text{NIR} + (\text{R} + \text{B})}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_red: int, index_blue: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_red: index of the Red band, B4 in Sentinel 2 imagery
            index_blue: index of the Blue band, B2 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_red, index_c=index_blue)
