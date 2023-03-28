# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L8Biome dataset."""

import abc
import glob
import hashlib
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from rasterio.crs import CRS
from torch import Tensor
from torch.utils.data import Dataset
from .utils import check_integrity

from .geo import NonGeoDataset, RasterDataset
from .utils import BoundingBox, download_url, extract_archive, working_dir


class L8Biome(RasterDataset):
    r"""L8 Biome datasets.

    The L8 Biome dataset `<https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data>`__ (Landsat 8 Cloud Cover Assessment Validation Data) 
    is a cloud validation dataset of Pre-Collection Landsat 8 Operational Land Imager (OLI) Thermal Infrared Sensor (TIRS) terrain-corrected (Level-1T) scenes.
    
    Dataset features:

    * images evenly divided between eight unique biomes
    * three cloud cover categories

    Dataset format:

    * Each cloud mask is in ENVI binary format. Includes all bands from the original Landsat Level-1 data product (GeoTIFF), and its associated Level-1 metadata (MTL.txt file)

    If you use this dataset in your research, please cite the following:

    * doi:10.5066/F7251GDH
    * 10.1016/j.rse.2017.03.026

    .. versionadded:: 0.5
    """

    url = "https://huggingface.co/datasets/torchgeo/l8biome" # redistributed from https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data
    filenames_to_md5 = {
        "barren": "0ff8e32511e3e62fce9a08bed88bebab",
        "forest": "8adf6b7b33b65e16c0f692c227cd38de",
        "grass_crops": "fd38f2b6ff00cce8cd14ea6db82f10f4",
        "shrubland": "2de5cdeea89390f5908663dcab5ca869",
        "snow_ice": "e431f0b8e8431f8b04a440bf5083a91c",
        "urban": "b6d39858117940930bf6fa4d63488374",
        "water": "8ce42b9f7f58f4d9f8fd6801aa894421",
        "wetlands": "ccd193eb7509e262cbe9588ecb8eddb4"
    }


    def __init__(
        self, 
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
        cache: bool = True,
    ) -> None:
        """Initialize a new L8Biome dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        self._verify()

        super().__init__(root, crs = crs, res=res, transforms = transforms, cache = cache)
    
    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
        """

        # Check if the all files of dataset exist and the MD5 matches
        for filename in self.filenames_to_md5.keys():
            gz_path = os.path.join(self.root, f'{filename}.tar.gz')
            if os.path.exists(gz_path):
                md5 = self.filenames_to_md5[filename]
                integrity = check_integrity(gz_path, md5)
                if not integrity:
                    raise RuntimeError(f"Dataset not found or corrupted.")
            else:
                raise RuntimeError(f"Dataset not found or corrupted.")

    def __getitem__(self, query: Any) -> Dict[str, Any]:
        """Retrieve image, mask and metadata indexed by index.

        Args:
            query: coordinates or an index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

    def _verify_data(self) -> bool:
        """Verify if the images and masks are present."""

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy().astype("uint8").squeeze(), 0, 3)
        mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=4, cmap=self._lc_cmap, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=4, cmap=self._lc_cmap, interpolation="none"
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

