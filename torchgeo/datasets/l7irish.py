# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Landsat 7 Cloud Cover Assessment Validation Data"""
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

from .geo import NonGeoDataset, RasterDataset
from .landsat import Landsat7
from .utils import BoundingBox, download_url, extract_archive, working_dir


class L7Irish(RasterDataset):    
    """L7 Irish dataset.

    The `L7 Irish <https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data>`__ dataset is based on Landsat 7

    Dataset format:
    * # images: 
    * image size: 
    * # spectral bands: 
    * Level-1G
    
    * labels are values [0, 64, 128, 192, 255]

     Dataset classes:
     * Fill
     * Cloud Shadow
     * Clear
     * Thin Cloud
     * Cloud

     If you use this dataset in your research, please cite the following papers:

     * https://doi.org/10.5066/F7XD0ZWC
     * https://doi.org/10.1109/TGRS.2011.2164087
     """
    
    # naming... [need to change]
    filename_regex = r"""
        ^L
        (?P<sensor>[COTEM])
        (?P<satellite>\d{2})
        _(?P<processing_correction_level>[A-Z0-9]{4})
        _(?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        _(?P<date>\d{8})
        _(?P<processing_date>\d{8})
        _(?P<collection_number>\d{2})
        _(?P<collection_category>[A-Z0-9]{2})
        _(?P<band>[A-Z0-9_]+)
        \.
    """

    tarfile_glob = "*.tar.gz"

    url = "https://huggingface.co/datasets/torchgeo/l7irish/resolve/main/{}.tar.gz"
    
    md5s = [("austral", "dbb6b5628f50861b9b89f548d25a925f"),
           ("boreal", "cecc72de09aacde4c4f8d7f0cf0d3f6f"),
           ("mid_latitude_north", "0f8382ca6554fb7cf9aff42226a14f9d"),
           ("mid_latitude_south", "b17cf6d023f752c533211fdb742f296b"),
           ("polar_north", "73923dcaf1b9b79bad82de1aa0740d1e"),
           ("polar_south", "3bc9f4c6f8955b10b4d55d23e0ab2da7"),
           ("subtropical_north", "f8f039970256902e6e9ebd6747589294"),
           ("subtropical_south", "8346d73a983396c5d41b577c3a94bc26"),
           ("tropical", "abe19b22b5d031e6b609cc7207706c3d")
           ]
        
    classes = ["Fill", "Cloud Shadow", "Clear", "Thin Cloud", "Cloud"]
    
    cmap = {
        0: (0, 0, 0),
        64: (64, 64, 64),
        128: (128, 128, 128),
        192: (192, 192, 192),
        255: (255, 255, 255),
    }

    def __init__(
        self, root: str = "data", download: bool = False, checksum: bool = False
    ) -> None:
        
        """Initialize a new Landsat 7 Cloud Cover Assessment Validation dataset instance.

            Args:
                root: root directory where dataset can be found
                transforms: a function/transform that takes input sample and its target as
                    entry and returns a transformed version
                cache: if True, cache file handle to speed up repeated sampling
                download: if True, download dataset and store it in the root directory
                checksum: if True, check the MD5 of the downloaded files (may be slow)

            Raises:
                RuntimeError: if ``download=False`` and data is not found, or checksums
                    don't match
            """
        self.root = root
        self.download = download
        self.checksum = checksum
    
        for i in list(self.cmap.keys()):
            self.cmap[np.ceil(i/64).astype(int)] = self.cmap.pop(i)

        lc_colors = np.zeros((max(self.cmap.keys()) + 1, 3))
        lc_colors[list(self.cmap.keys())] = list(self.cmap.values())
        lc_colors = lc_colors / 255
        _lc_cmap = ListedColormap(lc_colors)

        self._verify()
        super().__init__(root, download, checksum)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            if not fname.endswith(".zip"):
                return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.tarfile_glob)
        if glob.glob(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )
        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for biome, md5 in self.md5s:
            download_url(
                self.url.format(biome), self.root, md5=md5 if self.checksum else None
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.tarfile_glob)
        for tarfile in glob.iglob(pathname):
            extract_archive(tarfile)
    
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        img_filepaths = cast(List[str], [hit.object for hit in hits])
        mask_filepaths = [path.replace("","") for path in img_filepaths]
        # mask_filepaths = [path.replace("images", "masks") for path in img_filepaths]

        if not img_filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        img = self._merge_files(img_filepaths, query, self.band_indexes)
        mask = self._merge_files(mask_filepaths, query, self.band_indexes)
        # Mask needs to be converted from 0, 64, 128, 192, 255 to 0, 1, 2, 3, 4
        
        sample = {
            "crs": self.crs,
            "bbox": query,
            "image": img.float(),
            "mask": mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

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
        axs[1].imshow(mask, vmin=0, vmax=4,
                        cmap=self._lc_cmap, interpolation="none")
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
