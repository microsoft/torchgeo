import glob
import os
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import BoundingBox, DatasetNotFoundError, download_url


class SouthAmericaSoybean(RasterDataset):
    """South America Soybean Dataset

This dataset produced annual 30-m soybean maps of South America from 2001 to 2021.

    Link: https://www.nature.com/articles/s41893-021-00729-z

    Dataset contains 2 classes:
    0: nodata
    1: soybean

    Dataset Format:

    * 21 .tif files 
     

    If you use this dataset in your research, please use the corresponding citation:

* https://doi.org/10.1038/s41893-021-00729-z 

.. versionadded:: 0.6

    """
    filename_glob = "SouthAmerica_Soybean_*.tif"
    filename_regex = (r"SouthAmerica_Soybean_(?P<year>\d{4})\.tif")


    date_format = "%Y"
    is_image = False

    url = "https://glad.umd.edu/projects/AnnualClassMapsV1/SouthAmerica_Soybean_"

    md5s = {
        2001: "2914b0af7590a0ca4dfa9ccefc99020f", 
        2002: "8a4a9dcea54b3ec7de07657b9f2c0893",
        2003: "cad5ed461ff4ab45c90177841aaecad2",
        2004: "f9882ca9c70e054e50172835cb75a8c3",
        2005: "89faae27f9b5afbd06935a465e5fe414",
        2006: "eabaa525414ecbff89301d3d5c706f0b",
        2007: "bb8549b6674163fe20ffd47ec4ce8903",
        2008: "96fc3f737ab3ce9bcd16cbf7761427e2",
        2009: "341387c1bb42a15140c80702e4cca02d",
        2010: "9264532d36ffa93493735a6e44caef0d",
        2011: "b73352ebea3d5658959e9044ec526143",
        2012: "9f3a71097c9836fcff18a13b9ba608b2",
        2013: "0263e19b3cae6fdaba4e3b450cef985e",
        2014: "824ff91c62a4ba9f4ccfd281729830e5",
        2015: "6beb96a61fe0e9ce8c06263e500dde8f",
        2016: "770c558f6ac40550d0e264da5e44b3e",
        2017: "4d0487ac1105d171e5f506f1766ea777",
        2018: "503c2d0a803c2a2629ebbbd9558a3013",
        2019: "441836493bbcd5e123cff579a58f5a4f",
        2020: "0709dec807f576c9707c8c7e183db31",
        2021: "edff3ada13a1a9910d1fe844d28ae4f",

    }


    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        years: list[int] = [2021],
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.
        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            years: list of years to use
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            AssertionError: if ``year`` is invalid
        """
        assert set(years) <= self.md5s.keys(), (
            "South America Soybean data only exists for the following years: "
            f"{list(self.md5s.keys())}."
        )

        self.years = years
        self.paths = paths
        self.download = download
        self.checksum = checksum
    
        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)
            
    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve mask and metadata indexed by query.
        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            sample of mask and metadata at that index
        Raises:
            IndexError: if query is not found in the index
        """
        sample = super().__getitem__(query)
        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset.
        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        if self.files:
            return
        assert isinstance(self.paths, str)

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
    def _download(self) -> None:
        """Download the dataset."""
        for i in range(21): 
            ext = ".tif"
            downloadUrl = self.url + str(i+2001) + ext
            download_url(downloadUrl,self.paths,md5 = self.md5s if self.checksum else None)
    

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample["mask"].squeeze()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 4, 4), squeeze=False
        )

        axs[0, 0].imshow(mask, interpolation="none")
        axs[0, 0].axis("off")

        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(pred, interpolation="none")
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig



