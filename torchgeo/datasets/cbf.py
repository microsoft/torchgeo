# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Canadian Building Footprints dataset."""

import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from .geo import VectorDataset
from .utils import check_integrity, download_and_extract_archive


class CanadianBuildingFootprints(VectorDataset):
    """Canadian Building Footprints dataset.

    The `Canadian Building Footprints
    <https://github.com/Microsoft/CanadianBuildingFootprints>`__ dataset contains
    11,842,186 computer generated building footprints in all Canadian provinces and
    territories in GeoJSON format. This data is freely available for download and use.
    """

    # TODO: how does one cite this dataset?
    # https://github.com/microsoft/CanadianBuildingFootprints/issues/11

    url = "https://usbuildingdata.blob.core.windows.net/canadian-buildings-v2/"
    provinces_territories = [
        "Alberta",
        "BritishColumbia",
        "Manitoba",
        "NewBrunswick",
        "NewfoundlandAndLabrador",
        "NorthwestTerritories",
        "NovaScotia",
        "Nunavut",
        "Ontario",
        "PrinceEdwardIsland",
        "Quebec",
        "Saskatchewan",
        "YukonTerritory",
    ]
    md5s = [
        "8b4190424e57bb0902bd8ecb95a9235b",
        "fea05d6eb0006710729c675de63db839",
        "adf11187362624d68f9c69aaa693c46f",
        "44269d4ec89521735389ef9752ee8642",
        "65dd92b1f3f5f7222ae5edfad616d266",
        "346d70a682b95b451b81b47f660fd0e2",
        "bd57cb1a7822d72610215fca20a12602",
        "c1f29b73cdff9a6a9dd7d086b31ef2cf",
        "76ba4b7059c5717989ce34977cad42b2",
        "2e4a3fa47b3558503e61572c59ac5963",
        "9ff4417ae00354d39a0cf193c8df592c",
        "a51078d8e60082c7d3a3818240da6dd5",
        "c11f3bd914ecabd7cac2cb2871ec0261",
    ]

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: float = 0.00001,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` and data is not found, or
                ``checksum=True`` and checksums don't match
        """
        self.root = root
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        super().__init__(root, crs, res, transforms)

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for prov_terr, md5 in zip(self.provinces_territories, self.md5s):
            filepath = os.path.join(self.root, prov_terr + ".zip")
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for prov_terr, md5 in zip(self.provinces_territories, self.md5s):
            download_and_extract_archive(
                self.url + prov_terr + ".zip",
                self.root,
                md5=md5 if self.checksum else None,
            )

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`VectorDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
            Method now takes a sample dict, not a Tensor. Additionally, it is possible
            to show subplot titles and/or use a custom suptitle.
        """
        image = sample["mask"].squeeze(0)
        ncols = 1

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred = sample["prediction"].squeeze(0)
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 4))

        if showing_prediction:
            axs[0].imshow(image)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(image)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
