# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""European Digital Elevation Model (EU-DEM) dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import check_integrity, extract_archive


class EUDEM(RasterDataset):
    """European Digital Elevation Model (EU-DEM) Dataset.

    The `EU-DEM
    <https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1?tab=mapview>`__
    dataset is a Digital Elevation Model of reference for the entire European region.
    The dataset can be downloaded from this `website
    <https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1?tab=mapview>`_
    after making an account. A dataset factsheet is available
    `here <https://land.copernicus.eu/user-corner/publications/eu-dem-flyer/view>`__.

    Dataset features:

    * DEMs at 25 m per pixel spatial resolution (~40,000x40,0000 px)
    * vertical accuracy of +/- 7 m RMSE
    * data fused from `ASTER GDEM
      <https://lpdaac.usgs.gov/news/nasa-and-meti-release-aster-global-dem-version-3/>`_,
      `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ and Russian topomaps

    Dataset format:

    * DEMs are single-channel tif files

    If you use this dataset in your research, please give credit to:

    * `Copernicus <https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1>`_

    .. versionadded:: 0.3
    """

    is_image = False
    filename_glob = "eu_dem_v11_*.TIF"
    zipfile_glob = "eu_dem_v11_*[A-Z0-9].zip"
    filename_regex = "(?P<name>[eudem_v11]{10})_(?P<id>[A-Z0-9]{6})"

    md5s = {
        "eu_dem_v11_E00N20.zip": "96edc7e11bc299b994e848050d6be591",
        "eu_dem_v11_E10N00.zip": "e14be147ac83eddf655f4833d55c1571",
        "eu_dem_v11_E10N10.zip": "2eb5187e4d827245b33768404529c709",
        "eu_dem_v11_E10N20.zip": "1afc162eb131841aed0d00b692b870a8",
        "eu_dem_v11_E20N10.zip": "77b040791b9fb7de271b3f47130b4e0c",
        "eu_dem_v11_E20N20.zip": "89b965abdcb1dbd479c61117f55230c8",
        "eu_dem_v11_E20N30.zip": "f5cb1b05813ae8ffc9e70f0ad56cc372",
        "eu_dem_v11_E20N40.zip": "81be551ff646802d7d820385de7476e9",
        "eu_dem_v11_E20N50.zip": "bbc351713ea3eb7e9eb6794acb9e4bc8",
        "eu_dem_v11_E30N10.zip": "68fb95aac33a025c4f35571f32f237ff",
        "eu_dem_v11_E30N20.zip": "da8ad029f9cc1ec9234ea3e7629fe18d",
        "eu_dem_v11_E30N30.zip": "de27c78d0176e45aec5c9e462a95749c",
        "eu_dem_v11_E30N40.zip": "4c00e58b624adfc4a5748c922e77ee40",
        "eu_dem_v11_E30N50.zip": "4a21a88f4d2047b8995d1101df0b3a77",
        "eu_dem_v11_E40N10.zip": "32fdf4572581eddc305a21c5d2f4bc81",
        "eu_dem_v11_E40N20.zip": "71b027f29258493dd751cfd63f08578f",
        "eu_dem_v11_E40N30.zip": "c6c21289882c1f74fc4649d255302c64",
        "eu_dem_v11_E40N40.zip": "9f26e6e47f4160ef8ea5200e8cf90a45",
        "eu_dem_v11_E40N50.zip": "a8c3c1c026cdd1537b8a3822c15834d9",
        "eu_dem_v11_E50N10.zip": "9584273c7708b8e935f2bac3e30c19c6",
        "eu_dem_v11_E50N20.zip": "8efdea43e7b6819861935d5a768a55f2",
        "eu_dem_v11_E50N30.zip": "e39e58df1c13ac35eb0b29fb651f313c",
        "eu_dem_v11_E50N40.zip": "d84395ab52ad254d930db17398fffc50",
        "eu_dem_v11_E50N50.zip": "6abe852f4a20962db0e355ffc0d695a4",
        "eu_dem_v11_E60N10.zip": "b6a3b8a39a4efc01c7e2cd8418672559",
        "eu_dem_v11_E60N20.zip": "71dc3c55ab5c90628ce2149dbd60f090",
        "eu_dem_v11_E70N20.zip": "5342465ad60cf7d28a586c9585179c35",
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found, here the collection of
                individual zip files for each tile should be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.root = root
        self.checksum = checksum

        self._verify()

        super().__init__(root, crs, res, transforms, cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
        """
        # Check if the extracted file already exists
        pathname = os.path.join(self.root, self.filename_glob)
        if glob.glob(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
        if glob.glob(pathname):
            for zipfile in glob.iglob(pathname):
                filename = os.path.basename(zipfile)
                if self.checksum and not check_integrity(zipfile, self.md5s[filename]):
                    raise RuntimeError("Dataset found, but corrupted.")
                extract_archive(zipfile)
            return

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` "
            "either specify a different `root` directory or make sure you "
            "have manually downloaded the dataset as suggested in the documentation."
        )

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
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

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
