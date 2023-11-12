"""Northeastern China Crop Map Dataset."""

import glob
import os
import shutil
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive


class NCCM(RasterDataset):
    """The Northeastern China Crop Map Dataset.

    Link: https://www.nature.com/articles/s41597-021-00827-9

    This dataset produced annual 10-m crop maps of the
    major crops (maize, soybean, and rice)
    in Northeast China from 2017 to 2019, using hierarchial mapping strategies,
    random forest classifiers, interpolated and
    smoothed 10-day Sentinel-2 time series data and
    optimized features from spectral, temporal and
    textural characteristics of the land surface.
    The resultant maps have high overall accuracies (OA)
    based on ground truth data. The dataset contains information
    specific to three years: 2017, 2018, 2019.

    The dataset contains 4 classes:

    0. paddy rice
    1. maize
    2. soybean
    3. others

    Dataset format:

    1. Three .TIF files containing the labels
    2. JavaScript code to download images from the dataset.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41597-021-00827-9

    .. versionadded:: 0.6
    """

    filename_regex = r"CDL(?P<year>\d{4})_clip"
    filename_glob = "CDL*.*"
    zipfile_glob = "13090442.zip"

    date_format = "%Y"
    is_image = False
    url = "https://figshare.com/ndownloader/articles/13090442/versions/1"
    md5 = "eae952f1b346d7e649d027e8139a76f5"

    # years = [2017, 2018, 2019]

    cmap = {
        0: (0, 255, 0, 255),
        1: (255, 0, 0, 255),
        2: (255, 255, 0, 255),
        3: (128, 128, 128, 255),
    }

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        # years: list[int] = [2017, 2018, 2019],
        classes: list[int] = list(cmap.keys()),
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new dataset.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            classes: list of classes to include, the rest will be mapped to 0
                (defaults to all classes)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``years`` or ``classes`` are invalid
            FileNotFoundError: if no files are found in ``paths``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # assert all(
        #     year in self.years for year in years
        # ), f"NCCM data product only exists for the following years: {self.years}"

        assert (
            set(classes) <= self.cmap.keys()
        ), f"Only the following classes are valid: {list(self.cmap.keys())}."

        self.paths = paths
        # self.years = years
        self.classes = classes
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)

        curr_path = os.getcwd()
        curr_path += "/data"
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)
        else:
            contents = os.listdir(curr_path)
            for item in contents:
                item_path = os.path.join(curr_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        self._verify()
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

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
        sample["mask"] = self.ordinal_map[sample["mask"]]
        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        if self.files:
            return

        # Check if the zip file has already been downloaded
        assert isinstance(self.paths, str)
        pathname = os.path.join(self.paths, "**", self.zipfile_glob)
        if glob.glob(pathname, recursive=True):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `paths={self.paths!r}` and `download=False`, "
                "either specify different `paths` or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_root = "data"
        filename = "13090442.zip"
        download_url(
            self.url, download_root, filename, md5=self.md5 if self.checksum else None
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str)
        pathname = os.path.join(self.paths, "**", self.zipfile_glob)
        extract_archive(glob.glob(pathname, recursive=True)[0], self.paths)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NCCM.__getitem__`
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

        axs[0, 0].imshow(self.ordinal_cmap[mask], interpolation="none")
        axs[0, 0].axis("off")

        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(self.ordinal_cmap[pred], interpolation="none")
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
