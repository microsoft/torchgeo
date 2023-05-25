# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation."""

import os
import random
from typing import Callable, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive


class SSL4EO(NonGeoDataset):
    """Base class for all SSL4EO datasets.

    Self-Supervised Learning for Earth Observation (SSL4EO) is a collection of
    large-scale multimodal multitemporal datasets for unsupervised/self-supervised
    pre-training in Earth observation.

    .. versionadded:: 0.5
    """


class SSL4EOL(NonGeoDataset):
    """SSL4EO-L dataset.

    Landsat version of SSL4EO.

    The dataset consists of a parallel corpus (same locations for all splits, same dates
    for SR/TOA) for the following sensors:

    .. list-table::
       :widths: 10 10 10 10 10
       :header-rows: 1

       * - Satellites
         - Sensors
         - Level
         - # Bands
         - Link
       * - Landsat 4--5
         - TM
         - TOA
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA>`__
       * - Landsat 4--7
         - TM
         - SR
         - 6
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2>`__
       * - Landsat 7
         - ETM+
         - TOA
         - 9
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA>`__
       * - Landsat 8--9
         - OLI+TIRS
         - TOA
         - 11
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA>`__
       * - Landsat 8--9
         - OLI
         - SR
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2>`__

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * Single multispectral GeoTIFF file

    .. versionadded:: 0.5
    """  # noqa: E501

    class _Metadata(TypedDict):
        num_bands: int
        rgb_bands: list[int]

    metadata: dict[str, _Metadata] = {
        "tm_toa": {"num_bands": 7, "rgb_bands": [2, 1, 0]},
        "etm_sr": {"num_bands": 6, "rgb_bands": [2, 1, 0]},
        "etm_toa": {"num_bands": 9, "rgb_bands": [2, 1, 0]},
        "oli_tirs_toa": {"num_bands": 11, "rgb_bands": [3, 2, 1]},
        "oli_sr": {"num_bands": 7, "rgb_bands": [3, 2, 1]},
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "oli_sr",
        seasons: int = 1,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new SSL4EOL instance.

        Args:
            root: root directory where dataset can be found
            split: one of ['tm_toa', 'etm_sr', 'etm_toa', 'oli_tirs_toa', 'oli_sr']
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            AssertionError: if ``split`` argument is invalid
        """
        assert split in self.metadata
        assert seasons in range(1, 5)

        self.root = root
        self.split = split
        self.seasons = seasons
        self.transforms = transforms

        self.scenes = sorted(os.listdir(root))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        root = os.path.join(self.root, self.scenes[index])
        subdirs = os.listdir(root)
        subdirs = random.sample(subdirs, self.seasons)

        images = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            filename = os.path.join(directory, "all_bands.tif")
            with rasterio.open(filename) as f:
                image = f.read()
                images.append(torch.from_numpy(image.astype(np.float32)))

        sample = {"image": torch.cat(images)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.scenes)

    def plot(
        self,
        sample: dict[str, Tensor],
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
        fig, axes = plt.subplots(
            ncols=self.seasons, squeeze=False, figsize=(4 * self.seasons, 4)
        )
        num_bands = self.metadata[self.split]["num_bands"]
        rgb_bands = self.metadata[self.split]["rgb_bands"]

        for i in range(self.seasons):
            image = sample["image"][i * num_bands : (i + 1) * num_bands].byte()

            image = image[rgb_bands].permute(1, 2, 0)
            axes[0, i].imshow(image)
            axes[0, i].axis("off")

            if show_titles:
                axes[0, i].set_title(f"Split {self.split}, Season {i + 1}")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class SSL4EOS12(NonGeoDataset):
    """SSL4EO-S12 dataset.

    `Sentinel-1/2 <https://github.com/zhu-xlab/SSL4EO-S12>`_ version of SSL4EO.

    The dataset consists of unlabeled patch triplets (Sentinel-1 dual-pol SAR,
    Sentinel-2 top-of-atmosphere multispectral, Sentinel-2 surface reflectance
    multispectral) from 251079 locations across the globe, each patch covering
    2640mx2640m and including four seasonal time stamps.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2211.07044

    .. note::

       This dataset can be downloaded using:

       .. code-block:: console

          $ export RSYNC_PASSWORD=m1660427.001
          $ rsync -av rsync://m1660427.001@dataserv.ub.tum.de/m1660427.001/ .

       The dataset is about 1.5 TB when compressed and 3.7 TB when uncompressed, and
       takes roughly 36 hrs to download, 1 hr to checksum, and 12 hrs to extract.

    .. versionadded:: 0.5
    """

    size = 264

    class _Metadata(TypedDict):
        filename: str
        md5: str
        bands: list[str]

    metadata: dict[str, _Metadata] = {
        "s1": {
            "filename": "s1.tar.gz",
            "md5": "51ee23b33eb0a2f920bda25225072f3a",
            "bands": ["VV", "VH"],
        },
        "s2c": {
            "filename": "s2_l1c.tar.gz",
            "md5": "b4f8b03c365e4a85780ded600b7497ab",
            "bands": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B10",
                "B11",
                "B12",
            ],
        },
        "s2a": {
            "filename": "s2_l2a.tar.gz",
            "md5": "85496cd9d6742aee03b6a1c99cee0ac1",
            "bands": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B11",
                "B12",
            ],
        },
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "s2c",
        seasons: int = 1,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EOS12 instance.

        Args:
            root: root directory where dataset can be found
            split: one of "s1" (Sentinel-1 dual-pol SAR), "s2c" (Sentinel-2 Level-1C
                top-of-atmosphere reflectance), and "s2a" (Sentinel-2 Level-2a surface
                reflectance)
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if dataset is missing or checksum fails
        """
        assert split in self.metadata
        assert seasons in range(1, 5)

        self.root = root
        self.split = split
        self.seasons = seasons
        self.transforms = transforms
        self.checksum = checksum

        self.bands = self.metadata[self.split]["bands"]

        self._verify()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        root = os.path.join(self.root, self.split, f"{index:07}")
        subdirs = os.listdir(root)
        subdirs = random.sample(subdirs, self.seasons)

        images = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            for band in self.bands:
                filename = os.path.join(directory, f"{band}.tif")
                with rasterio.open(filename) as f:
                    image = f.read(out_shape=(1, self.size, self.size))
                    images.append(torch.from_numpy(image.astype(np.float32)))

        sample = {"image": torch.cat(images)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return 251079

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        directory_path = os.path.join(self.root, self.split)
        if os.path.exists(directory_path):
            return

        # Check if the zip files have already been downloaded
        filename = self.metadata[self.split]["filename"]
        zip_path = os.path.join(self.root, filename)
        md5 = self.metadata[self.split]["md5"] if self.checksum else None
        integrity = check_integrity(zip_path, md5)
        if integrity:
            self._extract()
        else:
            raise RuntimeError(f"Dataset not found in `root={self.root}`")

    def _extract(self) -> None:
        """Extract the dataset."""
        filename = self.metadata[self.split]["filename"]
        extract_archive(os.path.join(self.root, filename))

    def plot(
        self,
        sample: dict[str, Tensor],
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
        nrows = 2 if self.split == "s1" else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=self.seasons,
            squeeze=False,
            figsize=(4 * self.seasons, 4 * nrows),
        )

        for i in range(self.seasons):
            image = sample["image"][i * len(self.bands) : (i + 1) * len(self.bands)]

            if self.split == "s1":
                axes[0, i].imshow(image[0])
                axes[1, i].imshow(image[1])
            else:
                image = image[[3, 2, 1]].permute(1, 2, 0)
                image = torch.clamp(image / 3000, min=0, max=1)
                axes[0, i].imshow(image)

            axes[0, i].axis("off")

            if show_titles:
                axes[0, i].set_title(f"Split {self.split}, Season {i + 1}")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
