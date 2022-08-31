# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SEN12MS dataset."""

import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, percentile_normalization


class SEN12MS(NonGeoDataset):
    """SEN12MS dataset.

    The `SEN12MS <https://doi.org/10.14459/2019mp1474000>`__ dataset contains
    180,662 patch triplets of corresponding Sentinel-1 dual-pol SAR data,
    Sentinel-2 multi-spectral images, and MODIS-derived land cover maps.
    The patches are distributed across the land masses of the Earth and
    spread over all four meteorological seasons. This is reflected by the
    dataset structure. All patches are provided in the form of 16-bit GeoTiffs
    containing the following specific information:

    * Sentinel-1 SAR: 2 channels corresponding to sigma nought backscatter
      values in dB scale for VV and VH polarization.
    * Sentinel-2 Multi-Spectral: 13 channels corresponding to the 13 spectral bands
      (B1, B2, B3, B4, B5, B6, B7, B8, B8a, B9, B10, B11, B12).
    * MODIS Land Cover: 4 channels corresponding to IGBP, LCCS Land Cover,
      LCCS Land Use, and LCCS Surface Hydrology layers.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019

    .. note::

       This dataset can be automatically downloaded using the following bash script:

       .. code-block:: bash

          for season in 1158_spring 1868_summer 1970_fall 2017_winter
          do
              for source in lc s1 s2
              do
                  wget "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs${season}_${source}.tar.gz"
                  tar xvzf "ROIs${season}_${source}.tar.gz"
              done
          done

          for split in train test
          do
              wget "https://raw.githubusercontent.com/schmitt-muc/SEN12MS/master/splits/${split}_list.txt"
          done

       or manually downloaded from https://dataserv.ub.tum.de/s/m1474000
       and https://github.com/schmitt-muc/SEN12MS/tree/master/splits.
       This download will likely take several hours.
    """  # noqa: E501

    BAND_SETS: Dict[str, Tuple[str, ...]] = {
        "all": (
            "VV",
            "VH",
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ),
        "s1": ("VV", "VH"),
        "s2-all": (
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ),
        "s2-reduced": ("B02", "B03", "B04", "B08", "B10", "B11"),
    }

    band_names = (
        "VV",
        "VH",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    RGB_BANDS = ["B04", "B03", "B02"]

    filenames = [
        "ROIs1158_spring_lc.tar.gz",
        "ROIs1158_spring_s1.tar.gz",
        "ROIs1158_spring_s2.tar.gz",
        "ROIs1868_summer_lc.tar.gz",
        "ROIs1868_summer_s1.tar.gz",
        "ROIs1868_summer_s2.tar.gz",
        "ROIs1970_fall_lc.tar.gz",
        "ROIs1970_fall_s1.tar.gz",
        "ROIs1970_fall_s2.tar.gz",
        "ROIs2017_winter_lc.tar.gz",
        "ROIs2017_winter_s1.tar.gz",
        "ROIs2017_winter_s2.tar.gz",
        "train_list.txt",
        "test_list.txt",
    ]
    light_filenames = [
        "ROIs1158_spring",
        "ROIs1868_summer",
        "ROIs1970_fall",
        "ROIs2017_winter",
        "train_list.txt",
        "test_list.txt",
    ]
    md5s = [
        "6e2e8fa8b8cba77ddab49fd20ff5c37b",
        "fba019bb27a08c1db96b31f718c34d79",
        "d58af2c15a16f376eb3308dc9b685af2",
        "2c5bd80244440b6f9d54957c6b1f23d4",
        "01044b7f58d33570c6b57fec28a3d449",
        "4dbaf72ecb704a4794036fe691427ff3",
        "9b126a68b0e3af260071b3139cb57cee",
        "19132e0aab9d4d6862fd42e8e6760847",
        "b8f117818878da86b5f5e06400eb1866",
        "0fa0420ef7bcfe4387c7e6fe226dc728",
        "bb8cbfc16b95a4f054a3d5380e0130ed",
        "3807545661288dcca312c9c538537b63",
        "0a68d4e1eb24f128fccdb930000b2546",
        "c7faad064001e646445c4c634169484d",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SEN12MS dataset instance.

        The ``bands`` argument allows for the subsetting of bands returned by the
        dataset. Integers in ``bands`` index into a stack of Sentinel 1 and Sentinel 2
        imagery. Indices 0 and 1 correspond to the Sentinel 1 imagery where indices 2
        through 14 correspond to the Sentinel 2 imagery.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            bands: a sequence of band indices to use where the indices correspond to the
                array index of combined Sentinel 1 and Sentinel 2
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if data is not found in ``root``, or checksums don't match
        """
        assert split in ["train", "test"]

        self._validate_bands(bands)
        self.band_indices = torch.tensor(
            [self.band_names.index(b) for b in bands]
        ).long()
        self.bands = bands

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if checksum:
            if not self._check_integrity():
                raise RuntimeError("Dataset not found or corrupted.")
        else:
            if not self._check_integrity_light():
                raise RuntimeError("Dataset not found or corrupted.")

        with open(os.path.join(self.root, split + "_list.txt")) as f:
            self.ids = [line.rstrip() for line in f.readlines()]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        filename = self.ids[index]

        lc = self._load_raster(filename, "lc")
        s1 = self._load_raster(filename, "s1")
        s2 = self._load_raster(filename, "s2")

        image = torch.cat(tensors=[s1, s2], dim=0)
        image = torch.index_select(image, dim=0, index=self.band_indices)

        sample: Dict[str, Tensor] = {"image": image, "mask": lc}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_raster(self, filename: str, source: str) -> Tensor:
        """Load a single raster image or target.

        Args:
            filename: name of the file to load
            source: one of "lc", "s1", or "s2"

        Returns:
            the raster image or target
        """
        parts = filename.split("_")
        parts[2] = source

        with rasterio.open(
            os.path.join(
                self.root,
                "{}_{}".format(*parts),
                "{2}_{3}".format(*parts),
                "{}_{}_{}_{}_{}".format(*parts),
            )
        ) as f:
            array = f.read()
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array)
            return tensor

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            AssertionError: if ``bands`` is not a sequence
            ValueError: if an invalid band name is provided
        """
        assert isinstance(bands, tuple), "'bands' must be a sequence"
        for band in bands:
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def _check_integrity_light(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for filename in self.light_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

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
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image, mask = sample["image"][rgb_indices].numpy(), sample["mask"][0]
        image = percentile_normalization(image)
        ncols = 2

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"][0]
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 5))

        axs[0].imshow(np.transpose(image, (1, 2, 0)))
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if showing_predictions:
            axs[2].imshow(prediction)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")
            if showing_predictions:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
