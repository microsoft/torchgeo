import os
from typing import Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity

from .geo import GeoDataset


class SEN12MS(GeoDataset):
    """The `SEN12MS <https://doi.org/10.14459/2019mp1474000>`_ dataset contains
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

    .. note:

       This dataset can be automatically downloaded using the following bash script:

       .. code-block: bash

          wget "ftp://m1474000:m1474000@dataserv.ub.tum.de/checksum.sha512"

          for season in 1158_spring 1868_summer 1970_fall 2017_winter
          do
              for source in lc s1 s2
              do
                  wget "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs${season}_${source}.tar.gz"
                  shasum -a 512 "ROIs${season}_${source}.tar.gz"
                  tar xvzf "ROIs${season}_${source}.tar.gz"
              done
          done

          for split in train test
          do
              wget "https://raw.githubusercontent.com/schmitt-muc/SEN12MS/master/splits/${split}_list.txt"
          done

       or manually downloaded from https://dataserv.ub.tum.de/s/m1474000
       and https://github.com/schmitt-muc/SEN12MS/tree/master/splits.
       This download will likely take several hours. The checksums.sha512
       file should be used to confirm the integrity of the downloads.
    """  # noqa: E501

    base_folder = "sen12ms"
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

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new SEN12MS dataset instance.

        Parameters:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["train", "test"]

        self.root = root
        self.transforms = transforms

        if not self._check_integrity():
            raise RuntimeError("Dataset not found.")

        with open(os.path.join(self.root, self.base_folder, split + "_list.txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and label at that index
        """
        filename = self.ids[index].rstrip()

        lc = self._load_raster(filename, "lc")
        s1 = self._load_raster(filename, "s1")
        s2 = self._load_raster(filename, "s2")

        sample = {
            "image": torch.cat([s1, s2]),  # type: ignore[attr-defined]
            "mask": lc,
        }

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

        Parameters:
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
                self.base_folder,
                "{0}_{1}".format(*parts),
                "{2}_{3}".format(*parts),
                "{0}_{1}_{2}_{3}_{4}".format(*parts),
            )
        ) as f:
            array = f.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if files exist, else False
        """
        # We could also check md5s, but it would take ~30 min to compute
        for filename in self.filenames:
            if not check_integrity(os.path.join(self.root, self.base_folder, filename)):
                return False
        return True
