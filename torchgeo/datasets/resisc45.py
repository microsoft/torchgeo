# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 dataset."""

import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import ImageFolder

from .geo import VisionDataset
from .utils import download_and_extract_archive


class RESISC45(VisionDataset, ImageFolder):  # type: ignore[misc]
    """RESISC45 dataset.

    The `RESISC45 <http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html>`_
    dataset is a dataset for remote sensing image scene classification.

    Dataset features:

    * 31,500 images with 0.2-30 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 45 scene classes, 700 images per class
    * images extracted from Google Earth from over 100 countries
    * images conditions with high variability (resolution, weather, illumination)

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. airplane
    1. airport
    2. baseball_diamond
    3. basketball_court
    4. beach
    5. bridge
    6. chaparral
    7. church
    8. circular_farmland
    9. cloud
    10. commercial_area
    11. dense_residential
    12. desert
    13. forest
    14. freeway
    15. golf_course
    16. ground_track_field
    17. harbor
    18. industrial_area
    19. intersection
    20. island
    21. lake
    22. meadow
    23. medium_residential
    24. mobile_home_park
    25. mountain
    26. overpass
    27. palace
    28. parking_lot
    29. railway
    30. railway_station
    31. rectangular_farmland
    32. river
    33. roundabout
    34. runway
    35. sea_ice
    36. ship
    37. snowberg
    38. sparse_residential
    39. stadium
    40. storage_tank
    41. tennis_court
    42. terrace
    43. thermal_power_station
    44. wetland

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/jproc.2017.2675998

    """

    url = "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
    md5 = "d824acb73957502b00efd559fc6cfbbb"
    filename = "NWPU-RESISC45.rar"
    directory = "NWPU-RESISC45"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new PatternNet dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
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

        # When transform & target_transform are None, ImageFolder.__getitem__[index]
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=os.path.join(root, self.directory),
            transform=None,
            target_transform=None,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load a single image and it's class label.

        Args:
            index: index to return

        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        array = np.array(img)
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        label = torch.tensor(label)  # type: ignore[attr-defined]
        return tensor, label

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        filepath = os.path.join(self.root, self.directory)
        if not os.path.exists(filepath):
            return False

        return True

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
