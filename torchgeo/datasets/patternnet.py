# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PatternNet dataset."""

import os
from typing import Any, Callable, Dict, Optional

from torch import Tensor

from .geo import VisionClassificationDataset
from .utils import download_and_extract_archive


class PatternNet(VisionClassificationDataset):
    """PatternNet dataset.

    The `PatternNet <https://sites.google.com/view/zhouwx/dataset>`_
    dataset is a dataset for remote sensing scene classification and image retrieval.

    Dataset features:

    * 30,400 images with 6-50 cm per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 38 scene classes, 800 images per class

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. airplane
    1. baseball_field
    2. basketball_court
    3. beach
    4. bridge
    5. cemetery
    6. chaparral
    7. christmas_tree_farm
    8. closed_road
    9. coastal_mansion
    10. crosswalk
    11. dense_residential
    12. ferry_terminal
    13. football_field
    14. forest
    15. freeway
    16. golf_course
    17. harbor
    18. intersection
    19. mobile_home_park
    20. nursing_home
    21. oil_gas_field
    22. oil_well
    23. overpass
    24. parking_lot
    25. parking_space
    26. railway
    27. river
    28. runway
    29. runway_marking
    30. shipping_yard
    31. solar_panel
    32. sparse_residential
    33. storage_tank
    34. swimming_pool
    35. tennis_court
    36. transformer_station
    37. wastewater_treatment_plant

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.isprsjprs.2018.01.004
    """

    url = "https://drive.google.com/file/d/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K"
    md5 = "96d54b3224c5350a98d55d5a7e6984ad"
    filename = "PatternNet.zip"
    directory = "images"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new PatternNet dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array (default=None returns PIL Image)
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

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            loader=loader,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

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
