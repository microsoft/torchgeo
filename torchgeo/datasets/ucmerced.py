# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced dataset."""

import os
import warnings
from functools import lru_cache
from typing import Callable, Dict, Optional

import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor

from .geo import VisionDataset
from .utils import check_integrity, download_and_extract_archive


class UCMerced(VisionDataset):
    """UC Merced dataset.

    The `UC Merced <http://weegee.vision.ucmerced.edu/datasets/landuse.html>`_
    dataset s a land use classification dataset of 2.1k 256x256 1ft resolution RGB
    images of urban locations around the U.S. extracted from the USGS National Map Urban
    Area Imagery collection with 21 land use classes (100 images per class).

    Dataset features:

    * land use class labels from around the U.S.
    * three spectral bands - RGB
    * 21 classes

    If you use this dataset in your research, please cite the following paper:

    * https://dl.acm.org/doi/10.1145/1869790.1869829
    """

    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"  # 318 MB
    filename = "UCMerced_LandUse.zip"
    md5 = "5b7ec56793786b6dc8a908e8854ac0e4"

    base_dir = os.path.join("UCMerced_LandUse", "Images")
    classes = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

    class_counts = {class_name: 100 for class_name in classes}
    class_name_to_label_idx = {class_name: i for i, class_name in enumerate(classes)}

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new UC Merced dataset instance.

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
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.fns = []
        self.labels = []
        for class_name, class_count in self.class_counts.items():
            for i in range(0, class_count):
                self.fns.append(
                    os.path.join(
                        self.root, self.base_dir, class_name, f"{class_name}{i:02d}.tif"
                    )
                )
                self.labels.append(self.class_name_to_label_idx[class_name])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: Dict[str, Tensor] = {
            "image": self._load_image(index),
            "label": torch.tensor(self.labels[index]),  # type: ignore[attr-defined]
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.labels)

    @lru_cache()
    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: unique ID of the image

        Returns:
            the image
        """
        filename = self.fns[index]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(filename) as f:
                array = f.read()
                tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        return tensor

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename),
            self.md5 if self.checksum else None,
        )

        return integrity

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
