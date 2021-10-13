# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet dataset."""

import glob
import json
import os
from typing import Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url, extract_archive


def sort_bands(x):
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split("_")[-1]
    x = os.path.splitext(x)[0]
    if x == "B8A":
        x = "B08A"
    return x


class BigEarthNet(VisionDataset):
    """BigEarthNet dataset.

    The `BigEarthNet <http://bigearth.net/>`_
    dataset is a dataset for multilabel remote sensing image scene classification.

    Dataset features:

    * 590,326 patches from 125 Sentinel-2 tiles in Europe between Jun 2017 - May 2018
    * 13 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 43 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image

    Dataset classes:

    0. Agro-forestry areas
    1. Airports
    2. Annual crops associated with permanent crops
    3. Bare rock
    4. Beaches, dunes, sands
    5. Broad-leaved forest
    6. Burnt areas
    7. Coastal lagoons
    8. Complex cultivation patterns
    9. Coniferous forest
    10. Construction sites
    11. Continuous urban fabric
    12. Discontinuous urban fabric
    13. Dump sites
    14. Estuaries
    15. Fruit trees and berry plantations
    16. Green urban areas
    17. Industrial or commercial units
    18. Inland marshes
    19. Intertidal flats
    20. Land principally occupied by agriculture, with significant areas of natural vege
    tation
    21. Mineral extraction sites
    22. Mixed forest
    23. Moors and heathland
    24. Natural grassland
    25. Non-irrigated arable land
    26. Olive groves
    27. Pastures
    28. Peatbogs
    29. Permanently irrigated land
    30. Port areas
    31. Rice fields
    32. Road and rail networks and associated land
    33. Salines
    34. Salt marshes
    35. Sclerophyllous vegetation
    36. Sea and ocean
    37. Sparsely vegetated areas
    38. Sport and leisure facilities
    39. Transitional woodland/shrub
    40. Vineyards
    41. Water bodies
    42. Water courses

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """

    classes = [
        "Agro-forestry areas",
        "Airports",
        "Annual crops associated with permanent crops",
        "Bare rock",
        "Beaches, dunes, sands",
        "Broad-leaved forest",
        "Burnt areas",
        "Coastal lagoons",
        "Complex cultivation patterns",
        "Coniferous forest",
        "Construction sites",
        "Continuous urban fabric",
        "Discontinuous urban fabric",
        "Dump sites",
        "Estuaries",
        "Fruit trees and berry plantations",
        "Green urban areas",
        "Industrial or commercial units",
        "Inland marshes",
        "Intertidal flats",
        "Land principally occupied by agriculture, with significant areas of "
        "natural vegetation",
        "Mineral extraction sites",
        "Mixed forest",
        "Moors and heathland",
        "Natural grassland",
        "Non-irrigated arable land",
        "Olive groves",
        "Pastures",
        "Peatbogs",
        "Permanently irrigated land",
        "Port areas",
        "Rice fields",
        "Road and rail networks and associated land",
        "Salines",
        "Salt marshes",
        "Sclerophyllous vegetation",
        "Sea and ocean",
        "Sparsely vegetated areas",
        "Sport and leisure facilities",
        "Transitional woodland/shrub",
        "Vineyards",
        "Water bodies",
        "Water courses",
    ]
    url = "http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz"
    md5 = "5a64e9ce38deb036a435a7b59494924c"
    filename = "BigEarthNet-S2-v1.0.tar.gz"
    directory = "BigEarthNet-v1.0"
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self._verify()
        self.files = glob.glob(os.path.join(self.root, self.directory, "*"))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {
            "image": image,
            "label": label,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        folder = self.files[index]
        paths = glob.glob(os.path.join(folder, "*.tif"))
        paths = sorted(paths, key=sort_bands)
        images = []
        for path in paths:
            # Images are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=dataset.count,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays = np.stack(images, axis=0)
        tensor: Tensor = torch.from_numpy(arrays)  # type: ignore[attr-defined]
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        folder = self.files[index]
        path = glob.glob(os.path.join(folder, "*.json"))[0]
        with open(path, "r") as f:
            labels = json.load(f)["labels"]
        indices = [self.class2idx[label] for label in labels]
        target: Tensor = torch.zeros(  # type: ignore[attr-defined]
            self.num_classes, dtype=torch.long  # type: ignore[attr-defined]
        )
        target[indices] = 1
        return target

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)
