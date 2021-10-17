# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet dataset."""

import glob
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url, extract_archive


def sort_bands(x: str) -> str:
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

    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

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
    20. Land principally occupied by agriculture, with
        significant areas of natural vegetation
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
    metadata = {
        "s1": {
            "url": "http://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz",
            "md5": "5a64e9ce38deb036a435a7b59494924c",
            "filename": "BigEarthNet-S1-v1.0.tar.gz",
            "directory": "BigEarthNet-S1-v1.0",
        },
        "s2": {
            "url": "http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "5a64e9ce38deb036a435a7b59494924c",
            "filename": "BigEarthNet-S2-v1.0.tar.gz",
            "directory": "BigEarthNet-v1.0",
        },
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        bands: str = "all",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert bands in ["s1", "s2", "all"]
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self._verify()

        if bands == "s2":
            self.files = glob.glob(
                os.path.join(self.root, self.metadata["s2"]["directory"], "*")
            )
        else:
            self.files = glob.glob(
                os.path.join(self.root, self.metadata["s1"]["directory"], "*")
            )

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

    def _load_paths(self, index: int) -> List[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder = self.files[index]
        paths = glob.glob(os.path.join(folder, "*.tif"))
        # S1->S2 patch mapping is in S1 patch metadata json file
        if self.bands == "all":
            paths = sorted(paths)

            metadata_path = glob.glob(os.path.join(folder, "*.json"))[0]
            with open(metadata_path, "r") as f:
                name_s2 = json.load(f)["corresponding_s2_patch"]

            folder_s2 = os.path.join(
                self.root, self.metadata["s2"]["directory"], name_s2
            )
            paths_s2 = glob.glob(os.path.join(folder_s2, "*.tif"))
            paths_s2 = sorted(paths_s2, key=sort_bands)
            paths.extend(paths_s2)
        elif self.bands == "s1":
            paths = sorted(paths)
        else:
            paths = sorted(paths, key=sort_bands)
        return paths

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
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
        keys = ["s1", "s2"] if self.bands == "all" else [self.bands]
        urls = [self.metadata[k]["url"] for k in keys]
        md5s = [self.metadata[k]["md5"] for k in keys]
        filenames = [self.metadata[k]["filename"] for k in keys]
        directories = [self.metadata[k]["directory"] for k in keys]

        # Check if the files already exist
        exists = [
            os.path.exists(os.path.join(self.root, directory))
            for directory in directories
        ]
        if all(exists):
            return

        # Check if zip file already exists (if so then extract)
        exists = []
        for filename in filenames:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                exists.append(True)
                self._extract(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download and extract the dataset
        for url, filename, md5 in zip(urls, filenames, md5s):
            self._download(url, filename, md5)
            filepath = os.path.join(self.root, filename)
            self._extract(filepath)

    def _download(self, url: str, filename: str, md5: str) -> None:
        """Download the dataset.

        Args:
            url: url to download file
            filename: output filename to write downloaded file
            md5: md5 of downloaded file
        """
        download_url(
            url,
            self.root,
            filename=filename,
            md5=md5 if self.checksum else None,
        )

    def _extract(self, filepath: str) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        extract_archive(filepath)
