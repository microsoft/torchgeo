# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet dataset."""

import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from rasterio.enums import Resampling
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from .geo import VisionDataset
from .utils import download_url, extract_archive

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


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
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

    Dataset classes (43):

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
    20. Land principally occupied by agriculture, with significant
        areas of natural vegetation
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

    Dataset classes (19):

    0. Urban fabric
    1. Industrial or commercial units
    2. Arable land
    3. Permanent crops
    4. Pastures
    5. Complex cultivation patterns
    6. Land principally occupied by agriculture, with significant
        areas of natural vegetation
    7. Agro-forestry areas
    8. Broad-leaved forest
    9. Coniferous forest
    10. Mixed forest
    11. Natural grassland and sparsely vegetated areas
    12. Moors, heathland and sclerophyllous vegetation
    13. Transitional woodland, shrub
    14. Beaches, dunes, sands
    15. Inland wetlands
    16. Coastal wetlands
    17. Inland waters
    18. Marine waters

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """

    classes_43 = [
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
    classes_19 = [
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural "
        "vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    ]
    label_converter = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }
    splits_metadata = {
        "train": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/train.csv?inline=false",  # noqa: E501
            "filename": "bigearthnet-train.csv",
            "md5": "623e501b38ab7b12fe44f0083c00986d",
        },
        "val": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/val.csv?inline=false",  # noqa: E501
            "filename": "bigearthnet-val.csv",
            "md5": "22efe8ed9cbd71fa10742ff7df2b7978",
        },
        "test": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/test.csv?inline=false",  # noqa: E501
            "filename": "bigearthnet-test.csv",
            "md5": "697fb90677e30571b9ac7699b7e5b432",
        },
    }
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
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits_metadata
        assert bands in ["s1", "s2", "all"]
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.classes_43)}
        self._verify()
        self.folders = self._load_folders()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _load_folders(self) -> List[Dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        folders = [
            {
                "s1": os.path.join(self.root, dir_s1, pair[1]),
                "s2": os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders

    def _load_paths(self, index: int) -> List[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        if self.bands == "all":
            folder_s1 = self.folders[index]["s1"]
            folder_s2 = self.folders[index]["s2"]
            paths_s1 = glob.glob(os.path.join(folder_s1, "*.tif"))
            paths_s2 = glob.glob(os.path.join(folder_s2, "*.tif"))
            paths_s1 = sorted(paths_s1)
            paths_s2 = sorted(paths_s2, key=sort_bands)
            paths = paths_s1 + paths_s2
        elif self.bands == "s1":
            folder = self.folders[index]["s1"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths)
        else:
            folder = self.folders[index]["s2"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
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
        if self.bands == "s2":
            folder = self.folders[index]["s2"]
        else:
            folder = self.folders[index]["s1"]

        path = glob.glob(os.path.join(folder, "*.json"))[0]
        with open(path, "r") as f:
            labels = json.load(f)["labels"]

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]

        # Map 43 to 19 class labels
        if self.num_classes == 19:
            indices = [
                self.label_converter.get(idx) for idx in indices  # type: ignore[misc]
            ]
            indices = [idx for idx in indices if idx is not None]

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
        urls.extend([self.splits_metadata[k]["url"] for k in self.splits_metadata])
        md5s.extend([self.splits_metadata[k]["md5"] for k in self.splits_metadata])
        filenames_splits = [
            self.splits_metadata[k]["filename"] for k in self.splits_metadata
        ]
        filenames.extend(filenames_splits)

        # Check if the split file already exist
        exists = []
        for filename in filenames_splits:
            exists.append(os.path.exists(os.path.join(self.root, filename)))

        # Check if the files already exist
        for directory in directories:
            exists.append(os.path.exists(os.path.join(self.root, directory)))

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
        if not os.path.exists(filename):
            download_url(
                url, self.root, filename=filename, md5=md5 if self.checksum else None
            )

    def _extract(self, filepath: str) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        if not filepath.endswith(".csv"):
            extract_archive(filepath)


class BigEarthNetDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the BigEarthNet dataset.

    Uses the train/val/test splits from the dataset.
    """

    # (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    # min/max band statistics computed on 100k random samples
    band_mins_raw = torch.tensor(  # type: ignore[attr-defined]
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    band_maxs_raw = torch.tensor(  # type: ignore[attr-defined]
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    band_mins = torch.tensor(  # type: ignore[attr-defined]
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    band_maxs = torch.tensor(  # type: ignore[attr-defined]
        [
            6.0,
            16.0,
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        bands: str = "all",
        num_classes: int = 19,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for BigEarthNet based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the BigEarthNet Dataset classes
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.bands = bands
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        if bands == "all":
            self.mins = self.band_mins[:, None, None]
            self.maxs = self.band_maxs[:, None, None]
        elif bands == "s1":
            self.mins = self.band_mins[:2, None, None]
            self.maxs = self.band_maxs[:2, None, None]
        else:
            self.mins = self.band_mins[2:, None, None]
            self.maxs = self.band_maxs[2:, None, None]

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()
        sample["image"] = (sample["image"] - self.mins) / (self.maxs - self.mins)
        sample["image"] = torch.clip(  # type: ignore[attr-defined]
            sample["image"], min=0.0, max=1.0
        )
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        BigEarthNet(self.root_dir, split="train", bands=self.bands, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        transforms = Compose([self.preprocess])
        self.train_dataset = BigEarthNet(
            self.root_dir,
            split="train",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
        )
        self.val_dataset = BigEarthNet(
            self.root_dir,
            split="val",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
        )
        self.test_dataset = BigEarthNet(
            self.root_dir,
            split="test",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
