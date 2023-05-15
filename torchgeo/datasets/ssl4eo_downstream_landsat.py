# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation Downstream Evaluation."""

import glob
import os
import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive


class SSL4EODownstream(NonGeoDataset):
    """SSL4EO Downstream Evaluation Dataset.

    Dataset is intended to be used for evaluation of SSL techniques.

    Dataset format:

    * input landsat image and single channel mask

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * Single multispectral GeoTIFF file

    """

    valid_input_sensors = ["l7-l1", "l7-l2", "l8-l1", "l8-l2"]
    valid_mask_products = ["cdl", "ncdl"]
    valid_splits = ["train", "val", "test"]

    data_root = "ssl4eo-*-conus"

    def __init__(
        self,
        root: str = "data",
        input_sensor: str = "l7-l1",
        mask_product: str = "cdl",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EODownstream instance.

        Args:
            root: root directory where dataset can be found
            input_sensor: input sensor source, one of ['l7-l1', 'l7-l2', 'l8-l1, 'l8-l2']
            mask_product: mask target matched to input_sensor
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``input_sensor`` argument is invalid
            AssertionError: if ``mask_product`` argument is invalid
            AssertionError: if ``split`` argument is invalid
        """
        assert (
            input_sensor in self.valid_input_sensors
        ), f"Only supports one of {self.valid_input_sensors}, but found {input_sensor}."
        self.input_sensor = input_sensor
        assert (
            mask_product in self.valid_mask_products
        ), f"Only supports one of {self.valid_mask_products}, but found {mask_product}."
        self.mask_product = mask_product
        assert (
            split in self.valid_splits
        ), f"Only supports one of {self.valid_splits}, but found {split}."
        self.split = split

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.sample_collection = self.retrieve_sample_collection()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        img_path, mask_path = self.sample_collection[index]

        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_collection)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
        """
        pass

    def _download(self) -> None:
        """Download the dataset."""
        pass

    def _extract(self) -> None:
        """Extract the dataset."""
        pass

    def retrieve_sample_collection(self) -> tuple[str]:
        """Retrieve paths to samples in data directory."""
        data_dir = self.data_root.replace("*", self.input_sensor)
        img_paths = sorted(
            glob.glob(
                os.path.join(self.root, data_dir, "imgs", "**", "**", "all_bands.tif"),
                recursive=True,
            )
        )
        sample_collection = [
            (
                img_path,
                img_path.replace("imgs", "masks").replace("all_bands.tif", "mask.tif"),
            )
            for img_path in img_paths
        ]
        return sample_collection

    def _load_image(self, path: str) -> Tensor:
        """Load the input image.

        Args:
            path: path to input image

        Returns:
            image
        """
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)
        return torch.from_numpy(image)

    def _load_mask(self, path: str) -> Tensor:
        """Load the mask.

        Args:
            path: path to mask

        Retuns:
            mask
        """
        with rasterio.open(path) as src:
            image = src.read()
        return torch.from_numpy(image).long()
