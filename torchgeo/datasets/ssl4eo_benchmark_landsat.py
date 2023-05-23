# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation Landsat Benchmark Evaluation."""

import glob
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from .geo import NonGeoDataset


class SSL4EOLBenchmark(NonGeoDataset):
    """SSL4EO Landsat Benchmark Evaluation Dataset.

    Dataset is intended to be used for evaluation of SSL techniques.

    Dataset format:

    * input landsat image and single channel mask

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * Single multispectral GeoTIFF file

    .. versionadded:: 0.5
    """

    valid_input_sensors = ["tm_toa", "etm_toa", "etm_sr", "oli_tirs_toa", "oli_sr"]
    valid_mask_products = ["cdl", "nlcd"]
    valid_splits = ["train", "val", "test"]

    image_root = "ssl4eo_l_*_benchmark"
    mask_dir_dict = {
        "tm_toa": "ssl4eo_l_tm_*",
        "etm_toa": "ssl4eo_l_etm_*",
        "etm_sr": "ssl4eo_l_etm_*",
        "oli_tirs_toa": "ssl4eo_l_oli_*",
        "oli_sr": "ssl4eo_l_oli_*",
    }

    year_dict = {
        "tm_toa": 2011,
        "etm_toa": 2019,
        "etm_sr": 2019,
        "oli_tirs_toa": 2019,
        "oli_sr": 2019,
    }

    rgb_indices = {
        "tm_toa": [2, 1, 0],
        "etm_toa": [2, 1, 0],
        "etm_sr": [2, 1, 0],
        "oli_tirs_toa": [3, 2, 1],
        "oli_sr": [3, 2, 1],
    }

    def __init__(
        self,
        root: str = "data",
        input_sensor: str = "oli_sr",
        mask_product: str = "cdl",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new SSL4EO Landsat Benchmark instance.

        Args:
            root: root directory where dataset can be found
            input_sensor: one of ['etm_toa', 'etm_sr', 'oli_tirs_toa, 'oli_sr']
            mask_product: mask target matched to input_sensor
            split: dataset split, one of ['train', 'val', 'test']
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            AssertionError: if any arguments are invalid
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

    def retrieve_sample_collection(self) -> list[tuple[str]]:
        """Retrieve paths to samples in data directory."""
        img_data_dir = self.image_root.replace("*", self.input_sensor)
        mask_dir = self.mask_dir_dict[self.input_sensor].replace("*", self.mask_product)
        img_paths = glob.glob(
            os.path.join(self.root, img_data_dir, "**", "all_bands.tif"), recursive=True
        )
        sample_collection: list[tuple[str]] = []
        for img_path in img_paths:
            mask_path = img_path.replace(img_data_dir, mask_dir).replace(
                "all_bands.tif",
                f"{self.mask_product}_{self.year_dict[self.input_sensor]}.tif",
            )
            sample_collection.append((img_path, mask_path))
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
        ncols = 2
        image = sample["image"][self.rgb_indices[self.input_sensor]].permute(1, 2, 0)
        image = image.numpy() / 255
        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"].squeeze(0).numpy()
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(mask)
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(prediction_mask)
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
