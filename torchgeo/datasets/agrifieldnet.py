# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""AgriFieldNet India Challenge dataset."""

import glob
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_radiant_mlhub_collection, extract_archive


class AgriFieldNet(NonGeoDataset):
    """AgriFieldNet India Challenge dataset.

    The `AgriFieldNet India Challenge
    <https://zindi.africa/competitions/agrifieldnet-india-challenge>`__ dataset
    includes satellite imagery from Sentinel-2 cloud free composites
    (single snapshot) and labels for crop type that were collected by ground survey.
    The Sentinel-2 data are then matched with corresponding labels.
    The dataset contains 7081 fields, which have been split into training and
    test sets (5551 fields in the train and 1530 fields in the test).
    Satellite imagery and labels are tiled into 256x256 chips adding up to 1217 tiles.
    The fields are distributed across all chips, some chips may only have train or
    test fields and some may have both. Since the labels are derived from data
    collected on the ground, not all the pixels are labeled in each chip.
    If the field ID for a pixel is set to 0 it means that pixel is not included in
    either of the train or test set (and correspondingly the crop label
    will be 0 as well). For this challenge train and test sets have slightly
    different crop type distributions. The train set follows the distribution
    of ground reference data which is a skewed distribution with a few dominant
    crops being over represented. The test set was drawn randomly from an area
    weighted field list that ensured that fields with less common crop types
    were better represented in the test set. The original dataset can be
    downloaded from `Radiant MLHub <https://mlhub.earth/data/
    ref_agrifieldnet_competition_v1>`__.

    Dataset format:

    * images are 12-bands Sentinel-2 data
    * masks are tiff image with unique values representing the class and field id

    Dataset classes:

    1 - Wheat
    2 - Mustard
    3 - Lentil
    4 - No Crop/Fallow
    5 - Green pea
    6 - Sugarcane
    8 - Garlic
    9 - Maize
    13 - Gram
    14 - Coriander
    15 - Potato
    16 - Bersem
    36 - Rice

    If you use this dataset in your research, please cite the following dataset:
        Radiant Earth Foundation & IDinsight (2022) AgriFieldNet Competition Dataset,
        Version 1.0, Radiant MLHub. https://doi.org/10.34911/rdnt.wu92p1

    """

    filename = "ref_agrifieldnet_competition_v1.tar.gz"
    md5 = "85055da1e7eb69fa4b3d925ee1450a74"
    splits = ["train", "test"]
    collections = [
        "ref_agrifieldnet_competition_v1_source",
        "ref_agrifieldnet_competition_v1_labels_train",
        "ref_agrifieldnet_competition_v1_labels_test",
    ]

    # missing class corresponding values
    classes = [
        "Wheat",
        "Mustard",
        "Lentil",
        "No Crop/Fallow",
        "Green pea",
        "Sugarcane",
        "Garlic",
        "Maize",
        "Gram",
        "Coriander",
        "Potato",
        "Bersem",
        "Rice",
    ]

    rgb_bands = ["B04", "B03", "B02"]
    all_bands = [
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
        "B11",
        "B12",
    ]

    data_root = "ref_agrifieldnet_competition_v1"
    assets = ["field_ids", "raster_labels"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new AgriFieldNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        if download:
            self._download(api_key)

        # if split == "train":
        #     split_folder = f"{self.data_root}_labels_train"
        # else:
        #     split_folder = f"{self.data_root}_labels_test"

        source_collection = f"{self.data_root}_source"
        train_label = f"{self.data_root}_labels_train"

        train_folder_ids = []
        for dir in os.walk(
            os.path.join(".", root, self.data_root, train_label), topdown=True
        ):
            train_folder_ids.append(dir[0][-5:])

        train_folder_ids = train_folder_ids[1:]

        self.train_field_fns = [
            os.path.join(
                root, self.data_root, train_label, f"{train_label}_{i}", "field_ids.tif"
            )
            for i in train_folder_ids
        ]
        self.image_fns = [
            os.path.join(
                root, self.data_root, source_collection, f"{source_collection}_{i}"
            )
            for i in train_folder_ids
        ]

        self.mask_fns = [
            os.path.join(root, self.data_root, train_label, f"{train_label}_{i}")
            for i in train_folder_ids
        ]

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.data_root)):
            return

        # Check if .zip file already exists (if so extract)
        filepath = os.path.join(self.root, self.filename)

        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root`, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it."""
        for collection in self.collections:
            download_radiant_mlhub_collection(collection, self.root, api_key)

        pathname = os.path.join(self.root, "*.tar.gz")
        for tarfile in glob.iglob(pathname):
            extract_archive(tarfile)

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.image_fns)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        mask = self._load_target(index)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.image_fns[index]

        bands_src = [
            rasterio.open(os.path.join(path, f"{band}.tif")).read(1)
            for band in self.all_bands
        ]
        img_tile = np.stack(bands_src)
        tensor = torch.from_numpy(img_tile)
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        path = self.mask_fns[index]

        mask_src = rasterio.open(os.path.join(path, "raster_labels.tif")).read(1)
        tensor = torch.from_numpy(mask_src.astype(np.uint8))
        return tensor

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.all_bands:
                rgb_indices.append(self.all_bands.index(band))
            else:
                raise ValueError("Dataset does not contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2

        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy().astype("uint8").squeeze()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 10, 10))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
