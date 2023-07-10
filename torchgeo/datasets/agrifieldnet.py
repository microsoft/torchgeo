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
        * https://doi.org/10.34911/rdnt.wu92p1

    .. versionadded:: 0.5
    """

    splits = ["train", "test"]

    collections = [
        "ref_agrifieldnet_competition_v1_source",
        "ref_agrifieldnet_competition_v1_labels_train",
        "ref_agrifieldnet_competition_v1_labels_test",
    ]

    image_meta = {
        "filename": "ref_agrifieldnet_competition_v1.tar.gz",
        "md5": "85055da1e7eb69fa4b3d925ee1450a74",
    }

    rgb_bands = ["B04", "B03", "B02"]
    all_bands = (
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
    )

    data_root = "ref_agrifieldnet_competition_v1"

    def __init__(
        self,
        root: str = "data",
        bands: tuple[str, ...] = all_bands,
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new AgriFieldNet dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            split: split selection which must be in ["train", "test"]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits
        self._validate_bands(bands)

        self.root = root
        self.bands = bands
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        if download:
            self._download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.source_tiles: list[str]
        self.train_tiles: list[str]
        self.test_tiles: list[str]

        self.source_tiles, self.train_tiles, self.test_tiles = self.get_tiles()

        self.source_image_fns = [
            os.path.join(
                root,
                "ref_agrifieldnet_competition_v1",
                "ref_agrifieldnet_competition_v1_source",
                "ref_agrifieldnet_competition_v1_source_" + tile,
            )
            for tile in self.source_tiles
        ]

        self.train_label_fns = [
            os.path.join(
                root,
                "ref_agrifieldnet_competition_v1",
                "ref_agrifieldnet_competition_v1_labels_train",
                "ref_agrifieldnet_competition_v1_labels_train_" + tile,
            )
            for tile in self.train_tiles
        ]

        # self.test_image_fns = [
        #     os.path.join(
        #         root,
        #         "ref_agrifieldnet_competition_v1",
        #         "ref_agrifieldnet_competition_v1_labels_test",
        #         "ref_agrifieldnet_competition_v1_labels_test_" + tile,
        #     )
        #     for tile in self.test_tiles
        # ]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, label, and field ids at that index
        """
        # if self.split == "train":
        #     tile_name = self.train_tiles[index]
        # else:
        #     tile_name = self.test_tiles[index]
        tile_name = self.train_tiles[index]
        print(tile_name)
        image = self._load_image_tile(tile_name)
        labels, field_ids = self._load_label_tile(tile_name)

        sample = {"image": image, "mask": labels, "field_ids": field_ids}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_tiles(self) -> tuple[list[str], list[str], list[str]]:
        """Get the tile names from the dataset directory.

        Returns:
            list of source, training, and testing tile names
        """
        source_collection = "ref_agrifieldnet_competition_v1_source"
        train_label = "ref_agrifieldnet_competition_v1_labels_train"
        test_label = "ref_agrifieldnet_competition_v1_labels_test"

        source_tiles = []
        for dir in os.walk(
            os.path.join(".", self.root, self.data_root, source_collection),
            topdown=True,
        ):
            source_tiles.append(dir[0][-5:])
        source_tiles = source_tiles[1:]

        train_tiles = []
        for dir in os.walk(
            os.path.join(".", self.root, self.data_root, train_label), topdown=True
        ):
            train_tiles.append(dir[0][-5:])
        train_tiles = train_tiles[1:]

        test_tiles = []
        for dir in os.walk(
            os.path.join(".", self.root, self.data_root, test_label), topdown=True
        ):
            test_tiles.append(dir[0][-5:])
        test_tiles = test_tiles[1:]

        return (source_tiles, train_tiles, test_tiles)

    def get_splits(self) -> tuple[list[int], list[int]]:
        """Get the field_ids for the train/test splits from the dataset directory.

        Returns:
            list of training field_ids and list of testing field_ids
        """
        train_field_ids = []
        test_field_ids = []

        for tile_name in self.train_tiles:
            directory = os.path.join(
                self.root,
                "ref_agrifieldnet_competition_v1",
                "ref_agrifieldnet_competition_v1_labels_train",
                "ref_agrifieldnet_competition_v1_labels_train_" + tile_name,
            )

            array = rasterio.open(os.path.join(directory, "field_ids.tif")).read(1)
            train_field_ids.extend(np.unique(array))

        for tile_name in self.test_tiles:
            directory = os.path.join(
                self.root,
                "ref_agrifieldnet_competition_v1",
                "ref_agrifieldnet_competition_v1_labels_test",
                "ref_agrifieldnet_competition_v1_labels_test_" + tile_name,
            )

            array = rasterio.open(os.path.join(directory, "field_ids.tif")).read(1)
            test_field_ids.extend(np.unique(array))

        return train_field_ids, test_field_ids

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for collection in self.collections:
            download_radiant_mlhub_collection(collection, self.root, api_key)

        pathname = os.path.join(self.root, "*.tar.gz")
        for tarfile in glob.iglob(pathname):
            extract_archive(tarfile)

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        images: bool = check_integrity(
            os.path.join(self.root, self.image_meta["filename"]),
            self.image_meta["md5"] if self.checksum else None,
        )

        # targets: bool = check_integrity(
        #     os.path.join(self.root, self.target_meta["filename"]),
        #     self.target_meta["md5"] if self.checksum else None,
        # )

        # return images and targets
        return images

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        # return len(self.source_image_fns)
        return len(self.train_label_fns)

    def _validate_bands(self, bands: tuple[str, ...]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided tuple of bands to load

        Raises:
            AssertionError: if ``bands`` is not a tuple
            ValueError: if an invalid band name is provided
        """
        assert isinstance(bands, tuple), "The list of bands must be a tuple"
        for band in bands:
            if band not in self.all_bands:
                raise ValueError(f"'{band}' is an invalid band name.")

    def _load_image_tile(
        self, tile_name: str, bands: tuple[str, ...] = all_bands
    ) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        assert tile_name in self.source_tiles

        path = os.path.join(
            self.root,
            "ref_agrifieldnet_competition_v1",
            "ref_agrifieldnet_competition_v1_source",
            "ref_agrifieldnet_competition_v1_source_" + tile_name,
        )

        bands_src = [
            rasterio.open(os.path.join(path, f"{band}.tif")).read(1)
            for band in self.all_bands
        ]
        img_tile = np.stack(bands_src)
        tensor = torch.from_numpy(img_tile)

        return tensor

    def _load_label_tile(self, tile_name: str) -> tuple[Tensor, Tensor]:
        """Load a single _tile_ of labels and field_ids.

        Args:
            tile_name: name of tile to load

        Returns:
            tuple of labels and field ids

        Raises:
            AssertionError: if ``tile_name`` is invalid
        """
        assert tile_name in self.train_tiles

        directory = os.path.join(
            self.root,
            "ref_agrifieldnet_competition_v1",
            "ref_agrifieldnet_competition_v1_labels_train",
            "ref_agrifieldnet_competition_v1_labels_train_" + tile_name,
        )

        array = rasterio.open(os.path.join(directory, "raster_labels.tif")).read(1)
        labels = torch.from_numpy(array.astype(np.uint8))

        array = rasterio.open(os.path.join(directory, "field_ids.tif")).read(1)
        field_ids = torch.from_numpy(array.astype(np.uint8))

        return (labels, field_ids)

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
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset does not contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"]

        num_panels = 2

        if "prediction" in sample:
            predictions = sample["prediction"]
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 10, 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(predictions)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
