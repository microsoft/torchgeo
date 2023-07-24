# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""AgriFieldNet India Challenge dataset."""

import glob
import json
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    splits = ["train", "predict"]

    collections = [
        "ref_agrifieldnet_competition_v1_source",
        "ref_agrifieldnet_competition_v1_labels_train",
        "ref_agrifieldnet_competition_v1_labels_test",
    ]

    source_meta = {
        "filename": "ref_agrifieldnet_competition_v1_source.tar.gz",
        "md5": "62ec758cc5c4d58f73c47be07f3d9d73",
    }
    image_meta = {
        "filename": "ref_agrifieldnet_competition_v1_labels_train.tar.gz",
        "md5": "d5c8d7fa8e1e28ecec211c3b3633fb17",
    }
    target_meta = {
        "filename": "ref_agrifieldnet_competition_v1_labels_test.tar.gz",
        "md5": "8aa638cdbd7cd38da37c3e9fd77c3d4c",
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

    label_mapper = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        8: 6,
        9: 7,
        13: 8,
        14: 9,
        15: 10,
        16: 11,
        36: 12,
        0: -999,
    }

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
            split: split selection which must be in ["train", "predict"]
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

        self.train_tiles: list[str]
        self.test_tiles: list[str]

        self.train_tiles, self.test_tiles = self.get_tiles()

        self.train_label_fns = [
            os.path.join(
                root,
                "ref_agrifieldnet_competition_v1_labels_train",
                "ref_agrifieldnet_competition_v1_labels_train_" + tile,
            )
            for tile in self.train_tiles
        ]
        self.test_image_fns = [
            os.path.join(
                root,
                "ref_agrifieldnet_competition_v1_labels_test",
                "ref_agrifieldnet_competition_v1_labels_test_" + tile,
            )
            for tile in self.test_tiles
        ]

        print("================== current split: ", self.split)

        if self.split == "predict":
            print("getting splits...")
            self.test_field_ids = self.get_splits()

        # self.train_field_ids, self.test_field_ids = self.get_splits()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, label, and field ids at that index
        """
        if self.split == "train":
            tile_name = self.train_tiles[index]
            image = self._load_image_tile(tile_name)
            labels, field_ids = self._load_label_tile(tile_name)
            sample = {"image": image, "mask": labels, "field_ids": field_ids}

        if self.split == "predict":
            tile_name = self.test_tiles[index]
            image = self._load_image_tile(tile_name)
            field_ids = self._load_label_tile(tile_name)
            sample = {"image": image, "field_ids": field_ids}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_tiles(self) -> tuple[list[str], list[str]]:
        """Get the tile names from the dataset directory.

        Returns:
            list of source, training, and testing tile names
        """
        train_label = "ref_agrifieldnet_competition_v1_labels_train"
        test_label = "ref_agrifieldnet_competition_v1_labels_test"

        with open(os.path.join(self.root, train_label, "collection.json")) as f:
            train_json = json.load(f)
        train_tiles = [
            i["href"].split("_")[-1].split(".")[0][:-5]
            for i in train_json["links"][2:-1]
        ]

        with open(os.path.join(self.root, test_label, "collection.json")) as f:
            test_json = json.load(f)
        test_tiles = [
            i["href"].split("_")[-1].split(".")[0][:-5]
            for i in test_json["links"][2:-1]
        ]

        return (train_tiles, test_tiles)

    def get_splits(self) -> list[int]:  # tuple[list[int], list[int]]:
        """Get the field_ids for the train/test splits from the dataset directory.

        Returns:
            list of training field_ids and list of testing field_ids
        """
        # train_field_ids = np.empty((0, 1))
        test_field_ids = np.empty((0, 1))

        # for tile_name in self.train_tiles:
        #    directory = os.path.join(
        #        self.root,
        #        "ref_agrifieldnet_competition_v1_labels_train",
        #        "ref_agrifieldnet_competition_v1_labels_train_" + tile_name,
        #    )

        #    field_array = rasterio.open(os.path.join(
        #        directory,
        #        "field_ids.tif")).read(1)
        #    train_field_ids = np.append(train_field_ids, field_array.flatten())
        # train_field_ids = np.unique(train_field_ids[train_field_ids != 0])

        for tile_name in self.test_tiles:
            directory = os.path.join(
                self.root,
                "ref_agrifieldnet_competition_v1_labels_test",
                "ref_agrifieldnet_competition_v1_labels_test_" + tile_name,
            )
            field_array = rasterio.open(os.path.join(directory, "field_ids.tif")).read(
                1
            )
            test_field_ids = np.append(test_field_ids, field_array.flatten())
        test_field_ids = np.unique(test_field_ids[test_field_ids != 0])

        # return train_field_ids, test_field_ids
        return test_field_ids

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
        sources: bool = check_integrity(
            os.path.join(self.root, self.source_meta["filename"]),
            self.source_meta["md5"] if self.checksum else None,
        )

        images: bool = check_integrity(
            os.path.join(self.root, self.image_meta["filename"]),
            self.image_meta["md5"] if self.checksum else None,
        )

        targets: bool = check_integrity(
            os.path.join(self.root, self.target_meta["filename"]),
            self.target_meta["md5"] if self.checksum else None,
        )

        return sources and images and targets

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        if self.split == "train":
            return len(self.train_label_fns)
        if self.split == "predict":
            return len(self.test_image_fns)

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
        # assert tile_name in self.source_tiles

        path = os.path.join(
            self.root,
            "ref_agrifieldnet_competition_v1_source",
            "ref_agrifieldnet_competition_v1_source_" + tile_name,
        )

        bands_src = [
            rasterio.open(os.path.join(path, f"{band}.tif")).read(1)
            for band in self.all_bands
        ]
        img_tile = np.stack(bands_src)
        tensor = torch.from_numpy(img_tile.astype(np.float32))

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
        if self.split == "train":
            assert tile_name in self.train_tiles
            directory = os.path.join(
                self.root,
                "ref_agrifieldnet_competition_v1_labels_train",
                "ref_agrifieldnet_competition_v1_labels_train_" + tile_name,
            )
            array = rasterio.open(os.path.join(directory, "raster_labels.tif")).read(1)
            array = np.vectorize(lambda x: self.label_mapper[x])(array)
            labels = torch.tensor(array, dtype=torch.long)

            array = rasterio.open(os.path.join(directory, "field_ids.tif")).read(1)
            field_ids = torch.tensor(array.astype(np.int32), dtype=torch.long)

            return (labels, field_ids)

        if self.split == "predict":
            assert tile_name in self.test_tiles
            directory = os.path.join(
                self.root,
                "ref_agrifieldnet_competition_v1_labels_test",
                "ref_agrifieldnet_competition_v1_labels_test_" + tile_name,
            )
            array = rasterio.open(os.path.join(directory, "field_ids.tif")).read(1)
            field_ids = torch.tensor(array.astype(np.int32), dtype=torch.long)

            return field_ids

    def create_submission(self, predictions: list[float]) -> None:
        """Create a submission file for the competition.

        Args:
            predictions: list of predictions for the test set
        """
        test_fields, test_predictions = self.compute_prediction()

        crop_labels = [
            "Wheat",
            "Mustard",
            "Lentil",
            "No Crop",
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

        results = pd.DataFrame(test_fields, columns=["field_id"])
        results[crop_labels] = test_predictions
        results = results.groupby("field_id").mean().reset_index()

        results.to_csv(os.path.join(self.root, "pixelwise-unet.csv"), index=False)

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
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
