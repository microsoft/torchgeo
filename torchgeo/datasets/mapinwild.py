# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MapInWild dataset."""

import os
import shutil
from collections import defaultdict
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    DatasetNotFoundError,
    check_integrity,
    download_url,
    extract_archive,
    percentile_normalization,
)


class MapInWild(NonGeoDataset):
    """MapInWild dataset.

    The `MapInWild <https://ieeexplore.ieee.org/document/10089830>`__ dataset is
    curated for the task of wilderness mapping on a pixel-level. MapInWild is a
    multi-modal dataset and comprises various geodata acquired and formed from
    different RS sensors over 1018 locations: dual-pol Sentinel-1, four-season
    Sentinel-2 with 10 bands, ESA WorldCover map, and Visible Infrared Imaging
    Radiometer Suite NightTime Day/Night band. The dataset consists of 8144
    images with the shape of 1920 × 1920 pixels. The images are weakly annotated
    from the World Database of Protected Areas (WDPA).

    Dataset features:

    * 1018 areas globally sampled from the WDPA
    * 10-Band Sentinel-2
    * Dual-pol Sentinel-1
    * ESA WorldCover Land Cover
    * Visible Infrared Imaging Radiometer Suite NightTime Day/Night Band

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/10089830

    .. versionadded:: 0.5
    """

    url = "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/"

    modality_urls = {
        "esa_wc": {"esa_wc/ESA_WC.zip"},
        "viirs": {"viirs/VIIRS.zip"},
        "mask": {"mask/mask.zip"},
        "s1": {"s1/s1_part1.zip", "s1/s1_part2.zip"},
        "s2_temporal_subset": {
            "s2_temporal_subset/s2_temporal_subset_part1.zip",
            "s2_temporal_subset/s2_temporal_subset_part2.zip",
        },
        "s2_autumn": {"s2_autumn/s2_autumn_part1.zip", "s2_autumn/s2_autumn_part2.zip"},
        "s2_spring": {"s2_spring/s2_spring_part1.zip", "s2_spring/s2_spring_part2.zip"},
        "s2_summer": {"s2_summer/s2_summer_part1.zip", "s2_summer/s2_summer_part2.zip"},
        "s2_winter": {"s2_winter/s2_winter_part1.zip", "s2_winter/s2_winter_part2.zip"},
        "split_IDs": {"split_IDs/split_IDs.csv"},
    }

    md5s = {
        "ESA_WC.zip": "72b2ee578fe10f0df85bdb7f19311c92",
        "VIIRS.zip": "4eff014bae127fe536f8a5f17d89ecb4",
        "mask.zip": "87c83a23a73998ad60d448d240b66225",
        "s1_part1.zip": "d8a911f5c76b50eb0760b8f0047e4674",
        "s1_part2.zip": "a30369d17c62d2af5aa52a4189590e3c",
        "s2_temporal_subset_part1.zip": "78c2d05514458a036fe133f1e2f11d2a",
        "s2_temporal_subset_part2.zip": "076cd3bd00eb5b7f5d80c9e0a0de0275",
        "s2_autumn_part1.zip": "6ee7d1ac44b5107e3663636269aecf68",
        "s2_autumn_part2.zip": "4fc5e1d5c772421dba553722433ac3b9",
        "s2_spring_part1.zip": "2a89687d8fafa7fc7f5e641bfa97d472",
        "s2_spring_part2.zip": "5845dcae0ab3cdc174b7c41edd4283a9",
        "s2_summer_part1.zip": "73ca8291d3f4fb7533636220a816bb71",
        "s2_summer_part2.zip": "5b5816bbd32987619bf72cde5cacd032",
        "s2_winter_part1.zip": "ca958f7cd98e37cb59d6f3877573ee6d",
        "s2_winter_part2.zip": "e7aacb0806d6d619b6abc408e6b09fdc",
        "split_IDs.csv": "cb5c6c073702acee23544e1e6fe5856f",
    }

    mask_cmap = {1: (0, 153, 0), 0: (255, 255, 255)}

    wc_cmap = {
        10: (0, 160, 0),
        20: (150, 100, 0),
        30: (255, 180, 0),
        40: (255, 255, 100),
        50: (195, 20, 0),
        60: (255, 245, 215),
        70: (255, 255, 255),
        80: (0, 70, 200),
        90: (0, 220, 130),
        95: (0, 150, 120),
        100: (255, 235, 175),
    }

    def __init__(
        self,
        root: str = "data",
        modality: list[str] = ["mask", "esa_wc", "viirs", "s2_summer"],
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new MapInWild dataset instance.

        Args:
            root: root directory where dataset can be found
            modality: the modality to download. Choose from: "mask", "esa_wc",
                "viirs", "s1", "s2_temporal_subset", "s2_[season]".
            split: one of "train", "validation", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in ["train", "validation", "test"]

        self.checksum = checksum
        self.root = root
        self.transforms = transforms
        self.modality = modality
        self.download = download

        modality.append("split_IDs")
        for mode in modality:
            for modality_link in self.modality_urls[mode]:
                modality_url = os.path.join(self.url, modality_link)
                self._verify(
                    url=modality_url, md5=self.md5s[os.path.split(modality_link)[-1]]
                )

            # Merge modalities downloaded in two parts
            if (
                download
                and mode not in os.listdir(self.root)
                and len(self.modality_urls[mode]) == 2
            ):
                self._merge_parts(mode)

        # Masks will be loaded seperately in the :meth:`__getitem__`
        if "mask" in self.modality:
            self.modality.remove("mask")

        # Split IDs has been downloaded and is not needed in the list
        if "split_IDs" in self.modality:
            self.modality.remove("split_IDs")

        if os.path.exists(os.path.join(self.root, "split_IDs.csv")):
            split_dataframe = pd.read_csv(os.path.join(self.root, "split_IDs.csv"))
            self.ids = split_dataframe[split].dropna().values.tolist()
            self.ids = list(map(int, self.ids))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        list_modalities = []
        id = self.ids[index]

        mask = self._load_raster(id, "mask")
        mask[mask != 0] = 1

        for mode in self.modality:
            mode = mode.upper() if mode in ["esa_wc", "viirs"] else mode
            data = self._load_raster(id, mode)
            list_modalities.append(data)

        image = torch.cat(list_modalities, dim=0)

        sample: dict[str, Tensor] = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_raster(self, filename: int, source: str) -> Tensor:
        """Load a single raster image or target.

        Args:
            filename: name of the file to load
            source: the directory of the modality

        Returns:
            the raster image or target
        """
        with rasterio.open(os.path.join(self.root, source, f"{filename}.tif")) as f:
            raw_array = f.read()
            array: "np.typing.NDArray[np.int_]" = np.stack(raw_array, axis=0)
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array)
            tensor = tensor.float()
            return tensor

    def _verify(self, url: str, md5: Optional[str] = None) -> None:
        """Verify the integrity of the dataset.

        Args:
            url: url to the file
            md5: md5 of the file to be verified
        """
        modality_folder_name = url.split("/")[-1]
        mod_fold_no_ext = modality_folder_name.split(".")[0]
        modality_path = os.path.join(self.root, mod_fold_no_ext)
        split_path = os.path.join(self.root, modality_folder_name)
        if mod_fold_no_ext == "split_IDs":
            modality_path = split_path

        # Check if the files already exist
        if os.path.exists(modality_path):
            return

        # Check if the zip files have already been downloaded, if so, extract
        filepath = os.path.join(self.root, url.split("/")[-1])
        if os.path.isfile(filepath) and filepath.endswith(".zip"):
            if self.checksum and not check_integrity(filepath, md5):
                raise RuntimeError("Dataset found, but corrupted.")
            self._extract(url)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download(url, md5)
        if not url.endswith(".csv"):
            self._extract(url)

    def _download(self, url: str, md5: Optional[str]) -> None:
        """Downloads a modality.

        Args:
            url: download url of a modality
            md5: md5 of a modality
        """
        download_url(
            url,
            self.root,
            filename=os.path.split(url)[1],
            md5=md5 if self.checksum else None,
        )

    def _extract(self, path: str) -> None:
        """Extracts a modality.

        Args:
            path: path to the modality folder
        """
        filepath = os.path.join(self.root, os.path.split(path)[1])
        extract_archive(filepath)

    def _merge_parts(self, modality: str) -> None:
        """Merge the modalities that are downloaded and extracted in two parts.

        Args:
            root: root directory where dataset can be found
            modality: the filename of the modality
        """
        # Create a new folder named after the 'modality' variable
        modality_folder = os.path.join(self.root, modality)
        # Will not raise an error if the folder already exists
        os.makedirs(modality_folder, exist_ok=True)

        # List of source folders
        source_folders = [
            os.path.join(self.root, modality + "_part1"),
            os.path.join(self.root, modality + "_part2"),
        ]

        # Move files from each source folder to the new 'modality' folder
        for source_folder in source_folders:
            for file_name in os.listdir(source_folder):
                source = os.path.join(source_folder, file_name)
                destination = os.path.join(modality_folder, file_name)
                if os.path.isfile(source):
                    shutil.copy(source, destination)  # Move files to 'modality' folder

    def _convert_to_color(
        self, arr_2d: Tensor, cmap: dict[int, tuple[int, int, int]]
    ) -> "np.typing.NDArray[np.uint8]":
        """Numeric labels to RGB-color encoding.

        Args:
            arr_2d: 2D array to be colorized
            cmap: colormap to use when mapping the labels

        Returns:
            3D colored image
        """
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in cmap.items():
            m = arr_2d == c
            arr_3d[m] = i
        return arr_3d

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample image-mask pair returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        modality_channels = defaultdict(lambda: 10, {"viirs": 1, "esa_wc": 1, "s1": 2})

        start_idx = 0
        split_images = {}

        for modality in self.modality:
            end_idx = start_idx + modality_channels[modality]  # Start + n of channels
            split_images[modality] = sample["image"][start_idx:end_idx, :, :]  # Slicing
            start_idx = end_idx  # Update the iterator

        # Prepare the mask
        mask = sample["mask"].squeeze()
        color_mask = self._convert_to_color(mask, cmap=self.mask_cmap)

        num_subplots = len(split_images) + 1  # +1 for color_mask
        showing_predictions = "prediction" in sample
        if showing_predictions:
            num_subplots += 1

        fig, axs = plt.subplots(1, num_subplots, figsize=(num_subplots * 4, 5))

        # Plot each modality in its respective axis
        for i, (modality, image) in enumerate(split_images.items()):
            ax = axs[i]
            img = np.transpose(image, (1, 2, 0)).squeeze()
            # Apply transformations based on modality type
            if modality.startswith("s2"):
                img = img[:, :, [4, 3, 2]]
            if modality == "esa_wc":
                img = self._convert_to_color(torch.as_tensor(img), cmap=self.wc_cmap)
            if modality == "s1":
                img = img[:, :, 0]

            if not "esa_wc":
                img = percentile_normalization(img)

            ax.imshow(img)
            if show_titles:
                ax.set_title(modality)
            ax.axis("off")

        # Plot color_mask in its own axis
        axs[len(split_images)].imshow(color_mask)
        if show_titles:
            axs[len(split_images)].set_title("Annotation")
        axs[len(split_images)].axis("off")

        # If available, plot predictions in a new axis
        if showing_predictions:
            prediction = sample["prediction"].squeeze()
            color_predictions = self._convert_to_color(prediction, cmap=self.mask_cmap)
            axs[-1].imshow(color_predictions, vmin=0, vmax=1, interpolation="none")
            if show_titles:
                axs[-1].set_title("Prediction")
            axs[-1].axis("off")

        plt.tight_layout()
        return fig
