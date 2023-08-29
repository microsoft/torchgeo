# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MapInWild dataset."""

import os
import shutil
from collections.abc import Sequence
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from einops import rearrange
from torch import Tensor

from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (
    download_url,
    extract_archive,
    percentile_normalization,
)


class MapInWild(NonGeoDataset):
    """MapInWild dataset.

    The `MapInWild <https://arxiv.org/abs/2212.02265>`_ dataset is curated for the task
    of wilderness mapping on a pixel-level. MapInWild is a multi-modal dataset and
    comprises various geodata acquired and formed from different RS sensors
    over 1018 locations: dual-pol Sentinel-1, four-season Sentinel-2 with 10 bands,
    ESA WorldCover map, and Visible Infrared Imaging Radiometer Suite
    NightTime Day/Night band. The dataset consists of 8144 images with the shape
    of 1920 × 1920 pixels. The images are weakly annotated from the
    World Database of Protected Areas (WDPA).

    Dataset features:
    * 1018 areas globally sampled from the WDPA
    * 10-Band Sentinel-2
    * Dual-pol Sentinel-1
    * ESA WorldCover Land Cover
    * Visible Infrared Imaging Radiometer Suite NightTime Day/Night Band

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/10089830

    .. note::
       This dataset requires the following additional library to be installed:

       * `pandas <https://pypi.org/project/pandas/>`_ to load CSV files

    .. versionadded:: 0.5
    """

    band_sets: dict[str, tuple[str, ...]] = {
        "all": (
            "VV",
            "VH",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
            "2020_Map",
            "avg_rad",
        ),
        "s1": ("VV", "VH"),
        "s2": ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"),
    }

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
        "split_ids": "split_IDs/split_IDs.csv",
    }

    filenames = [
        "ESA_WC.zip",
        "VIIRS.zip",
        "mask.zip",
        "s1_part1.zip",
        "s1_part2.zip",
        "s2_autumn_part1.zip",
        "s2_autumn_part2.zip",
        "s2_spring_part1.zip",
        "s2_spring_part2.zip",
        "s2_summer_part1.zip",
        "s2_summer_part2.zip",
        "s2_temporal_subset_part1.zip",
        "s2_temporal_subset_part2.zip",
        "s2_winter_part1.zip",
        "s2_winter_part2.zip",
    ]

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
            ImportError: if pandas is not installed
        """
        assert split in ["train", "validation", "test"]

        self.checksum = checksum
        self.root = root
        self.transforms = transforms
        self.modality = modality
        self.download = download

        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        modality.append("split_ids")

        for modal in modality:
            for modality_relativ_path in self.modality_urls[modal]:
                if modality_relativ_path.endswith("csv"):
                    self._verify(relativ_path=modality_relativ_path, md5=None)
                else:
                    self._verify(
                        relativ_path=modality_relativ_path,
                        md5=self.md5s[modality_relativ_path.split("/")[1]],
                    )

            #  Merge modalities downloaded and extracted in two parts.
            if modal not in os.listdir(root) and len(self.modality_urls[modal]) == 2:
                self._merge_parts(root, modal)

        # Masks will be loaded seperately in the :meth:`__getitem__`
        if "mask" in self.modality:
            self.modality.remove("mask")

        # Split IDs has been downloaded and not needed in the list
        if "split_ids" in self.modality:
            self.modality.remove("split_ids")

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
        list_modals = []
        id = self.ids[index]

        mask = self._load_raster(id, "mask")
        mask[mask != 0] = 1

        for mode in self.modality:
            mode = mode.upper() if mode in ["esa_wc", "viirs"] else mode
            data = self._load_raster(id, mode)
            list_modals.append(data)

        image = torch.cat(list_modals, dim=0)

        if self.transforms is not None:
            sample_: dict[str, Tensor] = {"image": image, "mask": mask}

            transformed = self.transforms(sample_)

            image = rearrange(transformed["image"], "h w c -> c h w")
            mask = rearrange(transformed["mask"], "h w c -> c h w")

        sample: dict[str, Tensor] = {"image": image, "mask": mask}

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
            source: the filename of the modality

        Returns:
            the raster image or target
        """
        with rasterio.open(os.path.join(self.root, source, f"{filename}.tif")) as f:
            array = f.read()
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array)
            return tensor

    def _verify(self, relativ_path: str, md5: Optional[str] = None) -> None:
        """Verify the integrity of the dataset.

        Args:
            url: url to the file
            md5: md5 of the file to be verified

        Raises:
            RuntimeError: if dataset is not found
        """
        url = os.path.join(self.url, relativ_path)

        # Check if the zip file has already been downloaded
        filepath = os.path.join(self.root, url.split("/")[-1])
        if os.path.exists(filepath):
            self._extract(url)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` directory and `download=False`,"  # noqa: E501
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        elif self.download and url.endswith(".csv"):
            # Download the split file
            download_url(url, self.root, filename=os.path.split(url)[1], md5=None)

        elif self.download and not url.endswith(".csv"):
            # Download the dataset
            self._download(url, md5)
            self._extract(url)

    def _download(self, url: str, md5: Optional[str]) -> None:
        """Download the dataset."""
        download_url(
            url,
            self.root,
            filename=os.path.split(url)[1],
            md5=md5 if self.checksum else None,
        )

    def _extract(self, url: str) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, os.path.split(url)[1])
        extract_archive(filepath)

    def _merge_parts(self, root: str, modality: str) -> None:
        """Merge the modalities that are downloaded and extracted in two parts.

        Args:
            root: root directory where dataset can be found
            modality: the filename of the modality
        """
        fname_p1 = modality + "_part1"
        fname_p2 = modality + "_part2"
        source_folder = os.path.join(root, fname_p1)
        destination_folder = os.path.join(root, fname_p2)
        for file_name in os.listdir(source_folder):
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_folder, file_name)
            if os.path.isfile(source):
                shutil.move(source, destination)

        shutil.rmtree(source_folder)
        dest_split = os.path.split(destination_folder)
        if len(modality.split("_")) == 3:
            rename_dest = os.path.join(dest_split[0], "s2_temporal_subset")
        if len(modality.split("_")) == 2:
            rename_dest = os.path.join(
                dest_split[0],
                dest_split[1].split("_")[0] + "_" + dest_split[1].split("_")[1],
            )
        if "_" not in modality:
            rename_dest = os.path.join(dest_split[0], dest_split[1].split("_")[0])
        os.rename(destination_folder, rename_dest)

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

    def _get_bands(
        self,
        image: "np.typing.NDArray[np.float32]",
        all_bands: Sequence[str],
        select_bands: Sequence[str],
    ) -> "np.typing.NDArray[np.float32]":
        """Filters the bands for a given set of bands.

        Args:
            image: the image whose bands to be filtered
            all_bands: all Sentinel-2 bands
            select_bands: bands to filter

        Returns:
            the raster image with filtered bands
        """
        bands = [
            all_bands.index(select_bands)
            if isinstance(select_bands, str)
            else select_bands
            for select_bands in select_bands
        ]
        return image[bands]

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample image-mask pair returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = rearrange(sample["image"], "h w c -> w c h")
        mask = sample["mask"].squeeze()
        color_mask = self._convert_to_color(mask, cmap=self.mask_cmap)

        # Land cover
        if np.all(np.isin(image, np.arange(0, 110, 10))) and image.shape[-1] == 1:
            image = self._convert_to_color(image.squeeze(), cmap=self.wc_cmap)
        # Sentinel-1
        elif image.shape[-1] == 2:
            image = image[:, :, 0]
            image = percentile_normalization(image)
        # Sentinel-2
        elif image.shape[-1] > 3:
            rgb_s2 = self._get_bands(
                image=np.einsum("ijk->kij", image),
                all_bands=self.band_sets["s2"],
                select_bands=["B4", "B3", "B2"],
            )
            image = percentile_normalization(np.einsum("ijk->jki", rgb_s2))
        # Night-time light
        else:
            image = percentile_normalization(image)

        num_panels = 2
        showing_predictions = "prediction" in sample

        if showing_predictions:
            predictions = sample["prediction"].numpy().squeeze()
            num_panels += 1
            color_predictions = self._convert_to_color(predictions, cmap=self.mask_cmap)

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(color_mask, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(color_predictions, vmin=0, vmax=1, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
