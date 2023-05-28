# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation Benchmark Datasets."""

import glob
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor

from .cdl import CDL
from .geo import NonGeoDataset
from .nlcd import NLCD
from .utils import download_url, extract_archive


class SSL4EOLBenchmark(NonGeoDataset):
    """SSL4EO Landsat Benchmark Evaluation Dataset.

    Dataset is intended to be used for evaluation of SSL techniques. Each
    benchmark dataset consists of 25,000 images with corresponding land
    cover classification masks.

    Dataset format:

    * Input landsat image and single channel mask
    * 25,000 total samples split into train, val, test (70%, 15%, 15%)
    * NLCD dataset version has 17 classes
    * CDL dataset version has 134 classes

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * Single multispectral GeoTIFF file

    .. versionadded:: 0.5
    """

    url = "https://huggingface.co/datasets/torchgeo/{}/resolve/main/{}.tar.gz"

    valid_input_sensors = ["tm_toa", "etm_toa", "etm_sr", "oli_tirs_toa", "oli_sr"]
    valid_mask_products = ["cdl", "nlcd"]
    valid_splits = ["train", "val", "test"]

    image_root = "ssl4eo_l_{}_benchmark"
    img_md5s = {
        "tm_toa": "8e3c5bcd56d3780a442f1332013b8d15",
        "etm_toa": "1b051c7fe4d61c581b341370c9e76f1f",
        "etm_sr": "34a24fa89a801654f8d01e054662c8cd",
        "oli_tirs_toa": "6e9d7cf0392e1de2cbdb39962ba591aa",
        "oli_sr": "0700cd15cc2366fe68c2f8c02fa09a15",
    }

    mask_dir_dict = {
        "tm_toa": "ssl4eo_l_tm_{}",
        "etm_toa": "ssl4eo_l_etm_{}",
        "etm_sr": "ssl4eo_l_etm_{}",
        "oli_tirs_toa": "ssl4eo_l_oli_{}",
        "oli_sr": "ssl4eo_l_oli_{}",
    }
    mask_md5s = {
        "tm": {
            "cdl": "3d676770ffb56c7e222a7192a652a846",
            "nlcd": "261149d7614fcfdcb3be368eefa825c7",
        },
        "etm": {
            "cdl": "dd2560b18b89dfe7f0e867fcf7217bd0",
            "nlcd": "916f4a433df6c8abca15b45b60d005d3",
        },
        "oli": {
            "cdl": "1cb057de6eafeca975deb35cb9fb036f",
            "nlcd": "9de0d6d4d0b94313b80450f650813922",
        },
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

    split_percentages = [0.7, 0.15, 0.15]

    ordinal_label_map = {"nlcd": NLCD.ordinal_label_map, "cdl": CDL.ordinal_label_map}

    cmaps = {"nlcd": NLCD.cmap, "cdl": CDL.cmap}

    def __init__(
        self,
        root: str = "data",
        input_sensor: str = "oli_sr",
        mask_product: str = "cdl",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EO Landsat Benchmark instance.

        Args:
            root: root directory where dataset can be found
            input_sensor: one of ['etm_toa', 'etm_sr', 'oli_tirs_toa, 'oli_sr']
            mask_product: mask target one of ['cdl', 'nlcd']
            split: dataset split, one of ['train', 'val', 'test']
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

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
        self.download = download
        self.checksum = checksum
        self.img_dir_name = self.image_root.format(self.input_sensor)
        self.mask_dir_name = self.mask_dir_dict[self.input_sensor].format(
            self.mask_product
        )

        self._verify()

        self.sample_collection = self.retrieve_sample_collection()

        # train, val, test split
        np.random.seed(0)
        sizes = (np.array(self.split_percentages) * len(self.sample_collection)).astype(
            int
        )
        cutoffs = np.cumsum(sizes)[:-1]
        sample_indices = np.arange(len(self.sample_collection))
        np.random.shuffle(sample_indices)
        groups = np.split(sample_indices, cutoffs)
        split_indices = {"train": groups[0], "val": groups[1], "test": groups[2]}[
            self.split
        ]

        self.sample_collection = [self.sample_collection[idx] for idx in split_indices]

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        img_pathname = os.path.join(self.root, self.img_dir_name, "**", "all_bands.tif")
        exists = []
        exists.append(bool(glob.glob(img_pathname, recursive=True)))
        mask_pathname = os.path.join(
            self.root,
            self.mask_dir_name,
            "**",
            f"{self.mask_product}_{self.year_dict[self.input_sensor]}.tif",
        )
        exists.append(bool(glob.glob(mask_pathname, recursive=True)))

        if all(exists):
            return
        # Check if the tar.gz files have already been downloaded
        exists = []
        img_pathname = os.path.join(self.root, f"{self.img_dir_name}.tar.gz")
        exists.append(os.path.exists(img_pathname))

        mask_pathname = os.path.join(self.root, f"{self.mask_dir_name}.tar.gz")
        exists.append(os.path.exists(mask_pathname))

        if all(exists):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        # download imagery
        download_url(
            self.url.format(self.img_dir_name, self.img_dir_name),
            self.root,
            md5=self.img_md5s[self.input_sensor] if self.checksum else None,
        )
        # download mask
        download_url(
            self.url.format(self.mask_dir_name, self.mask_dir_name),
            self.root,
            md5=self.mask_md5s[self.input_sensor.split("_")[0]][self.mask_product]
            if self.checksum
            else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        img_pathname = os.path.join(self.root, f"{self.img_dir_name}.tar.gz")
        extract_archive(img_pathname)

        mask_pathname = os.path.join(self.root, f"{self.mask_dir_name}.tar.gz")
        extract_archive(mask_pathname)

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

    def retrieve_sample_collection(self) -> list[tuple[str, str]]:
        """Retrieve paths to samples in data directory."""
        img_paths = glob.glob(
            os.path.join(self.root, self.img_dir_name, "**", "all_bands.tif"),
            recursive=True,
        )
        img_paths = sorted(img_paths)
        sample_collection: list[tuple[str, str]] = []
        for img_path in img_paths:
            mask_path = img_path.replace(self.img_dir_name, self.mask_dir_name).replace(
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
            mask = src.read()

        for k, v in self.ordinal_label_map[self.mask_product].items():
            mask[mask == k] = v

        return torch.from_numpy(mask).long()

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

        plt_cmap = ListedColormap(
            np.stack(
                [np.array(val) / 255 for val in self.cmaps[self.mask_product].values()],
                axis=0,
            )
        )

        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"].squeeze(0).numpy()
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(mask, cmap=plt_cmap)
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(prediction_mask, cmap=plt_cmap)
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
