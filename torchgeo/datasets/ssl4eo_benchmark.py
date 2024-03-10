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
from matplotlib.figure import Figure
from torch import Tensor

from .cdl import CDL
from .geo import NonGeoDataset
from .nlcd import NLCD
from .utils import DatasetNotFoundError, download_url, extract_archive


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

    If you use this dataset in your research, please cite the following paper:

    * https://proceedings.neurips.cc/paper_files/paper/2023/hash/bbf7ee04e2aefec136ecf60e346c2e61-Abstract-Datasets_and_Benchmarks.html

    .. versionadded:: 0.5
    """

    url = "https://hf.co/datasets/torchgeo/ssl4eo-l-benchmark/resolve/da96ae2b04cb509710b72fce9131c2a3d5c211c2/{}.tar.gz"  # noqa: E501

    valid_sensors = ["tm_toa", "etm_toa", "etm_sr", "oli_tirs_toa", "oli_sr"]
    valid_products = ["cdl", "nlcd"]
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
            "cdl": "008098c968544049eaf7b307e14241de",
            "nlcd": "9c031049d665202ba42ac1d89b687999",
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

    cmaps = {"nlcd": NLCD.cmap, "cdl": CDL.cmap}

    def __init__(
        self,
        root: str = "data",
        sensor: str = "oli_sr",
        product: str = "cdl",
        split: str = "train",
        classes: Optional[list[int]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EO Landsat Benchmark instance.

        Args:
            root: root directory where dataset can be found
            sensor: one of ['etm_toa', 'etm_sr', 'oli_tirs_toa, 'oli_sr']
            product: mask target, one of ['cdl', 'nlcd']
            split: dataset split, one of ['train', 'val', 'test']
            classes: list of classes to include, the rest will be mapped to 0
                (defaults to all classes for the chosen product)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if any arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert (
            sensor in self.valid_sensors
        ), f"Only supports one of {self.valid_sensors}, but found {sensor}."
        self.sensor = sensor
        assert (
            product in self.valid_products
        ), f"Only supports one of {self.valid_products}, but found {product}."
        self.product = product
        assert (
            split in self.valid_splits
        ), f"Only supports one of {self.valid_splits}, but found {split}."
        self.split = split

        self.cmap = self.cmaps[product]
        if classes is None:
            classes = list(self.cmap.keys())

        assert (
            set(classes) <= self.cmap.keys()
        ), f"Only the following classes are valid: {list(self.cmap.keys())}."
        assert 0 in classes, "Classes must include the background class: 0"

        self.root = root
        self.classes = classes
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=torch.long)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        self.img_dir_name = self.image_root.format(self.sensor)
        self.mask_dir_name = self.mask_dir_dict[self.sensor].format(self.product)

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

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        img_pathname = os.path.join(self.root, self.img_dir_name, "**", "all_bands.tif")
        exists = []
        exists.append(bool(glob.glob(img_pathname, recursive=True)))
        mask_pathname = os.path.join(
            self.root,
            self.mask_dir_name,
            "**",
            f"{self.product}_{self.year_dict[self.sensor]}.tif",
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
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        # download imagery
        download_url(
            self.url.format(self.img_dir_name),
            self.root,
            md5=self.img_md5s[self.sensor] if self.checksum else None,
        )
        # download mask
        download_url(
            self.url.format(self.mask_dir_name),
            self.root,
            md5=(
                self.mask_md5s[self.sensor.split("_")[0]][self.product]
                if self.checksum
                else None
            ),
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
                "all_bands.tif", f"{self.product}_{self.year_dict[self.sensor]}.tif"
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
            image = torch.from_numpy(src.read()).float()
        return image

    def _load_mask(self, path: str) -> Tensor:
        """Load the mask.

        Args:
            path: path to mask

        Retuns:
            mask
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        mask = self.ordinal_map[mask]
        return mask

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
        ncols = 2
        image = sample["image"][self.rgb_indices[self.sensor]].permute(1, 2, 0)
        image = image / 255

        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(self.ordinal_cmap[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
