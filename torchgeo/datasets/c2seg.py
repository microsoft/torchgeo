# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""C2Seg dataset."""

import glob
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive


class C2Seg(NonGeoDataset):
    """Cross-City Multimodal Semantic Segmentation Challenge (C2Seg) dataset.

    The `C2Seg <https://www.ieee-whispers.com/cross-city-challenge/>`_
    dataset for land cover land use (LULC) semantic segmentation is hosted as
    the 2023 IEEE WHISPERS conference challenge. The dataset can be downloaded
    from google drive the following `link
    <https://drive.google.com/drive/folders/1b5AYbOMQaE4Vz8XnZcH0N6v_Z-9LrrRX?usp=sharing`_.

    Dataset features:
    * Sentinel-2 or Gaofen-6 MSI imagery (4 bands)
    * Sentinel-1 or Gaofen-3 SAR imagery (2 bands)
    * EnMAP or Gaofen-6 HSI imagery (242 or 116 bands)
    * 10 m per pixel resolution (128x128 or 256x256 px)
    * LULC semantic masks with 14 categories

    Dataset format:
    * msi, sar, and hsi images are in geotiff format
    * masks are in geotiff format

    Dataset classes:
    0. Background
    1. Surface water
    2. Street
    3. Urban Fabric
    4. Industrial, commercial and transport
    5. Mine, dump, and construction sites
    6. Artificial, vegetated areas
    7. Arable Land
    8. Permanent Crops
    9. Pastures
    10. Forests
    11. Shrub
    12. Open spaces with no vegetation
    13. Inland wetlands

    .. versionadded:: 0.5
    """

    cmap = {
        0: (0, 0, 0, 255),
        1: (70, 107, 159, 255),
        2: (209, 222, 248, 255),
        3: (222, 197, 197, 255),
        4: (217, 146, 130, 255),
        5: (235, 0, 0, 255),
        6: (171, 0, 0, 255),
        7: (179, 172, 159, 255),
        8: (104, 171, 95, 255),
        9: (28, 95, 44, 255),
        10: (181, 197, 143, 255),
        11: (204, 184, 121, 255),
        12: (223, 223, 194, 255),
        13: (220, 217, 57, 255),
        14: (171, 108, 40, 255),
    }

    all_bands = ["msi", "sar", "hsi"]

    classes = [
        "Background",
        "Surface water",
        "Street",
        "Urban Fabric",
        "Industrial, commercial and transport",
        "Mine, dump, and construction sites",
        "Artificial, vegetated areas",
        "Arable Land",
        "Permanent Crops",
        "Pastures",
        "Forests",
        "Shrub",
        "Open spaces with no vegetation",
        "Inland wetlands",
    ]

    subsets = {
        "C2Seg_AB": {
            "directory": "C2Seg_AB",
            "filename": "C2Seg_AB.zip",
            "md5": "1a7eddf6b90b875ba1ab4cc173818296",
        },
        "C2Seg_BW": {
            "directory": "C2Seg_BW",
            "filename": "C2Seg_BW.zip",
            "md5": "4728fa97747fad55de8c55fcbef8be60",
        },
    }

    def __init__(
        self,
        root: str = "data",
        subset: str = "C2Seg_AB",
        split: str = "train",
        band_set: list[str] = all_bands,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new C2Seg dataset instance.

        Args:
            root: root directory where dataset can be found
            subset: subset of the dataset for training or testing
                one of C2Seg_AB or C2Seg_AB
            split: one of "train" or "val"
            band_set: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``subset``, ``split``, ``bands`` argument are invalid
            RuntimeError: if ``checksum=True`` and checksums don't match
        """
        assert split in ["train", "test"]
        assert subset in self.subsets
        for bands in band_set:
            assert bands in self.all_bands

        self.root = root
        self.subset = subset
        self.split = split
        self.band_set = band_set
        self.transforms = transforms
        self.checksum = checksum

        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample containing image and mask
        """
        paths = self.files[index]

        image_list = []
        for bands in self.band_set:
            image_list.append(self._load_image(paths[bands]))

        image = torch.cat(image_list, dim=0)
        sample = {"image": image}

        if self.split == "train":
            mask = self._load_target(paths["mask"])

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str]]:
        """Return the file paths for the given subset and split.

        Returns:
            the file paths
        """
        root = os.path.join(
            self.root, self.subsets[self.subset]["directory"], self.split
        )
        filepaths = {}
        for bands in self.all_bands:
            paths = sorted(glob.glob(os.path.join(root, bands, "*.tiff")))
            filepaths[bands] = paths

        if self.split == "train":
            paths = sorted(glob.glob(os.path.join(root, "label", "*.tiff")))
            filepaths["mask"] = paths

        files = [dict(zip(filepaths, i)) for i in zip(*filepaths.values())]
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as img:
            array = img.read().astype(np.float32)
            tensor = torch.from_numpy(array).float()
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Loads a single target mask.

        Args:
            path: path to the mask

        Returns:
            the target mask
        """
        with rasterio.open(path) as img:
            array = img.read().astype(np.int32)
            mask = torch.from_numpy(array).squeeze(dim=0).long()
            return mask

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails
        """
        # Check if the files already exist
        directory = os.path.join(self.root, self.subsets[self.subset]["directory"])
        if os.path.exists(directory):
            return

        # Check if the zip files have already been downloaded
        filepath = os.path.join(self.root, self.subsets[self.subset]["filename"])
        if os.path.exists(filepath):
            md5 = self.subsets[self.subset]["md5"]
            if self.checksum and not check_integrity(filepath, md5):
                raise RuntimeError("Dataset found, but corrupted.")
            self._extract()
            return

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` directory "
            "Specify a different `root` directory."
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.subsets[self.subset]["filename"])
        extract_archive(filepath, self.root)
        if self.subset == "C2Seg_BW":
            directory = os.path.join(
                self.root, self.subsets[self.subset]["directory"], "train"
            )
            zipfiles = glob.glob(os.path.join(directory, "*.zip"))
            for zipfile in zipfiles:
                extract_archive(zipfile, directory)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        hsi_indices: tuple[int, int, int] = (0, 1, 2),
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            hsi_indices: tuple of indices to create HSI false color image

        Returns:
            a matplotlib Figure with the rendered sample
        """
        assert len(hsi_indices) == 3

        def normalize(x: Tensor) -> Tensor:
            return (x - x.min()) / (x.max() - x.min())

        plt_cmap = ListedColormap(
            np.stack([np.array(val) / 255 for val in self.cmap.values()], axis=0)
        )

        ncols = len(self.band_set)
        showing_msi = "msi" in self.band_set
        showing_sar = "sar" in self.band_set
        showing_hsi = "hsi" in self.band_set
        showing_mask = "mask" in sample
        showing_predictions = "prediction" in sample

        if showing_mask:
            mask = sample["mask"].squeeze().numpy()
            ncols += 1
        if showing_predictions:
            preds = sample["prediction"].squeeze().numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))

        i = 0
        if showing_msi:
            rgb = normalize(sample["image"][:3]).permute((1, 2, 0)).numpy()
            rgb = normalize(rgb)
            axs[i].imshow(rgb)
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title("RGB Image")
            i += 1
        if showing_sar:
            sar = normalize(sample["image"][4:6])
            sar = torch.cat([sar, (sar[0] / sar[1]).unsqueeze(dim=0)])
            sar = sar.permute((1, 2, 0)).numpy()
            axs[i].imshow(sar)
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title("SAR False Color Image")
            i += 1
        if showing_hsi:
            hsi = normalize(sample["image"][6:][hsi_indices, ...])
            hsi = hsi.permute((1, 2, 0)).numpy()
            axs[i].imshow(hsi)
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title("Hyperspectral False Color Image")
            i += 1

        if showing_mask:
            axs[i].imshow(mask, cmap=plt_cmap)
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title("Ground Truth")
            i += 1
        if showing_predictions:
            axs[i].imshow(preds, cmap=plt_cmap)
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title("Predictions")
            i += 1

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
