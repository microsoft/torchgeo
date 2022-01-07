# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Inria Aerial Image Labeling Dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.geo import VisionDataset
from torchgeo.datasets.utils import (
    check_integrity,
    extract_archive,
    percentile_normalization,
)


class InriaBuildings(VisionDataset):
    r"""Inria Aerial Image Labeling Dataset.

    The `Inria Aerial Image Labeling
    <https://project.inria.fr/aerialimagelabeling/>`_ dataset is a building
    detection dataset over dissimilar settlements ranging ranging from densely
    populated areas to alpine towns.

    Dataset features:

    * Coverage of 810 km\ :sup:`2`\  (405 km\ :sup:`2`\  for training and 405
      km\ :sup:`2`\  for testing)
    * Aerial orthorectified color imagery with a spatial resolution of 0.3 m
    * Train cities: Austin, Chicago, Kitsap, West Tyrol, Vienna
    * Test cities: Bellingham, Bloomington, Innsbruck, San Francisco, East Tyrol

    Dataset format:

    * Imagery - Aerial GeoTIFFs
    * Labels - Aerial GeoTIFFs

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2017.8127684


    .. versionadded:: 0.3
    """

    foldername = "AerialImageDataset"
    archive_name = "NEW2-AerialImageDataset.zip"
    md5 = "4b1acfe84ae9961edc1a6049f940380f"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new InriaBuildings Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/test split
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if dataset is missing
        """
        self.root = root
        assert split in {"train", "test"}
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._check_integrity()
        self.files = self._load_files(root)

    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        root_dir = os.path.join(root, self.foldername, self.split)
        images = glob.glob(os.path.join(root_dir, "images", "*.tif"))
        images = sorted(images)
        if self.split == "train":
            labels = glob.glob(os.path.join(root_dir, "gt", "*.tif"))
            labels = sorted(labels)

            for img, lbl in zip(images, labels):
                files.append({"image_path": img, "label_path": lbl})
        else:
            for img in images:
                files.append({"image_path": img})

        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with rio.open(filename) as img:
            array = img.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_label(self, path: str) -> Tensor:
        """Loads a single label.

        Args:
            path: path to the label

        Returns:
            the label
        """
        filename = os.path.join(path)
        with rio.open(filename) as img:
            array = img.read().astype(np.int32)
            array = array // 255
            mask: Tensor = torch.from_numpy(array[0])  # type: ignore[attr-defined]
            return mask

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        sample = {}
        img = self._load_image(files["image_path"])
        sample["image"] = img
        if files.get("label_path"):
            mask = self._load_label(files["label_path"])
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> None:
        """Checks the integrity of the dataset structure."""
        if os.path.isdir(os.path.join(self.root, self.foldername)):
            return

        archive_path = os.path.join(self.root, self.archive_name)
        md5_hash = self.md5 if self.checksum else None
        if check_integrity(archive_path, md5_hash):
            print("Extracting...")
            extract_archive(archive_path)

    def plot(
        self,
        sample: Dict[str, Tensor],
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
        image = np.rollaxis(sample["image"][:3].numpy(), 0, 3)
        image = percentile_normalization(image, axis=(0, 1))

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = sample["mask"].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title("Image")

        if show_mask:
            axs[1].imshow(mask, interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[1].set_title("Label")

        if show_predictions:
            axs[2].imshow(prediction, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
