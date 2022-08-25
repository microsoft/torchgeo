# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai dataset."""

import glob
import hashlib
import os
from functools import lru_cache
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive, working_dir


class LandCoverAI(NonGeoDataset):
    r"""LandCover.ai dataset.

    The `LandCover.ai <https://landcover.ai/>`__ (Land Cover from Aerial Imagery)
    dataset is a dataset for automatic mapping of buildings, woodlands, water and
    roads from aerial images. This implementation is specifically for Version 1 of
    Landcover.ai.

    Dataset features:

    * land cover from Poland, Central Europe
    * three spectral bands - RGB
    * 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
    * 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
    * total area of 216.27 km\ :sup:`2`

    Dataset format:

    * rasters are three-channel GeoTiffs with EPSG:2180 spatial reference system
    * masks are single-channel GeoTiffs with EPSG:2180 spatial reference system

    Dataset classes:

    1. building (1.85 km\ :sup:`2`\ )
    2. woodland (72.02 km\ :sup:`2`\ )
    3. water (13.15 km\ :sup:`2`\ )
    4. road (3.5 km\ :sup:`2`\ )

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2005.02264v3

    .. note::

       This dataset requires the following additional library to be installed:

       * `opencv-python <https://pypi.org/project/opencv-python/>`_ to generate
         the train/val/test split
    """

    url = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
    filename = "landcover.ai.v1.zip"
    md5 = "3268c89070e8734b4e91d531c0617e03"
    sha256 = "15ee4ca9e3fd187957addfa8f0d74ac31bc928a966f76926e11b3c33ea76daa1"
    classes = ["Background", "Building", "Woodland", "Water", "Road"]
    cmap = ListedColormap(
        [
            [0.63921569, 1.0, 0.45098039],
            [0.61176471, 0.61176471, 0.61176471],
            [0.14901961, 0.45098039, 0.0],
            [0.0, 0.77254902, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["train", "val", "test"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        with open(os.path.join(self.root, split + ".txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = self.ids[index].rstrip()
        sample = {"image": self._load_image(id_), "mask": self._load_target(id_)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    @lru_cache()
    def _load_image(self, id_: str) -> Tensor:
        """Load a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = os.path.join(self.root, "output", id_ + ".jpg")
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    @lru_cache()
    def _load_target(self, id_: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the target mask
        """
        filename = os.path.join(self.root, "output", id_ + "_m.png")
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        jpg = os.path.join(self.root, "output", "*_*.jpg")
        png = os.path.join(self.root, "output", "*_*_m.png")
        if glob.glob(jpg) and glob.glob(png):
            return

        # Check if the zip file has already been downloaded
        pathname = os.path.join(self.root, self.filename)
        if os.path.exists(pathname):
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
        download_url(self.url, self.root, md5=self.md5 if self.checksum else None)

    def _extract(self) -> None:
        """Extract the dataset.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        extract_archive(os.path.join(self.root, self.filename))

        # Generate train/val/test splits
        # Always check the sha256 of this file before executing
        # to avoid malicious code injection
        with working_dir(self.root):
            with open("split.py") as f:
                split = f.read().encode("utf-8")
                assert hashlib.sha256(split).hexdigest() == self.sha256
                exec(split)

    def plot(
        self,
        sample: Dict[str, Tensor],
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

        .. versionadded:: 0.2
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        mask = sample["mask"].numpy()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=4, cmap=self.cmap, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=4, cmap=self.cmap, interpolation="none"
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
