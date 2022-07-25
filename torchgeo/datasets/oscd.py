# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    download_url,
    draw_semantic_segmentation_masks,
    extract_archive,
    sort_sentinel2_bands,
)


class OSCD(NonGeoDataset):
    """OSCD dataset.

    The `Onera Satellite Change Detection <https://rcdaudt.github.io/oscd/>`_
    dataset addresses the issue of detecting changes between
    satellite images from different dates. Imagery comes from
    Sentinel-2 which contains varying resolutions per band.

    Dataset format:

    * images are 13-channel tifs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    0. no change
    1. change

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2018.8518015

    .. versionadded:: 0.2
    """

    urls = {
        "Onera Satellite Change Detection dataset - Images.zip": (
            "https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download"
        ),
        "Onera Satellite Change Detection dataset - Train Labels.zip": (
            "https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download"
        ),
        "Onera Satellite Change Detection dataset - Test Labels.zip": (
            "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download"
        ),
    }
    md5s = {
        "Onera Satellite Change Detection dataset - Images.zip": (
            "c50d4a2941da64e03a47ac4dec63d915"
        ),
        "Onera Satellite Change Detection dataset - Train Labels.zip": (
            "4d2965af8170c705ebad3d6ee71b6990"
        ),
        "Onera Satellite Change Detection dataset - Test Labels.zip": (
            "8177d437793c522653c442aa4e66c617"
        ),
    }

    zipfile_glob = "*Onera*.zip"
    filename_glob = "*Onera*"
    splits = ["train", "test"]

    colormap = ["blue"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new OSCD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in self.splits
        assert bands in ["rgb", "all"]

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files["images1"])
        image2 = self._load_image(files["images2"])
        mask = self._load_target(str(files["mask"]))

        image = torch.stack(tensors=[image1, image2], dim=0)
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

    def _load_files(self) -> List[Dict[str, Union[str, Sequence[str]]]]:
        regions = []
        labels_root = os.path.join(
            self.root,
            f"Onera Satellite Change Detection dataset - {self.split.capitalize()} "
            + "Labels",
        )
        images_root = os.path.join(
            self.root, "Onera Satellite Change Detection dataset - Images"
        )
        folders = glob.glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")

            def get_image_paths(ind: int) -> List[str]:
                return sorted(
                    glob.glob(
                        os.path.join(images_root, region, f"imgs_{ind}_rect", "*.tif")
                    ),
                    key=sort_sentinel2_bands,
                )

            images1, images2 = get_image_paths(1), get_image_paths(2)
            if self.bands == "rgb":
                images1, images2 = images1[1:4][::-1], images2[1:4][::-1]

            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple(
                    line.split()[-1] for line in f.read().strip().splitlines()
                )

            regions.append(
                dict(
                    region=region,
                    images1=images1,
                    images2=images2,
                    mask=mask,
                    dates=dates,
                )
            )

        return regions

    def _load_image(self, paths: Sequence[str]) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        images: List["np.typing.NDArray[np.int_]"] = []
        for path in paths:
            with Image.open(path) as img:
                images.append(np.array(img))
        array: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0).astype(np.int_)
        tensor = torch.from_numpy(array)
        return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            tensor = torch.clamp(tensor, min=0, max=1)
            tensor = tensor.to(torch.long)
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            if not fname.endswith(".zip"):
                return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
        if glob.glob(pathname):
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
        for f_name in self.urls:
            download_url(
                self.urls[f_name],
                self.root,
                filename=f_name,
                md5=self.md5s[f_name] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2

        rgb_inds = [3, 2, 1] if self.bands == "all" else [0, 1, 2]

        def get_masked(img: Tensor) -> "np.typing.NDArray[np.uint8]":
            rgb_img = img[rgb_inds].float().numpy()
            per02 = np.percentile(rgb_img, 2)
            per98 = np.percentile(rgb_img, 98)
            rgb_img = (np.clip((rgb_img - per02) / (per98 - per02), 0, 1) * 255).astype(
                np.uint8
            )
            array: "np.typing.NDArray[np.uint8]" = draw_semantic_segmentation_masks(
                torch.from_numpy(rgb_img),
                sample["mask"],
                alpha=alpha,
                colors=self.colormap,
            )
            return array

        image1, image2 = get_masked(sample["image"][0]), get_masked(sample["image"][1])
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Pre change")
            axs[1].set_title("Post change")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
