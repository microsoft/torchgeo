# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from ..datasets.utils import draw_semantic_segmentation_masks
from .geo import VisionDataset
from .utils import download_url, extract_archive, sort_sentinel2_bands


class OSCD(VisionDataset):
    """OSCD dataset.

    The Onera Satellite Change Detection dataset addresses the issue of detecting
    changes between satellite images from different dates.

    Dataset features:

    * 24 pairs of images from all over the world between 2015-2018
    * binary change masks representing urban changes such as new buildings or roads
    * 13 spectral bands
    * Images vary in spatial resolution between 10m, 20m, and 60m

    Dataset format:

    * images are 13-channel tifs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    1. no change
    2. change

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1810.08468
    """

    url = "https://drive.google.com/file/d/1jidN0DKEIybOrP0j7Bos8bGDDq3Varj3"
    md5 = "7383412da7ece1dca1c12dc92ac77f09"

    zipfile_glob = "*OSCD.zip"
    zipfile_glob2 = "*Onera*.zip"
    filename_glob = "*Onera*"
    filename = "OSCD.zip"
    splits = ["train", "test"]

    colormap = ["blue"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
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

        self.root = root
        self.split = split
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
        # TODO: implement choosing bands (right now assuming bands="all")
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
        temp_split = "Test" if self.split == "test" else "Train"
        labels_root = os.path.join(
            self.root, f"Onera Satellite Change Detection dataset - {temp_split} Labels"
        )
        images_root = os.path.join(
            self.root, "Onera Satellite Change Detection dataset - Images"
        )
        folders = glob.glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")
            images1 = glob.glob(
                os.path.join(images_root, region, "imgs_1_rect", "*.tif")
            )
            images2 = glob.glob(
                os.path.join(images_root, region, "imgs_2_rect", "*.tif")
            )
            images1 = sorted(images1, key=sort_sentinel2_bands)
            images2 = sorted(images2, key=sort_sentinel2_bands)
            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple(
                    [line.split()[-1] for line in f.read().strip().splitlines()]
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
        images = []
        for path in paths:
            with rasterio.open(path) as f:
                images.append(f.read())
        np_images = np.stack(images, axis=0).astype(np.int_).squeeze()
        tensor: Tensor = torch.from_numpy(np_images)  # type: ignore[attr-defined]
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
            array = np.array(img.convert("L"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = torch.clamp(tensor, min=0, max=1)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
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
                "to automaticaly download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)
        pathname = os.path.join(self.root, self.zipfile_glob2)
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

        rgb_img1 = sample["image"][0, 1:4].float().numpy()
        per02 = np.percentile(rgb_img1, 2)  # type: ignore[no-untyped-call]
        per98 = np.percentile(rgb_img1, 98)  # type: ignore[no-untyped-call]
        rgb_img1 = (np.clip((rgb_img1 - per02) / (per98 - per02), 0, 1) * 255).astype(
            np.uint8
        )

        rgb_img2 = sample["image"][1, 1:4].float().numpy()
        per02 = np.percentile(rgb_img2, 2)  # type: ignore[no-untyped-call]
        per98 = np.percentile(rgb_img2, 98)  # type: ignore[no-untyped-call]
        rgb_img2 = (np.clip((rgb_img2 - per02) / (per98 - per02), 0, 1) * 255).astype(
            np.uint8
        )

        image1 = draw_semantic_segmentation_masks(
            torch.from_numpy(rgb_img1),  # type: ignore[attr-defined]
            sample["mask"],
            alpha=alpha,
            colors=self.colormap,  # type: ignore[arg-type]
        )
        image2 = draw_semantic_segmentation_masks(
            torch.from_numpy(rgb_img2),  # type: ignore[attr-defined]
            sample["mask"],
            alpha=alpha,
            colors=self.colormap,  # type: ignore[arg-type]
        )

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
