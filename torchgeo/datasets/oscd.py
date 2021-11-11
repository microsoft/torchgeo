# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""

import os
import glob
import tifffile
import rasterio
from typing import Callable, Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url, extract_archive, sort_sentinel2_bands


class OSCD(VisionDataset):
    # TODO: update this to OSCD
    """OSCD dataset.

    The `LEVIR-CD+ <https://github.com/S2Looking/Dataset>`_
    dataset is a dataset for building change detection.

    Dataset features:

    * image pairs of 20 different urban regions across Texas between 2002-2020
    * binary change masks representing building change
    * three spectral bands - RGB
    * 985 image pairs with 50 cm per pixel resolution (~1024x1024 px)

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    1. no change
    2. change

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2107.09244
    """

    url = "https://drive.google.com/file/d/1jidN0DKEIybOrP0j7Bos8bGDDq3Varj3"
    md5 = "1adf156f628aa32fb2e8fe6cada16c04" # TODO: find this

    # TODO: find better way to solve nested zip file structure
    zipfile_glob = "*OSCD.zip"
    zipfile_glob2 = "*Onera*.zip"
    # TODO: need to change filename_glob due to how this is checked in verify
    filename_glob = "*Onera*"
    filename = "OSCD.zip"
    splits = ["train", "test"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LEVIR-CD+ dataset instance.

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
        mask = self._load_target(files["mask"])

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

    # TODO: this needs to be refactored 
    def _load_files(self) -> List[Dict[str, str]]:
        regions = []
        temp_split = "Test" if self.split == "test" else "Train"
        labels_root = os.path.join(self.root, f"Onera Satellite Change Detection dataset - {temp_split} Labels")
        images_root = os.path.join(self.root, "Onera Satellite Change Detection dataset - Images")
        folders = glob.glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")
            images1 = glob.glob(os.path.join(images_root, region, "imgs_1_rect", "*.tif"))
            images2 = glob.glob(os.path.join(images_root, region, "imgs_2_rect", "*.tif"))
            images1 = sorted(images1, key=sort_sentinel2_bands)
            images2 = sorted(images2, key=sort_sentinel2_bands)
            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple([line.split()[-1] for line in f.read().strip().splitlines()])

            regions.append(dict(region=region, images1=images1, images2=images2, mask=mask, dates=dates))

        return regions

    def _load_image(self, paths: List[str]) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """

        # images = np.stack([tifffile.imread(path) for path in paths], axis=0)
        images = np.stack([rasterio.open(path).read() for path in paths], axis=0)
        images = images.astype(np.long)
        return torch.from_numpy(images)


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
                md5=md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)
        # TODO: nicer way to solve this nested zipfile structure
        pathname = os.path.join(self.root, self.zipfile_glob2)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)
