# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD+ dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import VisionDataset
from .utils import download_and_extract_archive


class LEVIRCDPlus(VisionDataset):
    """LEVIR-CD+ dataset.

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

    url = "https://drive.google.com/file/d/1JamSsxiytXdzAIk6VDVWfc-OsX-81U81"
    md5 = "1adf156f628aa32fb2e8fe6cada16c04"
    filename = "LEVIR-CD+.zip"
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
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.files = self._load_files(self.root, self.split)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
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

    def _load_files(self, root: str, split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of image1, image2, mask
        """
        files = []
        images = glob.glob(os.path.join(root, split, "A", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "A", image)
            image2 = os.path.join(root, split, "B", image)
            mask = os.path.join(root, split, "label", image)
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
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

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for filename in self.splits:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
