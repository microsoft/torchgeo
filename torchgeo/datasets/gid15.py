# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""GID-15 dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_and_extract_archive


class GID15(NonGeoDataset):
    """GID-15 dataset.

    The `GID-15 <https://captain-whu.github.io/GID15/>`__
    dataset is a dataset for semantic segmentation.

    Dataset features:

    * images taken by the Gaofen-2 (GF-2) satellite over 60 cities in China
    * masks representing 15 semantic categories
    * three spectral bands - RGB
    * 150 with 3 m per pixel resolution (6800x7200 px)

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs
    * colormapped masks are 3 channel tifs

    Dataset classes:

    1. background
    2. industrial_land
    3. urban_residential
    4. rural_residential
    5. traffic_land
    6. paddy_field
    7. irrigated_land
    8. dry_cropland
    9. garden_plot
    10. arbor_woodland
    11. shrub_land
    12. natural_grassland
    13. artificial_grassland
    14. river
    15. lake
    16. pond

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2019.111322
    """

    url = "https://drive.google.com/file/d/1zbkCEXPEKEV6gq19OKmIbaT8bXXfWW6u"
    md5 = "615682bf659c3ed981826c6122c10c83"
    filename = "gid-15.zip"
    directory = "GID"
    splits = ["train", "val", "test"]
    classes = [
        "background",
        "industrial_land",
        "urban_residential",
        "rural_residential",
        "traffic_land",
        "paddy_field",
        "irrigated_land",
        "dry_cropland",
        "garden_plot",
        "arbor_woodland",
        "shrub_land",
        "natural_grassland",
        "artificial_grassland",
        "river",
        "lake",
        "pond",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new GID-15 dataset instance.

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
        image = self._load_image(files["image"])

        if self.split != "test":
            mask = self._load_target(files["mask"])
            sample = {"image": image, "mask": mask}
        else:
            sample = {"image": image}

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
            split: subset of dataset, one of [train, val, test]

        Returns:
            list of dicts containing paths for each pair of image, mask
        """
        image_root = os.path.join(root, "GID", "img_dir")
        images = glob.glob(os.path.join(image_root, split, "*.tif"))
        images = sorted(images)
        if split != "test":
            masks = [
                image.replace("img_dir", "ann_dir").replace(".tif", "_15label.png")
                for image in images
            ]
            files = [dict(image=image, mask=mask) for image, mask in zip(images, masks)]
        else:
            files = [dict(image=image) for image in images]

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
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
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
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        filepath = os.path.join(self.root, self.directory)
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

    def plot(
        self, sample: Dict[str, Tensor], suptitle: Optional[str] = None
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns;
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        if self.split != "test":
            image, mask = sample["image"], sample["mask"]
            ncols = 2
        else:
            image = sample["image"]
            ncols = 1

        if "prediction" in sample:
            ncols += 1
            pred = sample["prediction"]

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        if self.split != "test":
            axs[0].imshow(image.permute(1, 2, 0))
            axs[0].axis("off")
            axs[1].imshow(mask)
            axs[1].axis("off")
            if "prediction" in sample:
                axs[2].imshow(pred)
                axs[2].axis("off")
        else:
            if "prediction" in sample:
                axs[0].imshow(image.permute(1, 2, 0))
                axs[0].axis("off")
                axs[1].imshow(pred)
                axs[1].axis("off")
            else:
                axs.imshow(image.permute(1, 2, 0))
                axs.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
