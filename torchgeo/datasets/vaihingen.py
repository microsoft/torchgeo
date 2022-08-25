# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Vaihingen dataset."""

import os
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
    rgb_to_mask,
)


class Vaihingen2D(NonGeoDataset):
    """Vaihingen 2D Semantic Segmentation dataset.

    The `Vaihingen <https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/>`__
    dataset is a dataset for urban semantic segmentation used in the 2D Semantic Labeling
    Contest - Vaihingen. This dataset uses the "ISPRS_semantic_labeling_Vaihingen.zip"
    and "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip" files to create the
    train/test sets used in the challenge. The dataset can be requested at the challenge
    homepage. Note, the server contains additional data for 3D Semantic Labeling which
    are currently not supported.

    Dataset format:

    * images are 3-channel RGB geotiffs
    * masks are 3-channel geotiffs with unique RGB values representing the class

    Dataset classes:

    0. Clutter/background
    1. Impervious surfaces
    2. Building
    3. Low Vegetation
    4. Tree
    5. Car

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/isprsannals-I-3-293-2012

    .. versionadded:: 0.2
    """  # noqa: E501

    filenames = [
        "ISPRS_semantic_labeling_Vaihingen.zip",
        "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip",
    ]
    md5s = ["462b8dca7b6fa9eaf729840f0cdfc7f3", "4802dd6326e2727a352fb735be450277"]
    image_root = "top"
    splits = {
        "train": [
            "top_mosaic_09cm_area1.tif",
            "top_mosaic_09cm_area11.tif",
            "top_mosaic_09cm_area13.tif",
            "top_mosaic_09cm_area15.tif",
            "top_mosaic_09cm_area17.tif",
            "top_mosaic_09cm_area21.tif",
            "top_mosaic_09cm_area23.tif",
            "top_mosaic_09cm_area26.tif",
            "top_mosaic_09cm_area28.tif",
            "top_mosaic_09cm_area3.tif",
            "top_mosaic_09cm_area30.tif",
            "top_mosaic_09cm_area32.tif",
            "top_mosaic_09cm_area34.tif",
            "top_mosaic_09cm_area37.tif",
            "top_mosaic_09cm_area5.tif",
            "top_mosaic_09cm_area7.tif",
        ],
        "test": [
            "top_mosaic_09cm_area6.tif",
            "top_mosaic_09cm_area24.tif",
            "top_mosaic_09cm_area35.tif",
            "top_mosaic_09cm_area16.tif",
            "top_mosaic_09cm_area14.tif",
            "top_mosaic_09cm_area22.tif",
            "top_mosaic_09cm_area10.tif",
            "top_mosaic_09cm_area4.tif",
            "top_mosaic_09cm_area2.tif",
            "top_mosaic_09cm_area20.tif",
            "top_mosaic_09cm_area8.tif",
            "top_mosaic_09cm_area31.tif",
            "top_mosaic_09cm_area33.tif",
            "top_mosaic_09cm_area27.tif",
            "top_mosaic_09cm_area38.tif",
            "top_mosaic_09cm_area12.tif",
            "top_mosaic_09cm_area29.tif",
        ],
    }
    classes = [
        "Clutter/background",
        "Impervious surfaces",
        "Building",
        "Low Vegetation",
        "Tree",
        "Car",
    ]
    colormap = [
        (255, 0, 0),
        (255, 255, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Vaihingen2D dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.files = []
        for name in self.splits[split]:
            image = os.path.join(root, self.image_root, name)
            mask = os.path.join(root, name)
            if os.path.exists(image) and os.path.exists(mask):
                self.files.append(dict(image=image, mask=mask))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        mask = self._load_target(index)
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

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["image"]
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        path = self.files[index]["mask"]
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.uint8]" = np.array(img.convert("RGB"))
            array = rgb_to_mask(array, self.colormap)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.to(torch.long)
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.image_root)):
            return

        # Check if .zip files already exists (if so extract)
        exists = []
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

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
        ncols = 1
        image1 = draw_semantic_segmentation_masks(
            sample["image"][:3], sample["mask"], alpha=alpha, colors=self.colormap
        )
        if "prediction" in sample:
            ncols += 1
            image2 = draw_semantic_segmentation_masks(
                sample["image"][:3],
                sample["prediction"],
                alpha=alpha,
                colors=self.colormap,
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols > 1:
            (ax0, ax1) = axs
        else:
            ax0 = axs

        ax0.imshow(image1)
        ax0.axis("off")
        if ncols > 1:
            ax1.imshow(image2)
            ax1.axis("off")

        if show_titles:
            ax0.set_title("Ground Truth")
            if ncols > 1:
                ax1.set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
