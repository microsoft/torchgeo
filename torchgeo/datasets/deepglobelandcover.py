# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DeepGlobe Land Cover Classification Challenge dataset."""

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


class DeepGlobeLandCover(NonGeoDataset):
    """DeepGlobe Land Cover Classification Challenge dataset.

    The `DeepGlobe Land Cover Classification Challenge
    <https://competitions.codalab.org/competitions/18468>`__ dataset
    offers high-resolution sub-meter satellite imagery focusing for the task of
    semantic segmentation to detect areas of urban, agriculture, rangeland, forest,
    water, barren, and unknown. It contains 1,146 satellite images of size
    2448 x 2448 pixels in total, split into training/validation/test sets, the original
    dataset can be downloaded from `Kaggle <https://www.kaggle.com/datasets/balraj98/
    deepglobe-land-cover-classification-dataset>`__.
    However, we only use the training dataset with 803 images since the original test
    and valid dataset are not accompanied by labels. The dataset that we use with a
    custom train/test split can be downloaded from `Kaggle <https://www.kaggle.com/
    datasets/geoap96/deepglobe2018-landcover-segmentation-traindataset>`__ (created as a
    part of Computer Vision by Deep Learning (CS4245) course offered at TU Delft).

    Dataset format:

    * images are RGB data
    * masks are RGB image with with unique RGB values representing the class

    Dataset classes:

    0. Urban land
    1. Agriculture land
    2. Rangeland
    3. Forest land
    4. Water
    5. Barren land
    6. Unknown

    File names for satellite images and the corresponding mask image are id_sat.jpg and
    id_mask.png, where id is an integer assigned to every image.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/pdf/1805.06561.pdf

    .. versionadded:: 0.3
    """

    filename = "data.zip"
    data_root = "data"
    md5 = "f32684b0b2bf6f8d604cd359a399c061"
    splits = ["train", "test"]
    classes = [
        "Urban land",
        "Agriculture land",
        "Rangeland",
        "Forest land",
        "Water",
        "Barren land",
        "Unknown",
    ]
    colormap = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DeepGlobeLandCover dataset instance.

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
        if split == "train":
            split_folder = "training_data"
        else:
            split_folder = "test_data"

        self.image_fns = []
        self.mask_fns = []
        for image in sorted(
            os.listdir(os.path.join(root, self.data_root, split_folder, "images"))
        ):
            if image.endswith(".jpg"):
                id = image[:-8]
                image_path = os.path.join(
                    root, self.data_root, split_folder, "images", image
                )
                mask_path = os.path.join(
                    root, self.data_root, split_folder, "masks", str(id) + "_mask.png"
                )

                self.image_fns.append(image_path)
                self.mask_fns.append(mask_path)

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
        return len(self.image_fns)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.image_fns[index]

        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
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
        path = self.mask_fns[index]
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.uint8]" = np.array(img)
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
        if os.path.exists(os.path.join(self.root, self.data_root)):
            return

        # Check if .zip file already exists (if so extract)
        filepath = os.path.join(self.root, self.filename)

        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root`, either specify a different"
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
            sample["image"], sample["mask"], alpha=alpha, colors=self.colormap
        )
        if "prediction" in sample:
            ncols += 1
            image2 = draw_semantic_segmentation_masks(
                sample["image"], sample["prediction"], alpha=alpha, colors=self.colormap
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
