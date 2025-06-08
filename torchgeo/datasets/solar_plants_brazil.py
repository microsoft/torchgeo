# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SolarPlantsBrazil dataset."""

import glob
import os
from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_and_extract_archive


class SolarPlantsBrazil(NonGeoDataset):
    """Solar Plants Brazil dataset (semantic segmentation for photovoltaic detection).

    The `Solar Plants Brazil <https://huggingface.co/datasets/FederCO23/solar-plants-brazil>`__
    dataset provides satellite imagery and pixel-level annotations for detecting photovoltaic
    solar power stations.

    Dataset features:

    * 272 RGB+NIR GeoTIFF images (256x256 pixels)
    * Binary masks indicating presence of solar panels (1 = panel, 0 = background)
    * Organized into `train`, `val`, and `test` splits
    * Float32 GeoTIFF files for both input and mask images
    * Spatial metadata included (CRS, bounding box), but not used directly for training

    Folder structure:

    .. code-block:: text

        root/train/input/img(123).tif
        root/train/labels/target(123).tif

    Access:

    * Dataset is hosted on Hugging Face: https://huggingface.co/datasets/FederCO23/solar-plants-brazil
    * Code and preprocessing steps available at: https://github.com/FederCO23/UCSD_MLBootcamp_Capstone

    .. versionadded:: 0.8

    """

    url = 'https://huggingface.co/datasets/FederCO23/solar-plants-brazil/resolve/main/solarplantsbrazil.zip'
    bands = ('Red', 'Green', 'Blue', 'NIR')

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a SolarPlantsBrazil dataset split.

        Args:
            root: Root directory where dataset is stored.
            split: Dataset split to use, one of "train", "val", or "test".
            transforms: Optional transforms to apply.
            download: If True, download the dataset if it doesn't exist.

        Raises:
            DatasetNotFoundError: If the dataset is not found and
                ``download=False``.

        Returns:
            None

        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(
                f"Invalid split '{split}', expected one of: 'train', 'val', or 'test'"
            )

        self.root = root
        self.transforms = transforms
        self.dataset_path = os.path.join(self.root, split)
        self.split = split
        self.download = download

        self._verify()

        self.image_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, 'input', 'img(*).tif'))
        )
        self.mask_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, 'labels', 'target(*).tif'))
        )

        if len(self.image_paths) == 0:
            raise DatasetNotFoundError(self)

        assert len(self.image_paths) == len(self.mask_paths), (
            'Mismatch between image and mask files'
        )

    def _verify(self) -> None:
        """Verify the dataset exists or download it.

        Returns:
            None
        """
        if os.path.exists(self.dataset_path) and os.listdir(self.dataset_path):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset archive from Hugging Face and extract it.

        Returns:
            None
        """
        download_and_extract_archive(
            url=self.url, download_root=self.root, filename='solarplantsbrazil.zip'
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Retrieve an image-mask pair by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                - 'image': A float32 tensor of shape (C, H, W)
                - 'mask': A long tensor of shape (1, H, W), containing binary labels
        """

        image = self._load_image(self.image_paths[index])
        mask = self._load_mask(self.mask_paths[index])
        sample = {'image': image, 'mask': mask}
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of image-mask pairs in the dataset.
        """
        return len(self.image_paths)

    def _load_image(self, path: str) -> Tensor:
        """Load an image as a float32 torch tensor.

        Args:
            path: Path to the input image file.

        Returns:
            A float32 tensor with shape (C, H, W).
        """
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        return torch.from_numpy(arr)

    def _load_mask(self, path: str) -> Tensor:
        """Load a binary mask from file and return as a tensor.

        Args:
            path: Path to the binary mask file.

        Returns:
            A long tensor with shape (H, W), with values 0 or 1.
        """
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.uint8)
        bin_mask = (arr > 0).astype(np.uint8)
        return torch.from_numpy(bin_mask).long()

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the SolarPlantsBrazil dataset.

        Args:
            sample: A dictionary with 'image' and 'mask' tensors.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered image and mask.
        """
        image = sample['image']
        mask = sample['mask']

        # Use RGB only
        if image.shape[0] == 4:
            image = image[:3]

        # Normalize for display
        image_np = image.numpy()
        image_np = image_np / np.max(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))

        mask_np = mask.squeeze().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image_np)
        axs[0].set_title('RGB Image')
        axs[0].axis('off')

        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title('Mask')
        axs[1].axis('off')

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
