# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Glacier Calving Front dataset."""

import glob
import os
from collections.abc import Callable
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_and_extract_archive, extract_archive


class GlacierCalvingFront(NonGeoDataset):
    """Glacier Calving Front dataset.

    The `Glacier Calving Front <https://doi.pangaea.de/10.1594/PANGAEA.940950>`__ dataset is a
    semantic segmentation dataset of marine-terminating glaciers.

    Dataset features:

    * 46,700 train, 18,744 validation, and 28,376 test images
    * varying spatial resolution of 6-20m
    * paired binary calving front segmentation masks
    * paired multi-class land cover segmentation masks

    Dataset format:

    * images are single-channel pngs with dimension 256x256
    * segmentation masks are single-channel pngs

    Dataset classes:

    1. background
    2. ocean
    3. rock
    4. glacier

    If you use this dataset in your research, please cite the following paper:

    * https://essd.copernicus.org/articles/14/4287/2022/

    .. versionadded:: 0.7
    """

    valid_splits = ('train', 'val', 'test')

    zipfilename = 'glacier_calving_data.zip'

    data_dir = 'glacier_calving_data'

    image_dir = 'sar_images'

    mask_dirs = ('fronts', 'zones')

    url = 'https://huggingface.co/datasets/torchgeo/glacier_calving_front/resolve/fbf246c432a909ef96d96811b1f1b8a1e8c06542/glacier_calving_data.zip'

    md5 = '56e39e33f88a9842f48c513083e3c50a'

    px_class_values: ClassVar[dict[int, str]] = {
        0: 'background',
        64: 'ocean',
        127: 'rock',
        254: 'glacier',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new instance of GlacierCalvingFront dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.valid_splits, f'split must be one of {self.valid_splits}'

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.fpaths = glob.glob(
            os.path.join(
                self.root,
                self.zipfilename.replace('.zip', ''),
                self.mask_dirs[0],
                self.split,
                '*.png',
            )
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.fpaths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the image and mask at the given index.

        Args:
            idx: index of the image and mask to return

        Returns:
            dict: a dict containing the image and mask
        """
        zones_path = self.fpaths[idx]
        img_path = zones_path.replace('_zones_', '_')
        front_path = zones_path.replace('_zones_', '_front_')
        img = Image.open(img_path)

        front_mask = Image.open(front_path)
        zone_mask = Image.open(zones_path)

        sample = {
            'image': torch.from_numpy(np.array(img)).unsqueeze(0).float(),
            'mask_front': torch.from_numpy(np.array(front_mask)).long(),
            'mask_zone': torch.from_numpy(np.array(zone_mask)).long(),
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        exists = []
        if os.path.exists(
            os.path.join(
                self.root,
                self.zipfilename.replace('.zip', ''),
                self.image_dir,
                self.split,
            )
        ):
            exists.append(True)
        else:
            exists.append(False)

        for mask_dir in self.mask_dirs:
            if os.path.exists(
                os.path.join(
                    self.root,
                    self.zipfilename.replace('.zip', ''),
                    mask_dir,
                    self.split,
                )
            ):
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        # check download of zipfile
        if os.path.exists(os.path.join(self.root, self.zipfilename)):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfilename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfilename), self.root)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`GlacierCalvingFront.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if 'prediction' in sample:
            ncols = 4
        else:
            ncols = 3
        fig, axs = plt.subplots(1, ncols, figsize=(15, 5))

        axs[0].imshow(sample['image'].permute(1, 2, 0).numpy())
        axs[0].axis('off')

        axs[1].imshow(sample['mask_front'].numpy(), cmap='gray')
        axs[1].axis('off')

        axs[2].imshow(sample['mask_zone'].numpy(), cmap='gray')
        axs[2].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Front Mask')
            axs[2].set_title('Zone Mask')

        if 'prediction' in sample:
            axs[3].imshow(sample['prediction'].numpy(), cmap='gray')
            axs[3].axis('off')
            if show_titles:
                axs[3].set_title('Prediction')

        if suptitle:
            fig.suptitle(suptitle)

        return fig
