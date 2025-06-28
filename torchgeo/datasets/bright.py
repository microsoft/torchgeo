# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BRIGHT dataset."""

import os
import textwrap
from collections.abc import Callable
from typing import ClassVar

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from einops import repeat
from matplotlib import colors
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_url, extract_archive


class BRIGHTDFC2025(NonGeoDataset):
    """BRIGHT DFC2025 dataset.

    The `BRIGHT <https://github.com/ChenHongruixuan/BRIGHT>`__ dataset consists of bi-temporal
    high-resolution multimodal images for
    building damage assessment. The dataset is part of the 2025 IEEE GRSS Data Fusion Contest.
    The pre-disaster images are optical images and the post-disaster images are SAR images, and
    targets were manually annotated. The dataset is split into train, val, and test splits, but
    the test split does not contain targets in this version.

    More information can be found at the `Challenge website <https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/?tab=data-fusion-contest>`__.

    Dataset Features:

    * Pre-disaster optical images from MAXAR, NAIP, NOAA Digital Coast Raster Datasets, and the National Plan for Aerial Orthophotography Spain
    * Post-disaster SAR images from Capella Space and Umbra
    * high image resolution of 0.3-1m

    Dataset Format:

    * Images are in GeoTIFF format with pixel dimensions of 1024x1024
    * Pre-disaster are three channel images
    * Post-disaster SAR images are single channel but repeated to have 3 channels

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2501.06019

    .. versionadded:: 0.7
    """

    classes = ('background', 'intact', 'damaged', 'destroyed')

    colormap = (
        'white',  # background
        'green',  # intact
        'burlywood',  # damaged
        'red',  # destroyed
    )

    md5 = '45fd96716e7f5673869b166859a6cb3c'

    url = 'https://huggingface.co/datasets/torchgeo/bright/resolve/d19972f5e682ad684dcde35529a6afad4c719f1b/dfc25_track2_trainval_with_split.zip'

    data_dir = 'dfc25_track2_trainval'

    valid_split = ('train', 'val', 'test')

    # train_setlevels.txt are the training samples
    # holdout_setlevels.txt are the validation samples
    # val_setlevels.txt are the test samples
    split_files: ClassVar[dict[str, str]] = {
        'train': 'train_setlevel.txt',
        'val': 'holdout_setlevel.txt',
        'test': 'val_setlevel.txt',
    }

    px_class_values: ClassVar[dict[int, str]] = {
        0: 'background',
        1: 'intact',
        2: 'damaged',
        3: 'destroyed',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BRIGHT DFC2025 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            AssertionError: If *split* is not one of 'train', 'val', or 'test.
        """
        assert split in self.valid_split, f'Split must be one of {self.valid_split}'
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.sample_paths = self._get_paths()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and target at that index, pre and post image
            are returned under separate image keys
        """
        idx_paths = self.sample_paths[index]

        image_pre = self._load_image(idx_paths['image_pre']).float()
        image_post = self._load_image(idx_paths['image_post']).float()
        # https://github.com/ChenHongruixuan/BRIGHT/blob/11b1ffafa4d30d2df2081189b56864b0de4e3ed7/dfc25_benchmark/dataset/make_data_loader.py#L101
        # post image is stacked to also have 3 channels
        image_post = repeat(image_post, 'c h w -> (repeat c) h w', repeat=3)

        sample = {'image': torch.stack([image_pre, image_post])}

        if 'target' in idx_paths and self.split != 'test':
            target = self._load_image(idx_paths['target']).long()
            sample['mask'] = target

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _get_paths(self) -> list[dict[str, str]]:
        """Get paths to the dataset files based on specified splits.

        Returns:
            a list of dictionaries containing paths to the pre, post, and target images
        """
        split_file = self.split_files[self.split]

        file_path = os.path.join(self.root, self.data_dir, split_file)
        with open(file_path) as f:
            sample_ids = f.readlines()

        if self.split in ('train', 'val'):
            dir_split_name = 'train'
        else:
            dir_split_name = 'val'

        sample_paths = [
            {
                'image_pre': os.path.join(
                    self.root,
                    self.data_dir,
                    dir_split_name,
                    'pre-event',
                    f'{sample_id.strip()}_pre_disaster.tif',
                ),
                'image_post': os.path.join(
                    self.root,
                    self.data_dir,
                    dir_split_name,
                    'post-event',
                    f'{sample_id.strip()}_post_disaster.tif',
                ),
            }
            for sample_id in sample_ids
        ]
        if self.split != 'test':
            for sample, sample_id in zip(sample_paths, sample_ids):
                sample['target'] = os.path.join(
                    self.root,
                    self.data_dir,
                    dir_split_name,
                    'target',
                    f'{sample_id.strip()}_building_damage.tif',
                )

        return sample_paths

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # check if the text split files exist
        if all(
            os.path.exists(os.path.join(self.root, self.data_dir, split_file))
            for split_file in self.split_files.values()
        ):
            # if split txt files exist check whether sample files exist
            sample_paths = self._get_paths()
            exists = []
            for sample in sample_paths:
                exists.append(
                    all(os.path.exists(path) for name, path in sample.items())
                )
            if all(exists):
                return

        # check if .zip files already exists (if so, then extract)
        exists = []
        zip_file_path = os.path.join(self.root, self.data_dir + '.zip')
        if os.path.exists(zip_file_path):
            if self.checksum and not check_integrity(zip_file_path, self.md5):
                raise RuntimeError('Dataset found, but corrupted.')
            exists.append(True)
            extract_archive(zip_file_path, self.root)
        else:
            exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # download and extract the dataset
        self._download()
        extract_archive(zip_file_path, self.root)

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            self.data_dir + '.zip',
            md5=self.md5 if self.checksum else None,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            number of samples in the dataset
        """
        return len(self.sample_paths)

    def _load_image(self, path: Path) -> Tensor:
        """Load a file from disk.

        Args:
            path: path to the file to load

        Returns:
            image tensor
        """
        with rasterio.open(path) as src:
            img = src.read()
            tensor: Tensor = torch.from_numpy(img).float()
        return tensor

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2
        showing_mask = 'mask' in sample
        showing_prediction = 'prediction' in sample
        if showing_mask:
            ncols += 1
        if showing_prediction:
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))

        axs[0].imshow(sample['image'][0].permute(1, 2, 0) / 255.0)
        axs[0].axis('off')

        axs[1].imshow(sample['image'][1].permute(1, 2, 0) / 255.0)
        axs[1].axis('off')

        cmap = colors.ListedColormap(self.colormap)

        if showing_mask:
            axs[2].imshow(sample['mask'].squeeze(0), cmap=cmap, interpolation='none')
            axs[2].axis('off')
            unique_classes = np.unique(sample['mask'].numpy())
            handles = [
                mpatches.Patch(
                    color=cmap(ordinal),
                    label='\n'.join(
                        textwrap.wrap(self.px_class_values[px_class], width=10)
                    ),
                )
                for ordinal, px_class in enumerate(self.px_class_values.keys())
                if ordinal in unique_classes
            ]
            axs[2].legend(handles=handles, loc='upper right', bbox_to_anchor=(1.4, 1))
            if showing_prediction:
                axs[3].imshow(
                    sample['prediction'].squeeze(0), cmap=cmap, interpolation='none'
                )
                axs[3].axis('off')
        elif showing_prediction:
            axs[2].imshow(
                sample['prediction'].squeeze(0), cmap=cmap, interpolation='none'
            )
            axs[2].axis('off')

        if show_titles:
            axs[0].set_title('Pre-disaster image')
            axs[1].set_title('Post-disaster image')
            if showing_mask:
                axs[2].set_title('Ground truth')
            if showing_prediction:
                axs[-1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
