# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SkyScript dataset."""

import os
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_and_extract_archive, download_url, extract_archive


class SkyScript(NonGeoDataset):
    """SkyScript dataset.

    `SkyScript <https://github.com/wangzhecheng/SkyScript>`__ is a large and
    semantically diverse image-text dataset for remote sensing images. It contains
    5.2 million remote sensing image-text pairs in total, covering more than 29K
    distinct semantic tags.

    If you use this dataset in your research, please cite it using the following format:

    * https://arxiv.org/abs/2312.12856

    .. versionadded:: 0.6
    """

    url = 'https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/{}'

    image_dirs = tuple(f'images{i}' for i in range(2, 8))
    image_md5s = (
        'fbfb5f7aa1731f4106fc3ffbd608100a',
        'ad4fd9fdb9622d1ea360210cb222f2bd',
        'aeeb41e830304c74b14b5ffc1fc8e8c3',
        '02ee7e0e59f9ac1c87b678a155e1f1df',
        '350475f1e7fb996152fa16db891b4142',
        '5e2fbf3e9262b36e30b458ec9a1df625',
    )

    #: Can be modified in subclasses to change train/val/test split
    caption_files: ClassVar[dict[str, str]] = {
        'train': 'SkyScript_train_top30pct_filtered_by_CLIP_openai.csv',
        'val': 'SkyScript_val_5K_filtered_by_CLIP_openai.csv',
        'test': 'SkyScript_test_30K_filtered_by_CLIP_openai.csv',
    }
    caption_md5s: ClassVar[dict[str, str]] = {
        'train': '05b362e43a852667b5374c9a5ae53f8e',
        'val': 'c8d278fd29b754361989d5e7a6608f69',
        'test': '0135d9b49ce6751360912a4353e809dc',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SkyScript instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', 'test'.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.caption_files

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.captions = pd.read_csv(os.path.join(self.root, self.caption_files[split]))

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.captions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            A dict containing image and caption at index.
        """
        filepath, title = self.captions.iloc[index][:2]

        with Image.open(os.path.join(self.root, filepath)) as img:
            array = np.array(img, dtype=np.float32)
            array = rearrange(array, 'h w c -> c h w')
            image = torch.from_numpy(array)

        sample = {'image': image, 'text': title}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        md5: str | None
        for directory, md5 in zip(self.image_dirs, self.image_md5s):
            # Check if the extracted files already exist
            if os.path.isdir(os.path.join(self.root, directory)):
                continue

            # Check if the zip files have already been downloaded
            if os.path.isfile(os.path.join(self.root, f'{directory}.zip')):
                extract_archive(os.path.join(self.root, f'{directory}.zip'))
                continue

            # Check if the user requested to download the dataset
            if not self.download:
                raise DatasetNotFoundError(self)

            # Download the dataset
            url = self.url.format(f'{directory}.zip')
            md5 = md5 if self.checksum else None
            download_and_extract_archive(url, self.root, md5=md5)

        # Download the caption file
        if self.download:
            url = self.url.format(self.caption_files[self.split])
            md5 = self.caption_md5s[self.split] if self.checksum else None
            download_url(url, self.root, md5=md5)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = rearrange(sample['image'], 'c h w -> h w c') / 255

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            title = sample['text']
            if 'prediction' in sample:
                title += '\n' + sample['prediction']
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
