# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Fields Of The World dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, array_to_tensor, download_and_extract_archive, extract_archive


class FieldsOfTheWorld(NonGeoDataset):
    """Fields Of The World dataset.

    The `Fields Of The World <https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/>`__
    datataset is a semantic and instance segmentation dataset for delineating field
    boundaries.

    Dataset features:

    * 70462 patches across 24 countries
    * Each country has a train, val, and test split
    * Semantic segmentations masks with and without the field boundary class
    * Instance segmentation masks

    Dataset format:

    * images are four-channel GeoTIFFs with dimension 256x256
    * segmentation masks (both two and three class) are single-channel GeoTIFFs
    * instance masks are single-channel GeoTIFFs

    Dataset classes:

    1. background
    2. field
    3. field-boundary (three-class only)

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2409.16252

    .. versionadded:: 0.7
    """

    splits = ('train', 'val', 'test')
    targets = ('2-class', '3-class', 'instance')

    valid_countries = (
        'austria',
        'belgium',
        'brazil',
        'cambodia',
        'corsica',
        'croatia',
        'denmark',
        'estonia',
        'finland',
        'france',
        'germany',
        'india',
        'kenya',
        'latvia',
        'lithuania',
        'luxembourg',
        'netherlands',
        'portugal',
        'rwanda',
        'slovakia',
        'slovenia',
        'south_africa',
        'spain',
        'sweden',
        'vietnam',
    )

    base_url = 'https://data.source.coop/kerner-lab/fields-of-the-world-archive/'

    country_to_md5: ClassVar[dict[str, str]] = {
        'austria': '35604e3e3e78b4469e443bc756e19d26',
        'belgium': '111a9048e15391c947bc778e576e99b4',
        'brazil': '2ba96f9f01f37ead1435406c3f2b7c63',
        'cambodia': '581e9b8dae9713e4d03459bcec3c0bd0',
        'corsica': '0b38846063a98a31747fdeaf1ba03980',
        'croatia': 'dc5d33e19ae9e587c97f8f4c9852c87e',
        'denmark': 'ec817210b06351668cacdbd1a8fb9471',
        'estonia': 'b9c89e559e3c7d53a724e7f32ccf88ea',
        'finland': '23f853d6cbaea5a3596d1d38cc27fd65',
        'france': 'f05314f148642ff72d8bea903c01802d',
        'germany': 'd57a7ed203b9cf89c709aab29d687cee',
        'india': '361a688507e2e5cc7ca7138be01a5b80',
        'kenya': '80ca0335b25440379f99b7011dfbdfa2',
        'latvia': '6eeaaa57cdf18f25497f84e854a86d42',
        'lithuania': '0a2f4ab3309633e2de121d936e0763ba',
        'luxembourg': '5a8357eae364cca836b87827b3c6a3d3',
        'netherlands': '3afc61d184aab5c4fd6beaecf2b6c0a9',
        'portugal': '10485b747e1d8c082d33c73d032a7e05',
        'rwanda': '087ce56bbf06b32571ef27ff67bac43b',
        'slovakia': 'f66a0294491086d4c49dc4a804446e50',
        'slovenia': '6fa3ae3920bcc2c890a0d74435d9d29b',
        'south_africa': 'b7f1412d69922e8551cf91081401ec8d',
        'spain': '908bbf29597077c2c6954c439fe8265f',
        'sweden': '4b07726c421981bb2019e8900023393e',
        'vietnam': '32e1cacebcb2da656d40ab8522eb6737',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        target: str = '2-class',
        countries: str | Sequence[str] = ['austria'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Fields Of The World dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            target: one of "2-class", "3-class", or "instance" specifying which kind of
                target mask to load
            countries: which set of countries to load data from
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If any arguments are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        assert target in self.targets
        if isinstance(countries, str):
            countries = [countries]
        assert set(countries) <= set(self.valid_countries)

        self.root = root
        self.split = split
        self.target = target
        self.countries = countries
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and mask at that index with image of dimension 3x1024x1024
            and mask of dimension 1024x1024
        """
        win_a_fn = self.files[index]['win_a']
        win_b_fn = self.files[index]['win_b']
        mask_fn = self.files[index]['mask']

        win_a = self._load_image(win_a_fn)
        win_b = self._load_image(win_b_fn)
        mask = self._load_target(mask_fn)

        image = torch.cat((win_a, win_b), dim=0)
        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            a dictionary with "win_a", "win_b", and "mask" keys containing lists of
            file paths
        """
        files = []
        for country in self.countries:
            df = pd.read_parquet(
                os.path.join(self.root, country, f'chips_{country}.parquet')
            )
            aois = df[df['split'] == self.split]['aoi_id'].values

            for aoi in aois:
                if self.target == 'instance':
                    subdir = 'instance'
                elif self.target == '2-class':
                    subdir = 'semantic_2class'
                elif self.target == '3-class':
                    subdir = 'semantic_3class'

                win_a_fn = os.path.join(
                    self.root, country, 's2_images', 'window_a', f'{aoi}.tif'
                )
                win_b_fn = os.path.join(
                    self.root, country, 's2_images', 'window_b', f'{aoi}.tif'
                )

                # there are 333 AOIs that are missing imagery across the dataset
                if not (os.path.exists(win_a_fn) and os.path.exists(win_b_fn)):
                    continue

                sample = {
                    'win_a': win_a_fn,
                    'win_b': win_b_fn,
                    'mask': os.path.join(
                        self.root, country, 'label_masks', subdir, f'{aoi}.tif'
                    ),
                }
                files.append(sample)

        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the loaded image
        """
        filename = os.path.join(path)
        with rasterio.open(filename) as f:
            array: np.typing.NDArray[np.int_] = f.read()
            tensor = array_to_tensor(array).float()
        return tensor

    def _load_target(self, path: Path) -> Tensor:
        """Load a single mask corresponding to image.

        Args:
            path: path to the mask

        Returns:
            the mask of the image
        """
        filename = os.path.join(path)
        with rasterio.open(filename) as f:
            array: np.typing.NDArray[np.int_] = f.read(1)
            tensor = torch.from_numpy(array).long()
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        for country in self.countries:
            if self._verify_data(country):
                continue

            filename = f'{country}.zip'
            pathname = os.path.join(self.root, filename)
            if os.path.exists(pathname):
                extract_archive(pathname, os.path.join(self.root, country))
                continue

            if not self.download:
                raise DatasetNotFoundError(self)

            download_and_extract_archive(
                self.base_url + filename,
                os.path.join(self.root, country),
                filename=filename,
                md5=self.country_to_md5[country] if self.checksum else None,
            )

    def _verify_data(self, country: str) -> bool:
        """Verify that data for a country is extracted.

        Args:
            country: the country to check

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for entry in ['label_masks', 's2_images', f'chips_{country}.parquet']:
            if not os.path.exists(os.path.join(self.root, country, entry)):
                return False

        return True

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        win_a = einops.rearrange(sample['image'][0:3], 'c h w -> h w c')
        win_b = einops.rearrange(sample['image'][4:7], 'c h w -> h w c')
        mask = sample['mask']

        win_a = torch.clip(win_a / 3000, 0, 1)
        win_b = torch.clip(win_b / 3000, 0, 1)

        axs[0].imshow(win_a)
        axs[0].set_title('Window A')
        axs[1].imshow(win_b)
        axs[1].set_title('Window B')
        if self.target == 'instance':
            unique_vals = sorted(np.unique(mask))
            for i, val in enumerate(unique_vals):
                mask[mask == val] = i
            bg_mask = mask == 0
            mask = (mask % 9) + 1
            mask[bg_mask] = 0
            axs[2].imshow(mask, vmin=0, vmax=10, cmap='tab10', interpolation='none')
            axs[2].set_title('Instance mask')
        elif self.target == '2-class':
            axs[2].imshow(mask, vmin=0, vmax=2, cmap='gray', interpolation='none')
            axs[2].set_title('2-class mask')
        elif self.target == '3-class':
            axs[2].imshow(mask, vmin=0, vmax=2, cmap='gray', interpolation='none')
            axs[2].set_title('3-class mask')
        for ax in axs:
            ax.axis('off')

        if not show_titles:
            for ax in axs:
                ax.set_title('')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
