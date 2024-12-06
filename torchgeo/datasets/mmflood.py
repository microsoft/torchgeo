# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MMFlood dataset."""

import os
import pathlib
from collections.abc import Callable
from glob import glob
from typing import ClassVar, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import BoundingBox, Path, download_url, extract_archive


class MMFlood(RasterDataset):
    """MMFlood dataset.

    `MMFlood <https://huggingface.co/datasets/links-ads/mmflood>`__ dataset is a multimodal flood delineation dataset.

    Dataset features:

    * 1,748 Sentinel-1 acquisitions
    * multimodal dataset
    * 95 flood events from 42 different countries
    * hydrography maps (not available for all Sentinel-1 acquisitions)
    * flood delineation maps (ground truth) is obtained from Copernicus EMS

    Dataset classes:

    0. no flood
    1. flood

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/ACCESS.2022.3205419
    """

    url = 'https://huggingface.co/datasets/links-ads/mmflood/resolve/24ca097306c9e50ad0711903c11e1ba13ea1bedc/'
    _name = 'mmflood'
    _categories: ClassVar[dict[int, str]] = {0: 'background', 1: 'flood'}
    _palette: ClassVar[dict[int, tuple[int, int, int]]] = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        255: (255, 0, 255),
    }
    _ignore_index = 255
    _nparts = 11
    _mean = (0.1785585, 0.03574104, 168.45529)
    _median = (0.116051525, 0.025692634, 86.0)
    _std = (2.405442, 0.22719479, 242.74359)

    metadata: ClassVar[dict[str, str]] = {
        'part_file': 'activations.tar.{part}.gz.part',
        'filename': 'activations.tar.gz',
        'directory': 'activations',
        'metadata_file': 'activations.json',
    }
    _splits: ClassVar[set[str]] = {'train', 'val', 'test'}
    _md5: ClassVar[dict[str, str]] = {
        'activations.json': 'de33a3ac7e55a0051ada21cbdfbb4745',
        'activations.tar.gz': '3cd4c4fe7506aa40263f74639d85ccce',
        'activations.tar.000.gz.part': 'a8424653edca6e79999831bdda53d4dc',
        'activations.tar.001.gz.part': '517def8760d3ce86885c7600c77a1d6c',
        'activations.tar.002.gz.part': '6797b97121f5b98ff58fde7491f584b2',
        'activations.tar.003.gz.part': 'e69d2a6b1746ef869d1da4d22018a71a',
        'activations.tar.004.gz.part': '0ccf7ea69ea6c0e88db1b1015ec3361e',
        'activations.tar.005.gz.part': '8ef6765afe20f254b1e752d7a2742fda',
        'activations.tar.006.gz.part': '3f330a44b66511b7a95f4a555f8b793a',
        'activations.tar.007.gz.part': '1d2046b5f3c473c3681a05dc94b29b86',
        'activations.tar.008.gz.part': 'f386b5acf78f8ae34592404c6c7ec43c',
        'activations.tar.009.gz.part': 'dd5317a3c0d33de815beadb9850baa38',
        'activations.tar.010.gz.part': '5a14a7e3f916c5dcf288c2ca88daf4d0',
    }

    def __init__(
        self,
        root: Path = 'data',
        crs: CRS | None = None,
        split: str = 'train',
        include_dem: bool = False,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
        cache: bool = False,
    ) -> None:
        """Initialize a new MMFlood dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: coordinate reference system to be used
            split: train/val/test split to load
            include_dem: If True, DEM data is concatenated after Sentinel-1 bands.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        """
        assert split in self._splits

        self.root = root
        self.split = split
        self.include_dem = include_dem
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        # Verify integrity of the dataset, initializing
        # self.image_files, self.label_files, self.dem_files attributes
        self._verify()
        self.metadata_df = self._load_metadata()
        self.folders = self._load_folders(check_folders=True)
        paths = [x['s1_raw'] for x in self.folders]

        # Build the index
        super().__init__(paths=paths, crs=crs, transforms=transforms, cache=cache)
        return

    def _merge_tar_files(self) -> None:
        """Merge part tar gz files."""
        dst_filename = self.metadata['filename']
        dst_path = os.path.join(self.root, dst_filename)

        print('Merging separate part files...')
        with open(dst_path, 'wb') as dst_fp:
            for idx in range(self._nparts):
                part_filename = f'activations.tar.{idx:03}.gz.part'
                part_path = os.path.join(self.root, part_filename)
                print(f'Processing file {part_path!s}')

                with open(part_path, 'rb') as part_fp:
                    dst_fp.write(part_fp.read())
        return

    def __getitem__(self, query: BoundingBox) -> dict[str, Tensor]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample containing image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        indexes = cast(list[int], [hit.id for hit in hits])

        if not indexes:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        image = self._load_image(indexes, query)
        mask = self._load_target(indexes, query)

        sample = {'image': image, 'mask': mask, 'crs': self.crs, 'bounds': query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata.

        Returns:
            dataframe containing metadata
        """
        df = pd.read_json(
            os.path.join(self.root, self.metadata['metadata_file'])
        ).transpose()
        return df

    def _load_folders(self, check_folders: bool = False) -> list[dict[str, str]]:
        """Load folder paths.

        Args:
            check_folders: if True, verify pairings of all s1, dem and mask data across all the folders

        Returns:
            list of dicts of s1, dem and masks folder paths
        """
        # initialize tif file lists containing masks, DEM and S1_raw data
        folders = self.metadata_df[
            self.metadata_df['subset'] == self.split
        ].index.tolist()

        image_files = []
        mask_files = []
        dem_files = []
        for f in folders:
            path = os.path.join(self.root, self.metadata['directory'], f'{f}-*')
            image_files += glob(os.path.join(path, 's1_raw', '*.tif'))
            mask_files += glob(os.path.join(path, 'mask', '*.tif'))
            dem_files += glob(os.path.join(path, 'DEM', '*.tif'))

        image_files = sorted(image_files)
        mask_files = sorted(mask_files)
        dem_files = sorted(dem_files)

        # Verify image, dem and mask lengths
        assert (
            len(image_files) > 0
        ), f'No images found, is the given path correct? ({self.root!s})'
        assert (
            len(image_files) == len(mask_files)
        ), f'Length mismatch between tiles and masks: {len(image_files)} != {len(mask_files)}'
        assert len(image_files) == len(
            dem_files
        ), 'Length mismatch between tiles and DEMs'

        res_folders = [
            {'s1_raw': img_path, 'mask': mask_path, 'dem': dem_path}
            for img_path, mask_path, dem_path in zip(image_files, mask_files, dem_files)
        ]

        if not check_folders:
            return res_folders

        # Verify image, dem and mask pairings
        for image, mask, dem in zip(image_files, mask_files, dem_files):
            image_tile = pathlib.Path(image).stem
            mask_tile = pathlib.Path(mask).stem
            dem_tile = pathlib.Path(dem).stem
            assert (
                image_tile == mask_tile == dem_tile
            ), f'Filenames not matching: image {image_tile}; mask {mask_tile}; dem {dem_tile}'

        return res_folders

    def _load_image(self, index: list[int], query: BoundingBox) -> Tensor:
        """Load a either a single image or a set of images, merging them.

        Args:
            index: indexes to return
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            the merged image
        """
        image = self._load_tif(index, modality='s1_raw', query=query).float()
        if self.include_dem:
            dem = self._load_tif(index, modality='dem', query=query).float()
            image = torch.cat([image, dem], dim=0)
        return image

    def _load_tif(
        self,
        index: list[int],
        modality: Literal['s1_raw', 'dem', 'mask'],
        query: BoundingBox,
    ) -> Tensor:
        """Load either a single geotif or a set of geotifs, merging them.

        Args:
            index: indexes to return
            modality: s1_raw, dem or mask
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            the merged image
        """
        assert query is not None, 'Query must be specified.'
        paths = [self.folders[idx][modality] for idx in index]
        tensor = self._merge_files(paths, query)
        return tensor

    def _load_target(self, index: list[int], query: BoundingBox) -> Tensor:
        """Load the target mask for either a single image or a set of images, merging them.

        Args:
            index: indexes to return
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            the target mask
        """
        tensor = self._load_tif(index, modality='mask', query=query).type(torch.uint8)
        return tensor.squeeze(dim=0)

    def _download(self) -> None:
        """Download the dataset."""

        def _check_and_download(filename: str, url: str) -> None:
            path = os.path.join(self.root, filename)
            if not os.path.exists(path):
                md5 = self._md5[filename] if self.checksum else None
                download_url(url, self.root, filename, md5)
            return

        filename = self.metadata['filename']
        filepath = os.path.join(self.root, filename)
        if not os.path.exists(filepath):
            for idx in range(self._nparts):
                part_file = f'activations.tar.{idx:03}.gz.part'
                url = self.url + part_file

                _check_and_download(part_file, url)

        _check_and_download(
            self.metadata['metadata_file'], self.url + self.metadata['metadata_file']
        )
        return

    def _extract(self) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        filepath = os.path.join(self.root, self.metadata['filename'])
        if str(filepath).endswith('.tar.gz'):
            extract_archive(filepath)
        return

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        dirpath = os.path.join(self.root, self.metadata['directory'])
        metadata_filepath = os.path.join(self.root, self.metadata['metadata_file'])
        # Check if both metadata file and directory exist
        if os.path.isdir(dirpath) and os.path.isfile(metadata_filepath):
            return
        if not self.download:
            raise DatasetNotFoundError(self)
        self._download()
        self._merge_tar_files()
        self._extract()
        return

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
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        show_mask = 'mask' in sample
        image = sample['image'][[0, 1]].permute(1, 2, 0).numpy()
        ncols = 1
        mask_offset = 1
        show_predictions = 'prediction' in sample
        if self.include_dem:
            dem = sample['image'][-1].squeeze(0).numpy()
            ncols += 1
        if show_mask:
            mask = sample['mask'].numpy()
            ncols += 1
        if show_predictions:
            pred = sample['prediction'].numpy()
            ncols += 1
            mask_offset = 2

        # Compute False Color image, from biomassters plot function
        co_polarization = image[..., 0]  # transmit == receive
        cross_polarization = image[..., 1]  # transmit != receive
        ratio = co_polarization / cross_polarization

        # https://gis.stackexchange.com/a/400780/123758
        co_polarization = np.clip(co_polarization / 0.3, a_min=0, a_max=1)
        cross_polarization = np.clip(cross_polarization / 0.05, a_min=0, a_max=1)
        ratio = np.clip(ratio / 25, a_min=0, a_max=1)

        image = np.stack((co_polarization, cross_polarization, ratio), axis=-1)

        # Generate the figure
        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        axs[0].imshow(image)
        axs[0].axis('off')
        if self.include_dem:
            axs[1].imshow(dem, cmap='gray')
            axs[1].axis('off')
        if show_mask:
            axs[ncols - mask_offset].imshow(mask, cmap='gray')
            axs[ncols - mask_offset].axis('off')
        if show_predictions:
            axs[ncols - 1].imshow(pred, cmap='gray')
            axs[ncols - 1].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            if self.include_dem:
                axs[1].set_title('DEM')
            if show_mask:
                axs[ncols - mask_offset].set_title('Mask')
            if show_predictions:
                axs[ncols - 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
