# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MMFlood dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from glob import glob
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import IntersectionDataset, RasterDataset
from .utils import BoundingBox, Path, download_url, extract_archive


class MMFloodComponent(RasterDataset):
    """Base component for MMFlood dataset."""

    def __init__(
        self,
        subfolders: list[str],
        content: Literal['s1_raw', 'DEM', 'hydro', 'mask'],
        root: Path = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize MMFloodComponent dataset instance.

        Args:
            subfolders: list of directories to be loaded
            content: specifies which component to load
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
        """
        self.content = content
        self.is_image = content != 'mask'
        paths = []
        for s in subfolders:
            paths += glob(os.path.join(root, '**', f'{s}*-*', self.content, '*.tif'))
        paths = sorted(paths)
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)


class MMFloodIntersection(IntersectionDataset):
    """Intersection dataset used to merge two or more MMFloodComponents."""

    def __init__(
        self,
        dataset1: MMFloodComponent | MMFloodIntersection,
        dataset2: MMFloodComponent | MMFloodIntersection,
    ) -> None:
        """Initialize a new MMFloodIntersection instance.

        Args:
            dataset1: the first dataset to merge
            dataset2: the second dataset to merge
        """
        # if hydro component is passed, it should always be passed as dataset2
        super().__init__(dataset1, dataset2)


class MMFlood(IntersectionDataset):
    """MMFlood dataset.

    `MMFlood <https://huggingface.co/datasets/links-ads/mmflood>`__ dataset is a multimodal
    flood delineation dataset. Sentinel-1 data is matched with masks and DEM data for all
    available tiles. If hydrography maps are loaded, only a subset of the dataset is loaded,
    since only 1,012 Sentinel-1 tiles have a corresponding hydrography map.
    Some Sentinel-1 tiles have missing data, which are automatically set to 0.
    Corresponding pixels in masks are set to 255 and should be ignored in performance computation.

    Dataset features:

    * 1,748 Sentinel-1 tiles of varying pixel dimensions
    * multimodal dataset
    * 95 flood events from 42 different countries
    * includes DEMs
    * includes hydrography maps (available for 1,012 tiles out of 1,748)
    * flood delineation maps (ground truth) is obtained from Copernicus EMS

    Dataset classes:

    0. no flood
    1. flood

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/ACCESS.2022.3205419

    .. versionadded:: 0.7
    """

    url = 'https://huggingface.co/datasets/links-ads/mmflood/resolve/24ca097306c9e50ad0711903c11e1ba13ea1bedc/'
    _ignore_index = 255
    _nparts = 11

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
        res: float | tuple[float, float] | None = None,
        split: str = 'train',
        include_dem: bool = False,
        include_hydro: bool = False,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
        cache: bool = False,
    ) -> None:
        """Initialize a new MMFlood dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            split: train/val/test split to load
            include_dem: If True, DEM data is concatenated after Sentinel-1 bands.
            include_hydro: If True, hydrography data is concatenated as last channel.
                Only a smaller subset of the original dataset is loaded in this case.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            AssertionError: If *split* is invalid.

        """
        assert split in self._splits

        self.root = root
        self.split = split
        self.include_dem = include_dem
        self.include_hydro = include_hydro
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        # Verify integrity of the dataset
        self._verify()
        self.metadata_df = pd.read_json(
            os.path.join(self.root, self.metadata['metadata_file'])
        ).transpose()

        split_subfolders = self.metadata_df[
            self.metadata_df['subset'] == self.split
        ].index.tolist()
        self.image: MMFloodComponent | MMFloodIntersection = MMFloodComponent(
            split_subfolders, 's1_raw', root, crs, res, cache=cache
        )
        if include_dem:
            dem = MMFloodComponent(split_subfolders, 'DEM', root, crs, res, cache=cache)
            self.image = MMFloodIntersection(self.image, dem)
        if include_hydro:
            hydro = MMFloodComponent(
                split_subfolders, 'hydro', root, crs, res, cache=cache
            )
            self.image = MMFloodIntersection(self.image, hydro)
        self.mask = MMFloodComponent(
            split_subfolders, 'mask', root, crs, res, cache=cache
        )

        super().__init__(self.image, self.mask, transforms=transforms)

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

    def __getitem__(self, query: BoundingBox) -> dict[str, Tensor]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        data = super().__getitem__(query)
        missing_data = data['image'].isnan().any(dim=0)
        # Set all pixel values of invalid areas to 0, all mask values to 255
        data['image'][:, missing_data] = 0
        data['mask'][missing_data] = self._ignore_index
        return data

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

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.metadata['filename'])
        if str(filepath).endswith('.tar.gz'):
            extract_archive(filepath)

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
        show_predictions = 'prediction' in sample
        if self.include_dem:
            dem_idx = -2 if self.include_hydro else -1
            dem = sample['image'][dem_idx].squeeze(0).numpy()
            ncols += 1
        if self.include_hydro:
            hydro = sample['image'][-1].squeeze(0).numpy()
            ncols += 1
        if show_mask:
            mask = sample['mask'].numpy()
            # Set ignore_index values to 0
            mask[mask == self._ignore_index] = 0
            ncols += 1
        if show_predictions:
            pred = sample['prediction'].numpy()
            ncols += 1

        # Compute False Color image, from Sentinel1 plot function
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
        axs_idx = 1
        if self.include_dem:
            axs[axs_idx].imshow(dem, cmap='gray')
            axs[axs_idx].axis('off')
            axs_idx += 1
        if self.include_hydro:
            axs[axs_idx].imshow(hydro, cmap='gray')
            axs[axs_idx].axis('off')
            axs_idx += 1
        if show_mask:
            axs[axs_idx].imshow(mask, cmap='gray')
            axs[axs_idx].axis('off')
            axs_idx += 1
        if show_predictions:
            axs[axs_idx].imshow(pred, cmap='gray')
            axs[axs_idx].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs_idx = 1
            if self.include_dem:
                axs[axs_idx].set_title('DEM')
                axs_idx += 1
            if self.include_hydro:
                axs[axs_idx].set_title('Hydrography Map')
                axs_idx += 1
            if show_mask:
                axs[axs_idx].set_title('Mask')
                axs_idx += 1
            if show_predictions:
                axs[axs_idx].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
