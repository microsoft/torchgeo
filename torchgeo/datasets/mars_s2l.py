# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""IMEO MARS-S2L dataset."""

import math
import os
from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url


class MARSS2L(NonGeoDataset):
    """IMEO MARS-S2L methane plume detection dataset.

    The dataset consists of pre-chipped 200x200 Sentinel-2 tiles. Each tile holds
    two scenes stacked along the band axis: the first six bands belong to the scene
    that (may) contain a methane plume, the last six to a plume-free reference scene.
    Samples are described by CSV manifests (``train.csv``, ``val.csv``, ``test.csv``)
    located in ``root``.

    Dataset format:

    * ``<split>.csv`` manifest with the columns ``s2path``, ``plumepath``,
      ``cloudmaskpath``, ``ch4path`` and ``isplume``, plus ``wind_u``, ``wind_v``,
      ``ch4_fluxrate`` and ``ch4_fluxrate_std`` if the respective data is loaded
    * 12-band Sentinel-2 GeoTIFFs (``s2path``)
    * single-band plume masks (``plumepath``, may be empty for unlabeled tiles)
    * single-band cloud masks (``cloudmaskpath``)
    * single-band CH4 concentration maps (``ch4path``)

    All paths in the manifest are relative to ``root``.

    The dataset is hosted on
    `Hugging Face <https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L>`__ and is
    licensed under CC-BY-NC-SA-4.0. The full repository is roughly 100 GB; with
    ``download=True`` only the manifest of the requested split and the rasters it
    references (restricted to the ones selected by the ``load_*`` flags) are fetched.

    .. versionadded:: 0.10.1
    """

    splits = ('train', 'val', 'test')
    url = 'https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L/resolve/main/'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        load_ch4: bool = False,
        load_cloudmask: bool = False,
        load_wind: bool = False,
        load_fluxrate: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize a new MARSS2L dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            load_ch4: if True, also load the CH4 concentration raster
            load_cloudmask: if True, also load the cloud mask raster
            load_wind: if True, also load ``wind_u`` and ``wind_v``
            load_fluxrate: if True, also load ``fluxrate`` and ``flux_std``
            download: if True, download the split from Hugging Face and store it in
                ``root`` (files that already exist are skipped)

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.load_ch4 = load_ch4
        self.load_cloudmask = load_cloudmask
        self.load_wind = load_wind
        self.load_fluxrate = load_fluxrate

        self.download = download

        self.csv_path = os.path.join(self.root, f'{split}.csv')
        if download:
            self._download()

        if not os.path.isfile(self.csv_path):
            raise DatasetNotFoundError(self)

        self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.df)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """
        if not -len(self) <= index < len(self):
            raise IndexError(f'index {index} is out of range for size {len(self)}')

        row = self.df.iloc[index]

        # (12, H, W): plume scene with 6 bands followed by reference scene with 6 bands
        image = self._load_raster(row['s2path'], torch.float32, bands=None)

        # (H, W)
        if pd.notna(row['plumepath']):
            mask = self._load_raster(row['plumepath'], torch.uint8)
        else:
            mask = torch.zeros(image.shape[1:], dtype=torch.uint8)

        sample: Sample = {
            'image': image,
            'mask': mask,
            'is_plume': torch.tensor(bool(row['isplume']), dtype=torch.bool),
        }

        if self.load_ch4:
            if pd.notna(row['ch4path']):
                sample['ch4'] = self._load_raster(row['ch4path'], torch.float32)
            else:
                sample['ch4'] = torch.zeros(image.shape[1:], dtype=torch.float32)

        if self.load_cloudmask:
            sample['cloudmask'] = self._load_raster(row['cloudmaskpath'], torch.uint8)

        if self.load_fluxrate:
            sample['fluxrate'] = torch.tensor(row['ch4_fluxrate'], dtype=torch.float32)
            sample['flux_std'] = torch.tensor(
                row['ch4_fluxrate_std'], dtype=torch.float32
            )

        if self.load_wind:
            sample['wind_u'] = torch.tensor(row['wind_u'], dtype=torch.float32)
            sample['wind_v'] = torch.tensor(row['wind_v'], dtype=torch.float32)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _download(self) -> None:
        """Download the manifest of the split and the rasters it references."""
        self._download_file(f'{self.split}.csv')

        columns = ['s2path', 'plumepath']
        if self.load_ch4:
            columns.append('ch4path')
        if self.load_cloudmask:
            columns.append('cloudmaskpath')

        df = pd.read_csv(self.csv_path)
        for column in columns:
            for path in df[column].dropna().unique():
                self._download_file(path)

    def _download_file(self, path: str) -> None:
        """Download a single file of the repository unless it already exists.

        Args:
            path: path relative to the repository root, mirrored below ``root``
        """
        fpath = os.path.join(self.root, path)
        if not os.path.isfile(fpath):
            download_url(
                self.url + path, os.path.dirname(fpath), os.path.basename(fpath)
            )

    def _load_raster(
        self, path: Path, dtype: torch.dtype, bands: int | None = 1
    ) -> Tensor:
        """Load a raster file.

        Args:
            path: path to the file, relative to ``root``
            dtype: data type of the returned tensor
            bands: single band index to read (1-based), or None to read all bands

        Returns:
            the raster, 2D if a single band was read else 3D
        """
        with rasterio.open(os.path.join(self.root, path)) as src:
            array = src.read() if bands is None else src.read(bands)
        return torch.from_numpy(array).to(dtype)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image']  # (12, H, W)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Panel 1: true color RGB of the plume scene with wind arrow
        rgb = image[2:5].numpy().transpose(1, 2, 0)
        rgb_norm = np.zeros_like(rgb)
        for c in range(3):
            lo, hi = np.percentile(rgb[:, :, c], 2), np.percentile(rgb[:, :, c], 98)
            if hi > lo:
                rgb_norm[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo), 0, 1)
        axes[0].imshow(rgb_norm)
        axes[0].axis('off')

        wind_u = float(sample.get('wind_u', 0))
        wind_v = float(sample.get('wind_v', 0))
        wind_speed = math.hypot(wind_u, wind_v)
        if wind_speed > 0:
            wind_dir_deg = (270 - math.degrees(math.atan2(wind_v, wind_u))) % 360
            axes[0].arrow(
                0.5,
                0.5,
                0.15 * wind_u / wind_speed,
                0.15 * wind_v / wind_speed,
                head_width=0.04,
                head_length=0.06,
                fc='red',
                ec='red',
                transform=axes[0].transAxes,
            )
            if show_titles:
                axes[0].text(
                    0.5,
                    0.95,
                    f'Wind: {wind_speed:.1f} m/s, {wind_dir_deg:.0f}°',
                    transform=axes[0].transAxes,
                    ha='center',
                    va='top',
                    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7},
                )

        # Panels 2 and 3: SWIR2 (B12) of plume and reference scene
        for ax, band in zip(axes[1:3], (5, 11)):
            swir2 = image[band].numpy()
            vmin, vmax = np.percentile(swir2, 1), np.percentile(swir2, 99)
            ax.imshow(swir2, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')

        # Panel 4: plume mask
        mask = sample['mask'].numpy()
        axes[3].imshow(mask, cmap='Reds')
        axes[3].axis('off')

        if show_titles:
            axes[0].set_title('RGB (B04/B03/B02) - Plume Scene')
            axes[1].set_title('SWIR2 (B12) - Plume Scene')
            axes[2].set_title('SWIR2 (B12) - Reference Scene')
            title = f'Plume Mask\nis_plume={bool(sample["is_plume"])}'
            if 'fluxrate' in sample:
                title += f'\nCH4 flux: {float(sample["fluxrate"]):.1f} kg/h'
                title += f'\nCH4 flux std: {float(sample["flux_std"]):.1f}'
            axes[3].set_title(title)

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
