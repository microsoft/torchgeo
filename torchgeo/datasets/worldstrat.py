# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WorldStrat Dataset."""

import os
from collections.abc import Callable, Sequence
from glob import glob
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    array_to_tensor,
    check_integrity,
    download_and_extract_archive,
    download_url,
    extract_archive,
    quantile_normalization,
)


class WorldStrat(NonGeoDataset):
    """WorldStrat dataset.

    `WorldStrat <https://worldstrat.github.io/>`_ is a multi-modal dataset covering nearly 10,000km2 of matched high and low resolution
    satellite imagery across the globe. High-resolution SPOT 6/7 imagery comes at a resolution of 1.5m/pixel and is matched with a time-series
    of Sentinel 2 data.

    Dataset features:

    * High resolution (1.5m/pixel) Airbus SPOT 6/7 imagery with RGBN channels
    * Low resolution (8x lower) Sentinel 2 L1C and L2A
    * globally distributed areas of interest around the world


    Dataset format:

    * pixel dimensions vary across AOI tiles
    * all modalities are 'tif' files except for 'hr_rgbn' which is 'png'
    * 'hr_ps', 'hr_pan', 'hr_rgbn' are high resolution data
    * 'lr_rgbn' is low resolution data and roughly 4x lower resolution than 'hr_rgbn'
    * 'l1c' and 'l2a' are Sentinel-2 data with 13 and 12 bands respectively and roughly 8x lower resolution than 'hr_rgbn'

    If you use this dataset in your research, please cite the following entries:

    * https://zenodo.org/records/15382551
    * https://arxiv.org/abs/2207.06418

    .. versionadded:: 0.10
    """

    modality_titles: ClassVar[dict[str, str]] = {
        'l1c': 'Sentinel-2 L1C',
        'l2a': 'Sentinel-2 L2A',
        'lr_rgbn': 'Low-res RGBN',
        'hr_ps': 'High-res PS',
        'hr_pan': 'High-res PAN',
        'hr_rgbn': 'High-res RGB',
    }

    all_modalities = ('hr_ps', 'hr_pan', 'hr_rgbn', 'lr_rgbn', 'l1c', 'l2a')

    valid_splits = ('train', 'val', 'test')

    # Top-level directories the archives extract into
    hr_dir = 'hr_dataset'
    lr_dir = 'lr_dataset'

    file_info_dict: ClassVar[dict[str, dict[str, str]]] = {
        'hr_dataset': {
            'url': 'https://zenodo.org/records/15382551/files/hr_dataset.zip?download=1',
            'filename': 'hr_dataset.zip',
            'md5': '5ae09bb3557ce131242a133d9758d9e7',
        },
        'lr_dataset_l1c': {
            'url': 'https://zenodo.org/records/15382551/files/lr_dataset_l1c.zip?download=1',
            'filename': 'lr_dataset_l1c.zip',
            'md5': 'e90ecfa4bf838ace0b51dea1031b5ed1',
        },
        'lr_dataset_l2a': {
            'url': 'https://zenodo.org/records/15382551/files/lr_dataset_l2a.zip?download=1',
            'filename': 'lr_dataset_l2a.zip',
            'md5': '7aa1878a37d22a6c7c4b84b022a14ad7',
        },
        'metadata': {
            'url': 'https://zenodo.org/records/15382551/files/metadata.csv?download=1',
            'filename': 'metadata.csv',
            'md5': '1a66ac42b9a688be18debd0d95633fa1',
        },
        'train_val_test_split': {
            'url': 'https://zenodo.org/records/15382551/files/stratified_train_val_test_split.csv?download=1',
            'filename': 'stratified_train_val_test_split.csv',
            'md5': '874612b59bbf7987f7de7edd48a30c70',
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        modalities: Sequence[str] = all_modalities,
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize the WorldStrat dataset.

        Args:
            root: Root directory where the dataset can be found.
            modalities: Sequence of input modalities to load, choose from
                'hr_ps', 'hr_pan', 'hr_rgbn', 'lr_rgbn', 'l1c', 'l2a'.
            split: The dataset split to load, choose from 'train', 'val', 'test'.
            transforms: A function/transform that takes in a dictionary of tensors
                and returns a transformed version.
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` or ``modalities``arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert all(modality in self.all_modalities for modality in modalities), (
            f'Invalid modality: {modalities}, please choose from {self.all_modalities}'
        )
        assert split in self.valid_splits, (
            f'Invalid split: {split}, please choose from {self.valid_splits}'
        )

        self.root = root
        self.modalities = modalities
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.file_path_df = pd.read_csv(
            os.path.join(
                self.root, self.file_info_dict['train_val_test_split']['filename']
            )
        )

        self.file_path_df = self.file_path_df[
            self.file_path_df['split'] == self.split
        ].reset_index(drop=True)
        self.metadata_df = pd.read_csv(
            os.path.join(self.root, self.file_info_dict['metadata']['filename'])
        )

    def __getitem__(self, index: int) -> Sample:
        """Retrieve a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Selected modalities of low and high resolution images and metadata.
        """
        file_entry = self.file_path_df.iloc[index]
        aoi = file_entry['tile']
        hr_tile_dir = os.path.join(self.root, self.hr_dir, aoi)
        lr_tile_dir = os.path.join(self.root, self.lr_dir, aoi)

        sample: Sample = {}

        modality_loaders: dict[str, Callable[[], Tensor]] = {
            'l1c': lambda: self._load_sentinel_data(os.path.join(lr_tile_dir, 'L1C')),
            'l2a': lambda: self._load_sentinel_data(os.path.join(lr_tile_dir, 'L2A')),
            'lr_rgbn': lambda: self._load_tiff(
                os.path.join(hr_tile_dir, f'{aoi}_rgbn.tiff')
            ),
            'hr_ps': lambda: self._load_tiff(
                os.path.join(hr_tile_dir, f'{aoi}_ps.tiff')
            ),
            'hr_pan': lambda: self._load_tiff(
                os.path.join(hr_tile_dir, f'{aoi}_pan.tiff')
            ),
            'hr_rgbn': lambda: torch.from_numpy(
                np.array(
                    Image.open(os.path.join(hr_tile_dir, f'{aoi}_rgb.png'))
                ).transpose(2, 0, 1)
            ).float(),
        }

        for modality in self.modalities:
            sample[f'image_{modality}'] = modality_loaders[modality]()

        # Add metadata, one row per low-res timestep n, ordered to match the
        # stacked L1C/L2A time dimension
        metadata = (
            self.metadata_df[self.metadata_df['tile'] == aoi]
            .sort_values('n')
            .reset_index(drop=True)
        )
        sample.update(
            {
                'lon': metadata['lon'][0],
                'lat': metadata['lat'][0],
                'low_res_date': metadata['lowres_date'].tolist(),
                'high_res_date': metadata['highres_date'][0],
            }
        )

        return sample

    def _sentinel_paths(self, data_dir: str) -> list[tuple[int, str]]:
        """Find Sentinel time-series files sorted by their timestep index.

        Args:
            data_dir: Directory containing the Sentinel data, in the dataset
                this is either the L1C or L2A directory with time-series.

        Returns:
            List of (timestep index, file path) pairs sorted by timestep index.
        """
        level = os.path.basename(data_dir)
        tiff_paths = glob(os.path.join(data_dir, f'*{level}_data.tiff'))

        # filenames are '<AOI>-<n>-<level>_data.tiff' and indexed by n
        pairs = [
            (int(os.path.basename(tiff_path).split('-')[-2]), tiff_path)
            for tiff_path in tiff_paths
        ]

        return sorted(pairs)

    def _load_sentinel_data(self, data_dir: str) -> Tensor:
        """Load Sentinel data for a given AOI in a data directory.

        Args:
            data_dir: Directory containing the Sentinel data, in the dataset
                this is either the L1C or L2A directory with time-series.

        Returns:
            Loaded Sentinel data stacked as tensor of shape [T, C, H, W],
            ordered by ascending timestep index.
        """
        data = [
            self._load_tiff(tiff_path)
            for _, tiff_path in self._sentinel_paths(data_dir)
        ]

        return torch.stack(data)

    def _load_tiff(self, tiff_path: str) -> Tensor:
        """Load a tiff file as a tensor."""
        with rasterio.open(tiff_path) as src:
            data = src.read()
            tensor = array_to_tensor(data)
        return tensor.float()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_path_df)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # check if directories are present
        exists = []
        split_info_path = os.path.join(
            self.root, self.file_info_dict['train_val_test_split']['filename']
        )
        if os.path.exists(split_info_path):
            df = pd.read_csv(split_info_path)
            df = df[df['split'] == self.split]
            # check that all tiles are present
            for tile in df['tile']:
                exists.append(
                    os.path.exists(os.path.join(self.root, self.hr_dir, tile))
                    and os.path.exists(os.path.join(self.root, self.lr_dir, tile))
                )
        else:
            exists.append(False)

        if all(exists):
            return

        # check if downloaded files are present
        exists = []
        for file in self.file_info_dict.values():
            path = os.path.join(self.root, file['filename'])
            if os.path.exists(path):
                if self.checksum:
                    md5 = file['md5']
                    if not check_integrity(path, md5):
                        raise RuntimeError(f'Archive {file["filename"]} corrupted')
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _extract(self) -> None:
        """Extract archives to root directory."""
        for file in self.file_info_dict.values():
            if file['filename'].endswith('.zip'):
                extract_archive(os.path.join(self.root, file['filename']), self.root)

    def _download(self) -> None:
        """Download the dataset and extract it."""
        for metadata in self.file_info_dict.values():
            if metadata['filename'].endswith('.zip'):
                download_and_extract_archive(
                    metadata['url'],
                    self.root,
                    filename=metadata['filename'],
                    md5=metadata['md5'] if self.checksum else None,
                )
            else:
                download_url(
                    metadata['url'],
                    self.root,
                    filename=metadata['filename'],
                    md5=metadata['md5'] if self.checksum else None,
                )

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
        n_panels = len([k for k in sample if k.startswith('image_')])
        n_panels += 'prediction' in sample

        fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)

        for panel, modality in enumerate(self.modalities):
            key = f'image_{modality}'
            if key in sample:
                img = sample[key]

                if modality in ['hr_ps', 'hr_pan']:
                    img = img[0, ...]
                elif modality == 'hr_rgbn':
                    img = img[0:3, ...]
                elif modality in ['l1c', 'l2a']:
                    img = img[0, [3, 2, 1], ...]

                img = quantile_normalization(img).numpy()

                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)

                axs[0, panel].imshow(img)
                axs[0, panel].axis('off')
                if show_titles:
                    axs[0, panel].set_title(self.modality_titles[modality])

        if 'prediction' in sample:
            pred = sample['prediction']
            if pred.shape[0] == 4:
                pred = pred[:3]
            pred = quantile_normalization(pred).numpy().transpose(1, 2, 0)
            axs[0, -1].imshow(pred)
            axs[0, -1].axis('off')
            if show_titles:
                axs[0, -1].set_title('Prediction')

        if suptitle:
            fig.suptitle(suptitle)

        return fig
