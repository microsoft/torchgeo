# Copyright (c) Microsoft Corporation. All rights reserved.
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
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    array_to_tensor,
    check_integrity,
    download_and_extract_archive,
    download_url,
    extract_archive,
)


class WorldStrat(NonGeoDataset):
    """WorldStrat dataset.

    The WorldStrat dataset is a multi-modal dataset covering nearly 10,000km2 of matched high and low resolution
    satellite imagery across the globe. High-resolution SPOT 6/7 imagery comes at a resolution of 1.5m/pixel and is matched with a time-series
    of Sentinel 2 data.

    Dataset features:

    * High resolution (1.5m/pixel) Airbus SPOT 6/7 imagery with RGBN channels
    * Low resolution (8x lower) Sentinel 2 L1C and L2A data


    Dataset format:

    * pixel dimensions vary across AOI tiles
    * all modalities are 'tif' files except for 'hr_rgbn' which is 'png'
    * 'hr_ps', 'hr_pan', 'hr_rgbn' are high resolution data of pixel dimension
    * 'lr_rgbn' is low resolution data of pixel dimension and roughly 4x lower resolution than 'hr_rgbn'
    * 'l1c' and 'l2a' are Sentinel-2 data with 13 and 12 bands respectively and roughly 8x lower resolution than 'hr_rgbn'

    If you use this dataset in your research, please cite the following entries:

    * https://zenodo.org/records/6810792
    * https://arxiv.org/abs/2207.06418

    .. version_added:: 0.7
    """

    all_modalities = ('hr_ps', 'hr_pan', 'hr_rgbn', 'lr_rgbn', 'l1c', 'l2a')

    valid_splits = ('train', 'val', 'test')

    file_info_dict: ClassVar[dict[str, dict[str, str]]] = {
        'hr_dataset': {
            'url': 'https://zenodo.org/records/6810792/files/hr_dataset.tar.gz?download=1',
            'filename': 'hr_dataset.tar.gz',
            'md5': 'ca7167334006f3c17f9071f14c435335',
        },
        'lr_dataset_l1c': {
            'url': 'https://zenodo.org/records/6810792/files/lr_dataset_l1c.tar.gz?download=1',
            'filename': 'lr_dataset_l1c.tar.gz',
            'md5': 'd2dcafa207b1e1bc6c754607f15e9ed6',
        },
        'lr_dataset_l2a': {
            'url': 'https://zenodo.org/records/6810792/files/lr_dataset_l2a.tar.gz?download=1',
            'filename': 'lr_dataset_l2a.tar.gz',
            'md5': '8cfc6a477cee9e9cd8b20ea27227de65',
        },
        'metadata': {
            'url': 'https://zenodo.org/records/6810792/files/metadata.csv?download=1',
            'filename': 'metadata.csv',
            'md5': 'dfeb3348e79b719bf03c230d5d258839',
        },
        'train_val_test_split': {
            'url': 'https://zenodo.org/records/6810792/files/stratified_train_val_test_split.csv?download=1',
            'filename': 'stratified_train_val_test_split.csv',
            'md5': '745035835d835280aa0298a9dc1996d1',
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
        self.metadata_df.rename(columns={'Unnamed: 0': 'tile_id'}, inplace=True)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Selected modalities of low and high resolution images and metadata.
        """
        file_entry = self.file_path_df.iloc[idx]
        aoi = file_entry['tile']
        data_dir = os.path.join(self.root, aoi)

        sample: dict[str, Tensor] = {}

        modality_loaders = {
            'l1c': lambda: self._load_sentinel_data(os.path.join(data_dir, 'L1C')),
            'l2a': lambda: self._load_sentinel_data(os.path.join(data_dir, 'L2A')),
            'lr_rgbn': lambda: self._load_tiff(
                os.path.join(data_dir, f'{aoi}_rgbn.tiff')
            ),
            'hr_ps': lambda: self._load_tiff(os.path.join(data_dir, f'{aoi}_ps.tiff')),
            'hr_pan': lambda: self._load_tiff(
                os.path.join(data_dir, f'{aoi}_pan.tiff')
            ),
            'hr_rgbn': lambda: torch.from_numpy(
                np.array(
                    Image.open(os.path.join(data_dir, f'{aoi}_rgb.png'))
                ).transpose(2, 0, 1)
            ).float(),
        }

        # Load only selected modalities
        for modality in self.modalities:
            sample[f'image_{modality}'] = modality_loaders[modality]()

        # Add metadata
        metadata = self.metadata_df[self.metadata_df['tile'] == aoi].reset_index(
            drop=True
        )
        sample.update(
            {
                'lon': metadata['lon'][0],
                'lat': metadata['lat'][0],
                'low_res_date': metadata['lowres_date'][0],
                'high_res_date': metadata['highres_date'][0],
            }
        )

        return sample

    def _load_sentinel_data(self, data_dir: str) -> Tensor:
        """Load Sentinel data for a given AOI in a data directory.

        Args:
            data_dir: Directory containing the Sentinel data, in the dataset
                this is either the L1C or L2A directory with time-series.

        Returns:
            Loaded Sentinel data stacked as tensor so [T, C, H, W].
        """
        tiff_paths = glob(
            os.path.join(data_dir, f'*{os.path.basename(data_dir)}_data.tiff'),
            recursive=True,
        )

        # load and stack the data
        data = []
        for tiff_path in tiff_paths:
            data.append(self._load_tiff(tiff_path))

        return torch.stack(data).float()

    def _load_tiff(self, tiff_path: str) -> Tensor:
        """Load a tiff file as a tensor."""
        with rasterio.open(tiff_path) as src:
            data = src.read()
            tensor = array_to_tensor(data)
        return tensor

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
                exists.append(os.path.exists(os.path.join(self.root, tile)))
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
            # extract files
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # download
        self._download()

    def _extract(self) -> None:
        """Extract tar balls to root directory."""
        for file in self.file_info_dict.values():
            if 'tar.gz' in file['filename']:
                extract_archive(os.path.join(self.root, file['filename']), self.root)

    def _download(self) -> None:
        """Download the dataset and extract it."""
        # TODO: implement download
        for _, metadata in self.file_info_dict.items():
            if 'tar.gz' in metadata['filename']:
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
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by __getitem__
            show_titles: Flag indicating whether to show titles above each panel
            suptitle: Optional string to use as a suptitle

        Returns:
            A matplotlib Figure with the rendered sample
        """
        # Determine number of panels needed
        n_panels = len([k for k in sample.keys() if k.startswith('image_')])
        if 'prediction' in sample:
            n_panels += 1

        fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axs = [axs]

        panel = 0
        modality_titles = {
            'l1c': 'Sentinel-2 L1C',
            'l2a': 'Sentinel-2 L2A',
            'lr_rgbn': 'Low-res RGBN',
            'hr_ps': 'High-res PS',
            'hr_pan': 'High-res PAN',
            'hr_rgbn': 'High-res RGB',
        }

        # Plot each modality
        for modality in self.modalities:
            key = f'image_{modality}'
            if key in sample:
                img = sample[key]
                if img.ndim == 3:
                    img = img.permute(1, 2, 0)
                if key == 'image_l1c':
                    img = img[0].permute(1, 2, 0)[..., [3, 2, 1]]
                if key == 'image_l2a':
                    img = img[0].permute(1, 2, 0)[..., [3, 2, 1]]
                axs[panel].imshow(img)
                axs[panel].axis('off')
                if show_titles:
                    axs[panel].set_title(modality_titles[modality])
                panel += 1

        # Plot prediction if available
        if 'prediction' in sample:
            axs[-1].imshow(sample['prediction'])
            axs[-1].axis('off')
            if show_titles:
                axs[-1].set_title('Prediction')

        if suptitle:
            fig.suptitle(suptitle)

        return fig
