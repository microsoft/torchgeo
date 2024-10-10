# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""WorldStrat Dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import pandas as pd
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path


class WorldStrat(NonGeoDataset):
    """WorldStrat dataset."""

    all_modalities = ('sentinel1', 'sentinel2')

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
            modalities: Sequence of input modalities to load.
            transforms: A function/transform that takes in a dictionary of tensors
                and returns a transformed version.
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` or ``modalities``arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert all(
            modality in self.all_modalities for modality in modalities
        ), f'Invalid modality: {modalities}, please choose from {self.all_modalities}'
        assert (
            split in self.valid_splits
        ), f'Invalid split: {split}, please choose from {self.valid_splits}'

        self.root = root
        self.modalities = modalities
        self.split = split
        self.transforms = transforms
        self.download = download

        self._verify()

        self.file_path_df = pd.read_csv(
            os.path.join(
                self.root, self.file_info_dict['train_val_test_split']['filename']
            )
        )
        self.file_path_df = self.file_path_df[self.file_path_df['split'] == self.split]
        self.metadata_df = pd.read_csv(
            os.path.join(self.root, self.file_info_dict['metadata']['filename'])
        )
        self.metadata_df.rename(columns={'Unnamed: 0': 'tile'}, inplace=True)

        import pdb

        pdb.set_trace()

        print(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """"""
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_path_df)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # check if directories are present
        # maybe go through metatdata because that lists the directories?
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
            if os.path.exists(os.path.join(self.root, file['filename'])):
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            # extract files
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # download
        self._download()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        # TODO: implement download

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
