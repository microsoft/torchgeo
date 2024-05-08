# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CropHarvest datasets."""

import glob
import json
import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive, lazy_import


class CropHarvest(NonGeoDataset):
    """CropHarvest dataset.

    `CropHarvest <https://github.com/nasaharvest/cropharvest>`__ is a
    crop classification dataset.

    Dataset features:

    * single pixel time series with crop-type labels
    * 18 bands per image over 12 months

    Dataset format:

    * arrays are 12x18 with 18 bands over 12 months

    Dataset properties:

    1. is_crop - whether or not a single pixel contains cropland
    2. classification_label - optional field identifying a specific crop type
    3. dataset - source dataset for the imagery
    4. lat - latitude
    5. lon - longitude

    If you use this dataset in your research, please cite the following paper:

    * https://openreview.net/forum?id=JtjzUXPEaCu

    This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.6
    """

    # https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/bands.py
    all_bands = [
        'VV',
        'VH',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B11',
        'B12',
        'temperature_2m',
        'total_precipitation',
        'elevation',
        'slope',
        'NDVI',
    ]
    rgb_bands = ['B4', 'B3', 'B2']

    features_url = 'https://zenodo.org/records/7257688/files/features.tar.gz?download=1'
    labels_url = 'https://zenodo.org/records/7257688/files/labels.geojson?download=1'
    file_dict = {
        'features': {
            'url': features_url,
            'filename': 'features.tar.gz',
            'extracted_filename': os.path.join('features', 'arrays'),
            'md5': 'cad4df655c75caac805a80435e46ee3e',
        },
        'labels': {
            'url': labels_url,
            'filename': 'labels.geojson',
            'extracted_filename': 'labels.geojson',
            'md5': 'bf7bae6812fc7213481aff6a2e34517d',
        },
    }

    def __init__(
        self,
        root: str = 'data',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CropHarvest dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.files = self._load_features(self.root)
        self.labels = self._load_labels(self.root)
        self.classes = self.labels['properties.label'].unique()
        self.classes = self.classes[self.classes != np.array(None)]
        self.classes = np.insert(self.classes, 0, ['None', 'Other'])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            single pixel time-series array and label at that index

        Raises:
            MissingDependencyError: If h5py is not installed.
        """
        files = self.files[index]
        data = self._load_array(files['chip'])

        label = self._load_label(files['index'], files['dataset'])
        sample = {'array': data, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_features(self, root: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing path for each of hd5 single pixel time series and
            its key for associated data
        """
        files = []
        chips = glob.glob(
            os.path.join(root, self.file_dict['features']['extracted_filename'], '*.h5')
        )
        chips = sorted(os.path.basename(chip) for chip in chips)
        for chip in chips:
            chip_path = os.path.join(
                root, self.file_dict['features']['extracted_filename'], chip
            )
            index = chip.split('_')[0]
            dataset = chip.split('_')[1][:-3]
            files.append(dict(chip=chip_path, index=index, dataset=dataset))
        return files

    def _load_labels(self, root: str) -> pd.DataFrame:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            pandas dataframe containing label data for each feature
        """
        filename = self.file_dict['labels']['extracted_filename']
        with open(os.path.join(root, filename), encoding='utf8') as f:
            data = json.load(f)
            df = pd.json_normalize(data['features'])
            return df

    def _load_array(self, path: str) -> Tensor:
        """Load an individual single pixel time series.

        Args:
            path: path to the image

        Returns:
            the image
        """
        h5py = lazy_import('h5py')
        filename = os.path.join(path)
        with h5py.File(filename, 'r') as f:
            array = f.get('array')[()]
            tensor = torch.from_numpy(array)
            return tensor

    def _load_label(self, idx: str, dataset: str) -> Tensor:
        """Load the crop-type label for a single pixel time series.

        Args:
            idx: sample index in labels.geojson
            dataset: dataset name to query labels.geojson

        Returns:
            the crop-type label
        """
        index = int(idx)
        row = self.labels[
            (self.labels['properties.index'] == index)
            & (self.labels['properties.dataset'] == dataset)
        ]
        row = row.to_dict(orient='records')[0]
        label = 'None'
        if row['properties.label']:
            label = row['properties.label']
        elif row['properties.is_crop'] == 1:
            label = 'Other'

        return torch.tensor(np.where(self.classes == label)[0][0])

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if feature files already exist
        feature_path = os.path.join(
            self.root, self.file_dict['features']['extracted_filename']
        )
        feature_path_zip = os.path.join(
            self.root, self.file_dict['features']['filename']
        )
        label_path = os.path.join(
            self.root, self.file_dict['labels']['extracted_filename']
        )
        # Check if labels exist
        if os.path.exists(label_path):
            # Check if features exist
            if os.path.exists(feature_path):
                return
            # Check if features are downloaded in zip format
            if os.path.exists(feature_path_zip):
                self._extract()
                return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        features_path = os.path.join(self.file_dict['features']['filename'])
        download_url(
            self.file_dict['features']['url'],
            self.root,
            filename=features_path,
            md5=self.file_dict['features']['md5'] if self.checksum else None,
        )

        download_url(
            self.file_dict['labels']['url'],
            self.root,
            filename=os.path.join(self.file_dict['labels']['filename']),
            md5=self.file_dict['labels']['md5'] if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        features_path = os.path.join(self.root, self.file_dict['features']['filename'])
        extract_archive(features_path)

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset using bands for Agriculture RGB composite.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots()
        bands = [self.all_bands.index(band) for band in self.rgb_bands]
        rgb = np.array(sample['array'])[:, bands] / 3000
        axs.imshow(rgb[None, ...])
        axs.set_title(f'Crop type: {self.classes[sample["label"]]}')
        axs.set_xticks(np.arange(12))
        axs.set_xticklabels(np.arange(12) + 1)
        axs.set_yticks([])
        axs.set_xlabel('Month')
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
