# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars dataset."""

import glob
import os
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class USAVars(NonGeoDataset):
    """USAVars dataset.

    The USAVars dataset is reproduction of the dataset used in the paper "`A
    generalizable and accessible approach to machine learning with global satellite
    imagery <https://doi.org/10.1038/s41467-021-24638-z>`_". Specifically, this dataset
    includes 1 sq km. crops of NAIP imagery resampled to 4m/px cenetered on ~100k points
    that are sampled randomly from the contiguous states in the USA. Each point contains
    three continuous valued labels (taken from the dataset released in the paper): tree
    cover percentage, elevation, and population density.

    Dataset format:

    * images are 4-channel GeoTIFFs
    * labels are singular float values

    Dataset labels:

    * tree cover
    * elevation
    * population density

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41467-021-24638-z

    .. versionadded:: 0.3
    """

    data_url = 'https://hf.co/datasets/torchgeo/usavars/resolve/01377abfaf50c0cc8548aaafb79533666bbf288f/{}'  # noqa: E501
    dirname = 'uar'

    md5 = '677e89fd20e5dd0fe4d29b61827c2456'

    label_urls = {
        'housing': data_url.format('housing.csv'),
        'income': data_url.format('income.csv'),
        'roads': data_url.format('roads.csv'),
        'nightlights': data_url.format('nightlights.csv'),
        'population': data_url.format('population.csv'),
        'elevation': data_url.format('elevation.csv'),
        'treecover': data_url.format('treecover.csv'),
    }

    split_metadata = {
        'train': {
            'url': data_url.format('train_split.txt'),
            'filename': 'train_split.txt',
            'md5': '3f58fffbf5fe177611112550297200e7',
        },
        'val': {
            'url': data_url.format('val_split.txt'),
            'filename': 'val_split.txt',
            'md5': 'bca7183b132b919dec0fc24fb11662a0',
        },
        'test': {
            'url': data_url.format('test_split.txt'),
            'filename': 'test_split.txt',
            'md5': '97bb36bc003ae0bf556a8d6e8f77141a',
        },
    }

    ALL_LABELS = ['treecover', 'elevation', 'population']

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        labels: Sequence[str] = ALL_LABELS,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new USAVars dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            labels: list of labels to include
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if invalid labels are provided
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root

        assert split in self.split_metadata
        self.split = split

        for lab in labels:
            assert lab in self.ALL_LABELS

        self.labels = labels
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

        self.label_dfs = {
            lab: pd.read_csv(os.path.join(self.root, lab + '.csv'), index_col='ID')
            for lab in self.labels
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        tif_file = self.files[index]
        id_ = tif_file[5:-4]

        sample = {
            'labels': Tensor(
                [self.label_dfs[lab].loc[id_][lab] for lab in self.labels]
            ),
            'image': self._load_image(os.path.join(self.root, 'uar', tif_file)),
            'centroid_lat': Tensor([self.label_dfs[self.labels[0]].loc[id_]['lat']]),
            'centroid_lon': Tensor([self.label_dfs[self.labels[0]].loc[id_]['lon']]),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> list[str]:
        """Loads file names."""
        with open(os.path.join(self.root, f'{self.split}_split.txt')) as f:
            files = f.read().splitlines()
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: np.typing.NDArray[np.int_] = f.read()
            tensor = torch.from_numpy(array).float()
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, 'uar')
        csv_pathname = os.path.join(self.root, '*.csv')
        split_pathname = os.path.join(self.root, '*_split.txt')

        csv_split_count = (len(glob.glob(csv_pathname)), len(glob.glob(split_pathname)))
        if glob.glob(pathname) and csv_split_count == (7, 3):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dirname + '.zip')
        if glob.glob(pathname) and csv_split_count == (7, 3):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for f_name in self.label_urls:
            download_url(self.label_urls[f_name], self.root, filename=f_name + '.csv')

        download_url(self.data_url, self.root, md5=self.md5 if self.checksum else None)

        for metadata in self.split_metadata.values():
            download_url(
                metadata['url'],
                self.root,
                md5=metadata['md5'] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.dirname + '.zip'))

    def plot(
        self,
        sample: dict[str, Tensor],
        show_labels: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_labels: flag indicating whether to show labels above panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image'][:3].numpy()  # get RGB inds
        image = np.moveaxis(image, 0, 2)

        fig, axs = plt.subplots(figsize=(10, 10))
        axs.imshow(image)
        axs.axis('off')

        if show_labels:
            labels = [(lab, val) for lab, val in sample.items() if lab != 'image']
            label_string = ''
            for lab, val in labels:
                label_string += f'{lab}={round(val[0].item(), 2)} '
            axs.set_title(label_string)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
