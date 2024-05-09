# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SKy Images and Photovoltaic Power Dataset (SKIPP'D)."""

import os
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive, lazy_import


class SKIPPD(NonGeoDataset):
    """SKy Images and Photovoltaic Power Dataset (SKIPP'D).

    The `SKIPP'D dataset <https://purl.stanford.edu/dj417rh1007>`_
    contains ground-based fish-eye photos of the sky for solar
    forecasting tasks.

    Dataset Format:

    * .hdf5 file containing images and labels
    * .npy files with corresponding datetime timestamps

    Dataset Features:

    * fish-eye RGB images (64x64px)
    * power output measurements from 30-kW rooftop PV array
    * 1-min interval across 3 years (2017-2019)

    Nowcast task:

    * 349,372 images under the split key *trainval*
    * 14,003 images under the split key *test*

    Forecast task:

    * 130,412 images under the split key *trainval*
    * 2,462 images under the split key *test*
    * consists of a concatenated RGB time-series of 16
      time-steps

    If you use this dataset in your research, please cite:

    * https://doi.org/10.48550/arXiv.2207.00913

    .. note::

       This dataset requires the following additional library to be installed:

       * ` <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.5
    """

    url = 'https://hf.co/datasets/torchgeo/skippd/resolve/a16c7e200b4618cd93be3143cdb973e3f21498fa/{}'  # noqa: E501
    md5 = {
        'forecast': 'f4f3509ddcc83a55c433be9db2e51077',
        'nowcast': '0000761d403e45bb5f86c21d3c69aa80',
    }

    data_file_name = '2017_2019_images_pv_processed_{}.hdf5'
    zipfile_name = '2017_2019_images_pv_processed_{}.zip'

    valid_splits = ['trainval', 'test']

    valid_tasks = ['nowcast', 'forecast']

    dateformat = '%m/%d/%Y, %H:%M:%S'

    def __init__(
        self,
        root: str = 'data',
        split: str = 'trainval',
        task: str = 'nowcast',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "trainval", or "test"
            task: one fo "nowcast", or "forecast"
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``task`` or ``split`` is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
            MissingDependencyError: If h5py is not installed.
        """
        lazy_import('h5py')

        assert (
            split in self.valid_splits
        ), f'Please choose one of these valid data splits {self.valid_splits}.'
        self.split = split

        assert (
            task in self.valid_tasks
        ), f'Please choose one of these valid tasks {self.valid_tasks}.'
        self.task = task

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        h5py = lazy_import('h5py')
        with h5py.File(
            os.path.join(self.root, self.data_file_name.format(self.task)), 'r'
        ) as f:
            num_datapoints: int = f[self.split]['pv_log'].shape[0]

        return num_datapoints

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, str | Tensor] = {'image': self._load_image(index)}
        sample.update(self._load_features(index))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load the input image.

        Args:
            index: index of image to load

        Returns:
            image tensor at index
        """
        h5py = lazy_import('h5py')
        with h5py.File(
            os.path.join(self.root, self.data_file_name.format(self.task)), 'r'
        ) as f:
            arr = f[self.split]['images_log'][index]

        # forecast has dimension [16, 64, 64, 3] but reshape to [48, 64, 64]
        # https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset/blob/main/models/SUNSET_forecast.ipynb
        if self.task == 'forecast':
            arr = rearrange(arr, 't h w c-> (t c) h w')
        else:
            arr = rearrange(arr, 'h w c -> c h w')

        tensor = torch.from_numpy(arr).to(torch.float32)
        return tensor

    def _load_features(self, index: int) -> dict[str, str | Tensor]:
        """Load label.

        Args:
            index: index of label to load

        Returns:
            label tensor at index
        """
        h5py = lazy_import('h5py')
        with h5py.File(
            os.path.join(self.root, self.data_file_name.format(self.task)), 'r'
        ) as f:
            label = f[self.split]['pv_log'][index]

        path = os.path.join(self.root, f'times_{self.split}_{self.task}.npy')
        datestring = np.load(path, allow_pickle=True)[index].strftime(self.dateformat)

        features: dict[str, str | Tensor] = {
            'label': torch.tensor(label, dtype=torch.float32),
            'date': datestring,
        }
        return features

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.data_file_name.format(self.task))
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_name.format(self.task))
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        download_url(
            self.url.format(self.zipfile_name.format(self.task)),
            self.root,
            filename=self.zipfile_name.format(self.task),
            md5=self.md5[self.task] if self.checksum else None,
        )
        self._extract()

    def _extract(self) -> None:
        """Extract the dataset."""
        zipfile_path = os.path.join(self.root, self.zipfile_name.format(self.task))
        extract_archive(zipfile_path, self.root)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        In the ``forecast`` task the latest image is plotted.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if self.task == 'nowcast':
            image, label = sample['image'].permute(1, 2, 0), sample['label'].item()
        else:
            image, label = (
                sample['image'].permute(1, 2, 0).reshape(64, 64, 3, 16)[:, :, :, -1],
                sample['label'][-1].item(),
            )

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction'].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image / 255)
        ax.axis('off')

        if show_titles:
            title = f'Label: {label:.3f}'
            if showing_predictions:
                title += f'\nPrediction: {prediction:.3f}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
