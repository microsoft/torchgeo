# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tropical Cyclone Wind Estimation Competition dataset."""

import os
from collections.abc import Callable
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, which


class TropicalCyclone(NonGeoDataset):
    """Tropical Cyclone Wind Estimation Competition dataset.

    A collection of tropical storms in the Atlantic and East Pacific Oceans from 2000 to
    2019 with corresponding maximum sustained surface wind speed. This dataset is split
    into training and test categories for the purpose of a competition. Read more about
    the competition here:
    https://www.drivendata.org/competitions/72/predict-wind-speeds/.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/JSTARS.2020.3011907

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionchanged:: 0.4
       Class name changed from TropicalCycloneWindEstimation to TropicalCyclone
       to be consistent with TropicalCycloneDataModule.
    """

    url = (
        'https://radiantearth.blob.core.windows.net/mlhub/nasa-tropical-storm-challenge'
    )
    size = 366

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new TropicalCyclone instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in {'train', 'test'}

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download

        self.filename = f'{split}_set'
        if split == 'train':
            self.filename = f'{split}ing_set'

        self._verify()

        self.features = pd.read_csv(os.path.join(root, f'{self.filename}_features.csv'))
        self.labels = pd.read_csv(os.path.join(root, f'{self.filename}_labels.csv'))

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        sample: Sample = {
            'relative_time': torch.tensor(self.features.iat[index, 2]),
            'ocean': torch.tensor(self.features.iat[index, 3]),
            'label': torch.tensor(self.labels.iat[index, 1]),
        }

        image_id = self.labels.iat[index, 0]
        sample['image'] = self._load_image(image_id)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.labels)

    @lru_cache
    def _load_image(self, image_id: str) -> Tensor:
        """Load a single image.

        Args:
            image_id: Filename of the image.

        Returns:
            the image
        """
        filename = os.path.join(self.root, self.split, f'{image_id}.jpg')
        with Image.open(filename) as img:
            if img.height != self.size or img.width != self.size:
                # Moved in PIL 9.1.0
                try:
                    resample = Image.Resampling.BILINEAR
                except AttributeError:
                    resample = Image.BILINEAR  # type: ignore[attr-defined]
                img = img.resize(size=(self.size, self.size), resample=resample)
            array: np.typing.NDArray[np.int_] = np.array(img.convert('RGB'))
            tensor = torch.from_numpy(array)
            tensor = tensor.permute((2, 0, 1)).float()
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        files = [f'{self.filename}_features.csv', f'{self.filename}_labels.csv']
        exists = [os.path.exists(os.path.join(self.root, file)) for file in files]
        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        directory = os.path.join(self.root, self.split)
        os.makedirs(directory, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', f'{self.url}/{self.split}', directory, '--recursive=true')
        files = [f'{self.filename}_features.csv', f'{self.filename}_labels.csv']
        for file in files:
            azcopy('copy', f'{self.url}/{file}', self.root)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image, label = sample['image'], sample['label']

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction'].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0))
        ax.axis('off')

        if show_titles:
            title = f'Label: {label}'
            if showing_predictions:
                title += f'\nPrediction: {prediction}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
