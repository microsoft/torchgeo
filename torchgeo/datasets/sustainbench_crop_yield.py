# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SustainBench Crop Yield dataset."""

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url, extract_archive


class SustainBenchCropYield(NonGeoDataset):
    """SustainBench Crop Yield Dataset.

    This dataset contains MODIS band histograms and soybean yield
    estimates for selected counties in the USA, Argentina and Brazil.
    The dataset is part of the
    `SustainBench <https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_yield.html>`_
    datasets for tackling the UN Sustainable Development Goals (SDGs).

    Dataset Format:

    * .npz files of stacked samples

    Dataset Features:

    * input histogram of 7 surface reflectance and 2 surface temperature
      bands from MODIS pixel values in 32 ranges across 32 timesteps
      resulting in 32x32x9 input images
    * regression target value of soybean yield in metric tonnes per
      harvested hectare

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1145/3209811.3212707
    * https://doi.org/10.1609/aaai.v31i1.11172

    .. versionadded:: 0.5
    """

    valid_countries = ('usa', 'brazil', 'argentina')

    md5 = '362bad07b51a1264172b8376b39d1fc9'

    url = 'https://drive.google.com/file/d/1lhbmICpmNuOBlaErywgiD6i9nHuhuv0A/view?usp=drive_link'

    dir = 'soybeans'

    valid_splits = ('train', 'dev', 'test')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        countries: list[str] = ['usa'],
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "dev", or "test"
            countries: which countries to include in the dataset
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``countries`` contains invalid countries or if ``split``
                is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(countries).issubset(
            self.valid_countries
        ), f'Please choose a subset of these valid countried: {self.valid_countries}.'
        self.countries = countries

        assert (
            split in self.valid_splits
        ), f'Pleas choose one of these valid data splits {self.valid_splits}.'
        self.split = split

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.images = []
        self.features = []

        for country in self.countries:
            image_file_path = os.path.join(
                self.root, self.dir, country, f'{self.split}_hists.npz'
            )
            target_file_path = image_file_path.replace('_hists', '_yields')
            years_file_path = image_file_path.replace('_hists', '_years')
            ndvi_file_path = image_file_path.replace('_hists', '_ndvi')

            npz_file = np.load(image_file_path)['data']
            target_npz_file = np.load(target_file_path)['data']
            year_npz_file = np.load(years_file_path)['data']
            ndvi_npz_file = np.load(ndvi_file_path)['data']
            num_data_points = npz_file.shape[0]
            for idx in range(num_data_points):
                sample = npz_file[idx]
                sample = torch.from_numpy(sample).permute(2, 0, 1).to(torch.float32)
                self.images.append(sample)

                target = target_npz_file[idx]
                year = year_npz_file[idx]
                ndvi = ndvi_npz_file[idx]

                features = {
                    'label': torch.tensor(target).to(torch.float32),
                    'year': torch.tensor(int(year)),
                    'ndvi': torch.from_numpy(ndvi).to(dtype=torch.float32),
                }
                self.features.append(features)

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: Sample = {'image': self.images[index]}
        sample.update(self.features[index])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.dir)
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dir) + '.zip'
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
            self.url,
            self.root,
            filename=self.dir + '.zip',
            md5=self.md5 if self.checksum else None,
        )
        self._extract()

    def _extract(self) -> None:
        """Extract the dataset."""
        zipfile_path = os.path.join(self.root, self.dir) + '.zip'
        extract_archive(zipfile_path, self.root)

    def plot(
        self,
        sample: Sample,
        band_idx: int = 0,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            band_idx: which of the nine histograms to index
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        """
        image, label = sample['image'], sample['label'].item()

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction'].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0)[:, :, band_idx])
        ax.axis('off')

        if show_titles:
            title = f'Label: {label:.3f}'
            if showing_predictions:
                title += f'\nPrediction: {prediction:.3f}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
