# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TimeSen2Crop dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url, extract_archive


class TimeSen2Crop(NonGeoDataset):
    """TimeSen2Crop dataset.

    The `TimeSen2Crop <https://doi.org/10.5281/zenodo.4715630>`__ dataset is a
    pixel-based dataset of Sentinel-2 image time series for crop type
    classification, densely annotated with the Austrian Land Parcel
    Identification System (LPIS).

    Dataset features:

    * 1,212,224 labeled pixel time series
    * 15 Sentinel-2 tiles covering Austria for the 2017/2018 agronomic year
      (September 2017 to August 2018)
    * 1 additional tile (``2019_33UVP``) for the 2019 agronomic year
    * 9 spectral bands (B02, B03, B04, B05, B06, B07, B8A, B11, B12) of
      atmospherically corrected (Level-2A) surface reflectance
    * per-acquisition condition flag (clear, cloud, shadow, or snow)
    * 16 crop type classes

    Dataset format:

    * one directory per Sentinel-2 tile, each containing a ``dates.csv`` file
      with the acquisition dates in chronological order
    * one subdirectory per crop type class (``0``--``15``) inside each tile
    * one CSV file per labeled pixel, where each row is an acquisition and
      each column is a spectral band, plus a final condition flag column

    Dataset classes:

    0. Legumes
    1. Grassland
    2. Maize
    3. Potato
    4. Sunflower
    5. Soy
    6. Winter barley
    7. Winter caraway
    8. Rye
    9. Rapeseed
    10. Beet
    11. Spring cereals
    12. Winter wheat
    13. Winter triticale
    14. Permanent plantation
    15. Other crops

    Condition flags:

    0. clear
    1. cloud
    2. shadow
    3. snow

    .. note::

       The column headers in the per-pixel CSV files are generically named
       B1--B9, but according to the dataset description they correspond to the
       Sentinel-2 bands B02, B03, B04, B05, B06, B07, B8A, B11, and B12.

    .. note::

       The number of acquisitions T differs between tiles, so samples from
       different tiles cannot be batched together without padding or
       interpolation.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/JSTARS.2021.3073965

    .. versionadded:: 0.10
    """

    url = 'https://zenodo.org/records/4715631/files/TimeSen2Crop.zip'
    filename = 'TimeSen2Crop.zip'
    md5 = 'b5b7aad3fef192e78252e11c9a0e5cb8'
    directory = 'TimeSen2Crop'

    all_tiles = (
        '32TNT',
        '32TPT',
        '32TQT',
        '33TUM',
        '33TUN',
        '33TVM',
        '33TVN',
        '33TWM',
        '33TWN',
        '33TXN',
        '33UUP',
        '33UVP',
        '33UWP',
        '33UWQ',
        '33UXP',
        '2019_33UVP',
    )

    all_bands = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12')

    classes = (
        'Legumes',
        'Grassland',
        'Maize',
        'Potato',
        'Sunflower',
        'Soy',
        'Winter barley',
        'Winter caraway',
        'Rye',
        'Rapeseed',
        'Beet',
        'Spring cereals',
        'Winter wheat',
        'Winter triticale',
        'Permanent plantation',
        'Other crops',
    )

    flags = ('clear', 'cloud', 'shadow', 'snow')

    def __init__(
        self,
        root: Path = 'data',
        tiles: Sequence[str] = all_tiles[:-1],
        bands: Sequence[str] = all_bands,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new TimeSen2Crop dataset instance.

        Args:
            root: root directory where dataset can be found
            tiles: Sentinel-2 tiles to load (defaults to all tiles of the
                2017/2018 agronomic year)
            bands: spectral bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: If *tiles* or *bands* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert len(tiles) > 0, "'tiles' cannot be empty"
        assert set(tiles) <= set(self.all_tiles), (
            f'Only the following tiles are valid: {self.all_tiles}.'
        )
        assert len(bands) > 0, "'bands' cannot be empty"
        assert set(bands) <= set(self.all_bands), (
            f'Only the following bands are valid: {self.all_bands}.'
        )

        self.root = root
        self.tiles = tiles
        self.bands = bands
        self.band_indices = [self.all_bands.index(band) for band in bands]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.dates = {tile: self._load_dates(tile) for tile in tiles}
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            time series, acquisition dates, condition flags, and label at that
            index
        """
        tile, filepath, label = self.files[index]
        array = np.loadtxt(filepath, delimiter=',', skiprows=1, ndmin=2)
        sample = {
            'image': torch.from_numpy(array[:, self.band_indices]).float(),
            'condition': torch.from_numpy(array[:, -1]).long(),
            'date': self.dates[tile],
            'label': torch.tensor(label),
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

    def _load_dates(self, tile: str) -> Tensor:
        """Load the acquisition dates of a tile.

        Args:
            tile: Sentinel-2 tile to load

        Returns:
            acquisition dates in YYYYMMDD format
        """
        filepath = os.path.join(self.root, self.directory, tile, 'dates.csv')
        array = np.loadtxt(filepath, dtype=np.int64, skiprows=1, ndmin=1)
        return torch.from_numpy(array)

    def _load_files(self) -> list[tuple[str, str, int]]:
        """Return the paths and labels of the files in the dataset.

        Returns:
            list of (tile, filepath, label) tuples
        """
        files = []
        for tile in self.tiles:
            pathname = os.path.join(self.root, self.directory, tile, '*', '*.csv')
            for filepath in sorted(glob.glob(pathname)):
                label = int(os.path.basename(os.path.dirname(filepath)))
                files.append((tile, filepath, label))
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        path = os.path.join(self.root, self.directory)
        if os.path.exists(path):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

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
        image = sample['image'].numpy()
        label = cast(int, sample['label'].item())
        label_class = self.classes[label]

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = cast(int, sample['prediction'].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        for i, band in enumerate(self.bands):
            ax.plot(image[:, i], label=band)
        ax.set_xlabel('Acquisition')
        ax.set_ylabel('Surface reflectance (x10000)')
        ax.legend(loc='upper right', fontsize='small')
        if show_titles:
            title = f'Label: {label_class}'
            if showing_predictions:
                title += f'\nPrediction: {prediction_class}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
