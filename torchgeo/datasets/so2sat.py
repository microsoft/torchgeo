# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat dataset."""

import os
from collections.abc import Callable, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import check_integrity, lazy_import, percentile_normalization


class So2Sat(NonGeoDataset):
    """So2Sat dataset.

    The `So2Sat <https://doi.org/10.1109/MGRS.2020.2964708>`__ dataset consists of
    corresponding synthetic aperture radar and multispectral optical image data
    acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and a
    corresponding local climate zones (LCZ) label. The dataset is distributed over
    42 cities across different continents and cultural regions of the world, and comes
    with a variety of different splits.

    This implementation covers the *2nd* and *3rd* versions of the dataset as described
    in the author's github repository: https://github.com/zhu-xlab/So2Sat-LCZ42.

    The different versions are as follows:

    Version 2: This version contains imagery from 52 cities and is split into train/val/test as follows:

    * Training: 42 cities around the world
    * Validation: western half of 10 other cities covering 10 cultural zones
    * Testing: eastern half of the 10 other cities

    Version 3: A version of the dataset with 3 different train/test splits, as follows:

    * Random split: every city 80% training / 20% testing (randomly sampled)
    * Block split: every city is split in a geospatial 80%/20%-manner
    * Cultural 10: 10 cities from different cultural zones are held back for testing purposes

    Dataset classes:

    0. Compact high rise
    1. Compact middle rise
    2. Compact low rise
    3. Open high rise
    4. Open mid rise
    5. Open low rise
    6. Lightweight low rise
    7. Large low rise
    8. Sparsely built
    9. Heavy industry
    10. Dense trees
    11. Scattered trees
    12. Bush, scrub
    13. Low plants
    14. Bare rock or paved
    15. Bare soil or sand
    16. Water

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/MGRS.2020.2964708

    .. note::

       The version 2 dataset can be automatically downloaded using the following bash
       script:

       .. code-block:: bash

          for split in training validation testing
          do
              wget ftp://m1483140:m1483140@dataserv.ub.tum.de/$split.h5
          done

       or manually downloaded from https://dataserv.ub.tum.de/index.php/s/m1483140
       This download will likely take several hours.

       The version 3 datasets can be downloaded using the following bash script:

       .. code-block:: bash

          for version in random block culture_10
          do
            for split in training testing
            do
              wget -P $version/ ftp://m1613658:m1613658@dataserv.ub.tum.de/$version/$split.h5
            done
          done

       or manually downloaded from https://mediatum.ub.tum.de/1613658

    .. note::

       This dataset requires the following additional library to be installed:

       * `<https://pypi.org/project/h5py/>`_ to load the dataset
    """  # noqa: E501

    versions = ['2', '3_random', '3_block', '3_culture_10']
    filenames_by_version = {
        '2': {
            'train': 'training.h5',
            'validation': 'validation.h5',
            'test': 'testing.h5',
        },
        '3_random': {'train': 'random/training.h5', 'test': 'random/testing.h5'},
        '3_block': {'train': 'block/training.h5', 'test': 'block/testing.h5'},
        '3_culture_10': {
            'train': 'culture_10/training.h5',
            'test': 'culture_10/testing.h5',
        },
    }
    md5s_by_version = {
        '2': {
            'train': '702bc6a9368ebff4542d791e53469244',
            'validation': '71cfa6795de3e22207229d06d6f8775d',
            'test': 'e81426102b488623a723beab52b31a8a',
        },
        '3_random': {
            'train': '94e2e2e667b406c2adf61e113b42204e',
            'test': '1e15c425585ce816342d1cd779d453d8',
        },
        '3_block': {
            'train': 'a91d6150e8b059dac86105853f377a11',
            'test': '6414af1ec33ace417e879f9c88066d47',
        },
        '3_culture_10': {
            'train': '702bc6a9368ebff4542d791e53469244',
            'test': '58335ce34ca3a18424e19da84f2832fc',
        },
    }

    classes = [
        'Compact high rise',
        'Compact mid rise',
        'Compact low rise',
        'Open high rise',
        'Open mid rise',
        'Open low rise',
        'Lightweight low rise',
        'Large low rise',
        'Sparsely built',
        'Heavy industry',
        'Dense trees',
        'Scattered trees',
        'Bush, scrub',
        'Low plants',
        'Bare rock or paved',
        'Bare soil or sand',
        'Water',
    ]

    all_s1_band_names = (
        'S1_B1',
        'S1_B2',
        'S1_B3',
        'S1_B4',
        'S1_B5',
        'S1_B6',
        'S1_B7',
        'S1_B8',
    )
    all_s2_band_names = (
        'S2_B02',
        'S2_B03',
        'S2_B04',
        'S2_B05',
        'S2_B06',
        'S2_B07',
        'S2_B08',
        'S2_B8A',
        'S2_B11',
        'S2_B12',
    )
    all_band_names = all_s1_band_names + all_s2_band_names

    rgb_bands = ['S2_B04', 'S2_B03', 'S2_B02']

    BAND_SETS = {
        'all': all_band_names,
        's1': all_s1_band_names,
        's2': all_s2_band_names,
        'rgb': rgb_bands,
    }

    def __init__(
        self,
        root: str = 'data',
        version: str = '2',
        split: str = 'train',
        bands: Sequence[str] = BAND_SETS['all'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new So2Sat dataset instance.

        Args:
            root: root directory where dataset can be found
            version: one of "2", "3_random", "3_block", or "3_culture_10"
            split: one of "train", "validation", or "test"
            bands: a sequence of band names to use where the indices correspond to the
                array index of combined Sentinel 1 and Sentinel 2
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found.
            MissingDependencyError: If h5py is not installed.

        .. versionadded:: 0.3
           The *bands* parameter.

        .. versionadded:: 0.5
           The *version* parameter.
        """
        h5py = lazy_import('h5py')

        assert version in self.versions
        assert split in self.filenames_by_version[version]

        self._validate_bands(bands)
        self.s1_band_indices: 'np.typing.NDArray[np.int_]' = np.array(
            [
                self.all_s1_band_names.index(b)
                for b in bands
                if b in self.all_s1_band_names
            ]
        ).astype(int)

        self.s1_band_names = [self.all_s1_band_names[i] for i in self.s1_band_indices]

        self.s2_band_indices: 'np.typing.NDArray[np.int_]' = np.array(
            [
                self.all_s2_band_names.index(b)
                for b in bands
                if b in self.all_s2_band_names
            ]
        ).astype(int)

        self.s2_band_names = [self.all_s2_band_names[i] for i in self.s2_band_indices]

        self.bands = bands

        self.root = root
        self.version = version
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self.fn = os.path.join(self.root, self.filenames_by_version[version][split])

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        with h5py.File(self.fn, 'r') as f:
            self.size: int = f['label'].shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        h5py = lazy_import('h5py')
        with h5py.File(self.fn, 'r') as f:
            s1 = f['sen1'][index].astype(np.float64)  # convert from <f8 to float64
            s1 = np.take(s1, indices=self.s1_band_indices, axis=2)
            s2 = f['sen2'][index].astype(np.float64)  # convert from <f8 to float64
            s2 = np.take(s2, indices=self.s2_band_indices, axis=2)

            # convert one-hot encoding to int64 then torch int
            label = torch.tensor(f['label'][index].argmax())

            s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
            s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format

            s1 = torch.from_numpy(s1)
            s2 = torch.from_numpy(s2)

        sample = {'image': torch.cat([s1, s2]).float(), 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return self.size

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        md5 = self.md5s_by_version[self.version][self.split]
        if not check_integrity(self.fn, md5 if self.checksum else None):
            return False
        return True

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            AssertionError: if ``bands`` is not a sequence
            ValueError: if an invalid band name is provided

        .. versionadded:: 0.3
        """
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.s2_band_names:
                idx = self.s2_band_names.index(band) + len(self.s1_band_names)
                rgb_indices.append(idx)
            else:
                raise RGBBandsMissingError()

        image = np.take(sample['image'].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = percentile_normalization(image, 0, 100)

        label = cast(int, sample['label'].item())
        label_class = self.classes[label]

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = cast(int, sample['prediction'].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        if show_titles:
            title = f'Label: {label_class}'
            if showing_predictions:
                title += f'\nPrediction: {prediction_class}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
