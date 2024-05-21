# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT dataset."""

import os
from collections.abc import Callable, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoClassificationDataset
from .utils import check_integrity, download_url, extract_archive, rasterio_loader


class EuroSAT(NonGeoClassificationDataset):
    """EuroSAT dataset.

    The `EuroSAT <https://github.com/phelber/EuroSAT>`__ dataset is based on Sentinel-2
    satellite images covering 13 spectral bands and consists of 10 target classes with
    a total of 27,000 labeled and geo-referenced images.

    Dataset format:

    * rasters are 13-channel GeoTiffs
    * labels are values in the range [0,9]

    Dataset classes:

    * Annual Crop
    * Forest
    * Herbaceous Vegetation
    * Highway
    * Industrial Buildings
    * Pasture
    * Permanent Crop
    * Residential Buildings
    * River
    * SeaLake

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following papers:

    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248
    """

    url = 'https://hf.co/datasets/torchgeo/eurosat/resolve/06fd1b090bceecc0ce724cd21578ba7a6664fe8d/EuroSATallBands.zip'  # noqa: E501
    filename = 'EuroSATallBands.zip'
    md5 = '5ac12b3b2557aa56e1826e981e8e200e'

    # For some reason the class directories are actually nested in this directory
    base_dir = os.path.join(
        'ds', 'images', 'remote_sensing', 'otherDatasets', 'sentinel_2', 'tif'
    )

    splits = ['train', 'val', 'test']
    split_urls = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt',  # noqa: E501
        'val': 'https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt',  # noqa: E501
        'test': 'https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt',  # noqa: E501
    }
    split_md5s = {
        'train': '908f142e73d6acdf3f482c5e80d851b1',
        'val': '95de90f2aa998f70a3b2416bfe0687b4',
        'test': '7ae5ab94471417b6e315763121e67c5f',
    }

    all_band_names = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B10',
        'B11',
        'B12',
    )

    rgb_bands = ('B04', 'B03', 'B02')

    BAND_SETS = {'all': all_band_names, 'rgb': rgb_bands}

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        bands: Sequence[str] = BAND_SETS['all'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: a sequence of band names to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.3
           The *bands* parameter.
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = Tensor(
            [self.all_band_names.index(b) for b in bands if b in self.all_band_names]
        ).long()

        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f'eurosat-{split}.txt')) as f:
            for fn in f:
                valid_fns.add(fn.strip().replace('.jpg', '.tif'))

        def is_in_split(x: str) -> bool:
            return os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
            loader=rasterio_loader,
            is_valid_file=is_in_split,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)

        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )
        return integrity

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        filepath = os.path.join(self.root, self.base_dir)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        if self._check_integrity():
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
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f'eurosat-{split}.txt',
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

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
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
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
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = np.take(sample['image'].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = np.clip(image / 3000, 0, 1)

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


class EuroSAT100(EuroSAT):
    """Subset of EuroSAT containing only 100 images.

    Intended for tutorials and demonstrations, not for benchmarking.

    Maintains the same file structure, classes, and train-val-test split. Each class has
    10 images (6 train, 2 val, 2 test), for a total of 100 images.

    .. versionadded:: 0.5
    """

    url = 'https://hf.co/datasets/torchgeo/eurosat/resolve/06fd1b090bceecc0ce724cd21578ba7a6664fe8d/EuroSAT100.zip'  # noqa: E501
    filename = 'EuroSAT100.zip'
    md5 = 'c21c649ba747e86eda813407ef17d596'

    split_urls = {
        'train': 'https://hf.co/datasets/torchgeo/eurosat/raw/06fd1b090bceecc0ce724cd21578ba7a6664fe8d/eurosat-train.txt',  # noqa: E501
        'val': 'https://hf.co/datasets/torchgeo/eurosat/raw/06fd1b090bceecc0ce724cd21578ba7a6664fe8d/eurosat-val.txt',  # noqa: E501
        'test': 'https://hf.co/datasets/torchgeo/eurosat/raw/06fd1b090bceecc0ce724cd21578ba7a6664fe8d/eurosat-test.txt',  # noqa: E501
    }
    split_md5s = {
        'train': '033d0c23e3a75e3fa79618b0e35fe1c7',
        'val': '3e3f8b3c344182b8d126c4cc88f3f215',
        'test': 'f908f151b950f270ad18e61153579794',
    }


class SpatialSplitEuroSAT(EuroSAT):
    """A subclass of the EuroSAT dataset class overriding the default splits.

    Split 60/20/20 by longitude.

    Attributes:
        split_urls (dict): A dictionary mapping each split to its corresponding URL.
        split_md5s (dict): A dictionary mapping each split to its corresponding MD5.
    """
    
    split_urls = {
        "train": "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-spatial-train.txt?download=true",  # noqa: E501
        "val": "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-spatial-val.txt?download=true",  # noqa: E501
        "test": "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-spatial-test.txt?download=true",  # noqa: E501 
    }
    split_md5s = {
        "train": "7be3254be39f23ce4d4d144290c93292",
        "val": "acf392290050bb3df790dc8fc0ebf193",
        "test": "5ec1733f9c16116bf0aa2d921fc613ef",
    }