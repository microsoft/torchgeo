# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""South Africa Crop Type Competition Dataset."""

import os
import pathlib
import re
from collections.abc import Callable, Iterable
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import RasterDataset
from .utils import BoundingBox, Path, which


class SouthAfricaCropType(RasterDataset):
    """South Africa Crop Type Challenge dataset.

    The `South Africa Crop Type Challenge
    <https://beta.source.coop/repositories/radiantearth/south-africa-crops-competition/description/>`__
    dataset includes satellite imagery from Sentinel-1 and Sentinel-2 and labels for
    crop type that were collected by aerial and vehicle survey from May 2017 to March
    2018. Data was provided by the Western Cape Department of Agriculture and is
    available via the Radiant Earth Foundation. For each field id the dataset contains
    time series imagery and a single label mask. Since TorchGeo does not yet support
    timeseries datasets, the first available imagery in July will be returned for each
    field. Note that the dates for S1 and S2 imagery for a given field are not
    guaranteed to be the same. Due to this date mismatch only S1 or S2 bands may be
    queried at a time, a mix of both is not supported. Each pixel in the label
    contains an integer field number and crop type class.

    Dataset format:

    * images are 2-band Sentinel 1 and 12-band Sentinel-2 data with a cloud mask
    * masks are tiff images with unique values representing the class and field id.

    Dataset classes:

    0. No Data
    1. Lucerne/Medics
    2. Planted pastures (perennial)
    3. Fallow
    4. Wine grapes
    5. Weeds
    6. Small grain grazing
    7. Wheat
    8. Canola
    9. Rooibos

    If you use this dataset in your research, please cite the following dataset:

    * Western Cape Department of Agriculture, Radiant Earth Foundation (2021)
      "Crop Type Classification Dataset for Western Cape, South Africa",
      Version 1.0, Radiant MLHub, https://doi.org/10.34911/rdnt.j0co8q

    .. note::
       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.6
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/ref-south-africa-crops-competition-v1'

    filename_glob = '*_07_*_{}_10m.*'
    filename_regex = r"""
        ^(?P<field_id>\d+)
        _(?P<date>\d{4}_07_\d{2})
        _(?P<band>[BHV\d]+)
        _10m
    """
    date_format = '%Y_%m_%d'
    rgb_bands = ['B04', 'B03', 'B02']
    s1_bands = ['VH', 'VV']
    s2_bands = [
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
        'B11',
        'B12',
    ]
    all_bands: list[str] = s1_bands + s2_bands
    cmap = {
        0: (0, 0, 0, 255),
        1: (255, 211, 0, 255),
        2: (255, 37, 37, 255),
        3: (0, 168, 226, 255),
        4: (255, 158, 9, 255),
        5: (37, 111, 0, 255),
        6: (255, 255, 0, 255),
        7: (222, 166, 9, 255),
        8: (111, 166, 0, 255),
        9: (0, 175, 73, 255),
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        classes: list[int] = list(cmap.keys()),
        bands: list[str] = s2_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new South Africa Crop Type dataset instance.

        Args:
            paths: paths directory where dataset can be found
            crs: coordinate reference system to be used
            classes: crop type classes to be included
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert (
            set(classes) <= self.cmap.keys()
        ), f'Only the following classes are valid: {list(self.cmap.keys())}.'
        assert 0 in classes, 'Classes must include the background class: 0'

        self.paths = paths
        self.download = download
        self.filename_glob = self.filename_glob.format(bands[0])

        self._verify()

        super().__init__(paths=paths, crs=crs, bands=bands, transforms=transforms)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(classes), 4), dtype=torch.uint8)
        for v, k in enumerate(classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            data and labels at that index
        """
        assert isinstance(self.paths, str | pathlib.Path)

        # Get all files matching the given query
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[Path], [hit.object['filepath'] for hit in hits])

        if not filepaths:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        data_list: list[Tensor] = []
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        # Loop through matched filepaths and find all unique field ids
        field_ids: list[str] = []
        # Store date in July for s1 and s2 we want to use for each sample
        imagery_dates: dict[str, dict[str, str]] = {}

        for filepath in filepaths:
            filename = os.path.basename(filepath)
            match = re.match(filename_regex, filename)
            if match:
                field_id = match.group('field_id')
                date = match.group('date')
                band = match.group('band')
                band_type = 's1' if band in self.s1_bands else 's2'
                if field_id not in field_ids:
                    field_ids.append(field_id)
                    imagery_dates[field_id] = {'s1': '', 's2': ''}
                if (
                    date.split('_')[1] == '07'
                    and not imagery_dates[field_id][band_type]
                ):
                    imagery_dates[field_id][band_type] = date

        # Create Tensors for each band using stored dates
        for band in self.bands:
            band_type = 's1' if band in self.s1_bands else 's2'
            band_filepaths = []
            for field_id in field_ids:
                date = imagery_dates[field_id][band_type]
                filepath = os.path.join(
                    self.paths,
                    'train',
                    'imagery',
                    band_type,
                    field_id,
                    date,
                    f'{field_id}_{date}_{band}_10m.tif',
                )
                band_filepaths.append(filepath)
            data_list.append(self._merge_files(band_filepaths, query))
        image = torch.cat(data_list)

        # Add labels for each field
        mask_filepaths: list[str] = []
        for field_id in field_ids:
            file_path = filepath = os.path.join(
                self.paths, 'train', 'labels', f'{field_id}.tif'
            )
            mask_filepaths.append(file_path)

        mask = self._merge_files(mask_filepaths, query)

        sample = {
            'crs': self.crs,
            'bounds': query,
            'image': image.float(),
            'mask': mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if self.files:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        assert isinstance(self.paths, str | pathlib.Path)
        os.makedirs(self.paths, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', f'{self.url}', self.paths, '--recursive=true')

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
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample['mask'].squeeze()
        ncols = 2

        showing_prediction = 'prediction' in sample
        if showing_prediction:
            pred = sample['prediction'].squeeze()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(self.ordinal_cmap[mask], interpolation='none')
        axs[1].axis('off')
        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_prediction:
            axs[2].imshow(pred)
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
