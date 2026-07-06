# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""AgriFieldNet India Challenge dataset."""

import os
from collections.abc import Callable, Iterable, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import IntersectionDataset, RasterDataset
from .utils import GeoSlice, Path, Sample, quantile_normalization, which


class AgriFieldNetImage(RasterDataset):
    """AgriFieldNet Sentinel-2 imagery.

    .. versionadded:: 0.10
    """

    filename_glob = 'ref_agrifieldnet_competition_v1_source_*_{}_10m.*'
    filename_regex = r"""
        ^ref_agrifieldnet_competition_v1_source_
        (?P<unique_folder_id>[a-f0-9]{5})
        _(?P<band>B[0-9A]{2})_10m
    """
    separate_files = True

    rgb_bands = ('B04', 'B03', 'B02')
    all_bands = (
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
    )


class AgriFieldNetMask(RasterDataset):
    """AgriFieldNet masks.

    .. versionadded:: 0.10
    """

    filename_glob = 'ref_agrifieldnet_competition_v1_labels_*'
    filename_regex = r"""
        ^ref_agrifieldnet_competition_v1_labels_
        (?P<split>train|test)_
        (?P<unique_folder_id>[a-f0-9]{5})\.
    """
    is_image = False

    cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 255),
                (255, 211, 0, 255),
                (255, 37, 37, 255),
                (0, 168, 226, 255),
                (255, 158, 9, 255),
                (37, 111, 0, 255),
                (255, 255, 0, 255),
                (0, 0, 0, 255),
                (111, 166, 0, 255),
                (0, 175, 73, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (222, 166, 9, 255),
                (222, 166, 9, 255),
                (124, 211, 255, 255),
                (226, 0, 124, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (137, 96, 83, 255),
            ]
        )
        / 255
    )

    valid_classes = (0, 1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36)


class AgriFieldNet(IntersectionDataset):
    """AgriFieldNet India Challenge dataset.

    The `AgriFieldNet India Challenge
    <https://zindi.africa/competitions/agrifieldnet-india-challenge>`__ dataset
    includes satellite imagery from Sentinel-2 cloud free composites
    (single snapshot) and labels for crop type that were collected by ground survey.
    The Sentinel-2 data are then matched with corresponding labels.
    The dataset contains 7081 fields, which have been split into training and
    test sets (5551 fields in the train and 1530 fields in the test).
    Satellite imagery and labels are tiled into 256x256 chips adding up to 1217 tiles.
    The fields are distributed across all chips, some chips may only have train or
    test fields and some may have both. Since the labels are derived from data
    collected on the ground, not all the pixels are labeled in each chip.
    If the field ID for a pixel is set to 0 it means that pixel is not included in
    either of the train or test set (and correspondingly the crop label
    will be 0 as well). For this challenge train and test sets have slightly
    different crop type distributions. The train set follows the distribution
    of ground reference data which is a skewed distribution with a few dominant
    crops being over represented. The test set was drawn randomly from an area
    weighted field list that ensured that fields with less common crop types
    were better represented in the test set. The original dataset can be
    downloaded from `Source Cooperative <https://source.coop/
    radiantearth/agrifieldnet-competition>`__.

    Dataset format:

    * images are 12-band Sentinel-2 data
    * masks are tiff images with unique values representing the class and field id

    Dataset classes:

    * 0. No-Data
    * 1. Wheat
    * 2. Mustard
    * 3. Lentil
    * 4. No Crop/Fallow
    * 5. Green pea
    * 6. Sugarcane
    * 8. Garlic
    * 9. Maize
    * 13. Gram
    * 14. Coriander
    * 15. Potato
    * 16. Berseem
    * 36. Rice

    If you use this dataset in your research, please cite the following dataset:

    * https://doi.org/10.34911/rdnt.wu92p1

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.6
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/ref_agrifieldnet_competition_v1'

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        classes: list[int] = list(AgriFieldNetMask.valid_classes),
        bands: Sequence[str] = AgriFieldNetImage.all_bands,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        time_series: bool = False,
    ) -> None:
        """Initialize a new AgriFieldNet dataset instance.

        Args:
            paths: one or more root directories to search for files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            classes: list of classes to include, the rest will be mapped to 0
                (defaults to all classes)
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            cache: if True, cache the dataset in memory
            download: if True, download dataset and store it in the root directory
            time_series: if True, stack data along the time series dimension
                [T, C, H, W]. If False, merge data into a [C, H, W] mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.10
           The *res* parameter.

        .. versionadded:: 0.9
           The *time_series* parameter.
        """
        assert set(classes) <= set(AgriFieldNetMask.valid_classes), (
            f'Only the following classes are valid: {AgriFieldNetMask.valid_classes}.'
        )
        assert 0 in classes, 'Classes must include the background class: 0'

        self.paths = paths
        self.download = download
        AgriFieldNetImage.filename_glob = AgriFieldNetImage.filename_glob.format(
            bands[0]
        )

        self._verify()

        self.image = AgriFieldNetImage(
            paths, crs, res, bands, transforms, cache, time_series
        )
        self.mask = AgriFieldNetMask(
            paths, crs, res, None, transforms, cache, time_series
        )

        super().__init__(self.image, self.mask)

        # Ignore unintentional partial overlap
        self.index = self.image.index

        # Map chosen classes to ordinal numbers, all others mapped to background class
        self.ordinal_map = torch.zeros(
            self.mask.valid_classes[-1] + 1, dtype=self.mask.dtype
        )
        self.inverse_map = torch.zeros(len(classes), dtype=self.mask.dtype)
        for v, k in enumerate(classes):
            self.ordinal_map[k] = v
            self.inverse_map[v] = k

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        sample = super().__getitem__(index)
        sample['mask'] = self.ordinal_map[sample['mask']]
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
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        os.makedirs(paths, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', f'{self.url}', paths, '--recursive=true')

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

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.image.rgb_bands:
            if band in self.image.bands:
                rgb_indices.append(self.image.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0)
        image = quantile_normalization(image)

        mask = self.inverse_map[sample['mask']]
        ncols = 2

        showing_prediction = 'prediction' in sample
        if showing_prediction:
            pred = self.inverse_map[sample['prediction']]
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        kwargs = {
            'cmap': self.cmap,
            'vmin': self.valid_classes[0],
            'vmax': self.valid_classes[-1],
            'interpolation': 'none',
        }

        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, **kwargs)
        axs[1].axis('off')
        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_prediction:
            axs[2].imshow(pred, **kwargs)
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
