# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Smallholder Cashew Plantations in Benin dataset."""

import json
import os
from collections.abc import Callable, Sequence
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import Path, which


class BeninSmallHolderCashews(NonGeoDataset):
    r"""Smallholder Cashew Plantations in Benin dataset.

    This dataset contains labels for cashew plantations in a 120 km\ :sup:`2`\  area
    in the center of Benin. Each pixel is classified for Well-managed plantation,
    Poorly-managed plantation, No plantation and other classes. The labels are
    generated using a combination of ground data collection with a handheld GPS device,
    and final corrections based on Airbus Pléiades imagery. See `this website
    <https://beta.source.coop/technoserve/cashews-benin/>`__ for dataset details.

    Specifically, the data consists of Sentinel 2 imagery from a 120 km\ :sup:`2`\  area
    in the center of Benin over 71 points in time from 11/05/2019 to 10/30/2020
    and polygon labels for 6 classes:

    0. No data
    1. Well-managed plantation
    2. Poorly-managed planatation
    3. Non-plantation
    4. Residential
    5. Background
    6. Uncertain

    If you use this dataset in your research, please cite the following:

    * https://beta.source.coop/technoserve/cashews-benin/

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/technoserve-cashew-benin'
    dates = (
        '20191105',
        '20191110',
        '20191115',
        '20191120',
        '20191130',
        '20191205',
        '20191210',
        '20191215',
        '20191220',
        '20191225',
        '20191230',
        '20200104',
        '20200109',
        '20200114',
        '20200119',
        '20200124',
        '20200129',
        '20200208',
        '20200213',
        '20200218',
        '20200223',
        '20200228',
        '20200304',
        '20200309',
        '20200314',
        '20200319',
        '20200324',
        '20200329',
        '20200403',
        '20200408',
        '20200413',
        '20200418',
        '20200423',
        '20200428',
        '20200503',
        '20200508',
        '20200513',
        '20200518',
        '20200523',
        '20200528',
        '20200602',
        '20200607',
        '20200612',
        '20200617',
        '20200622',
        '20200627',
        '20200702',
        '20200707',
        '20200712',
        '20200717',
        '20200722',
        '20200727',
        '20200801',
        '20200806',
        '20200811',
        '20200816',
        '20200821',
        '20200826',
        '20200831',
        '20200905',
        '20200910',
        '20200915',
        '20200920',
        '20200925',
        '20200930',
        '20201010',
        '20201015',
        '20201020',
        '20201025',
        '20201030',
    )

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
        'CLD',
    )
    rgb_bands = ('B04', 'B03', 'B02')

    classes = [
        'No data',
        'Well-managed planatation',
        'Poorly-managed planatation',
        'Non-planatation',
        'Residential',
        'Background',
        'Uncertain',
    ]

    # Same for all tiles
    tile_height = 1186
    tile_width = 1122

    def __init__(
        self,
        root: Path = 'data',
        chip_size: int = 256,
        stride: int = 128,
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new Benin Smallholder Cashew Plantations Dataset instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
            stride: spacing between chips, if less than chip_size, then there
                will be overlap between chips
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: If *bands* is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.chip_size = chip_size
        self.stride = stride
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self._verify()

        # Calculate the indices that we will use over all tiles
        self.chips_metadata = []
        for y in list(range(0, self.tile_height - self.chip_size, stride)) + [
            self.tile_height - self.chip_size
        ]:
            for x in list(range(0, self.tile_width - self.chip_size, stride)) + [
                self.tile_width - self.chip_size
            ]:
                self.chips_metadata.append((y, x))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            a dict containing image, mask, transform, crs, and metadata at index.
        """
        y, x = self.chips_metadata[index]

        img, transform, crs = self._load_all_imagery()
        labels = self._load_mask(transform)

        img = img[:, :, y : y + self.chip_size, x : x + self.chip_size]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            'image': img,
            'mask': labels,
            'x': torch.tensor(x),
            'y': torch.tensor(y),
            'transform': transform,
            'crs': crs,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chips_metadata)

    @lru_cache(maxsize=128)
    def _load_all_imagery(self) -> tuple[Tensor, rasterio.Affine, CRS]:
        """Load all the imagery (across time) for the dataset.

        Returns:
            imagery of shape (70, number of bands, 1186, 1122) where 70 is the number
            of points in time, 1186 is the tile height, and 1122 is the tile width
            rasterio affine transform, mapping pixel coordinates to geo coordinates
            coordinate reference system of transform
        """
        img = torch.zeros(
            len(self.dates),
            len(self.bands),
            self.tile_height,
            self.tile_width,
            dtype=torch.float32,
        )

        for date_index, date in enumerate(self.dates):
            single_scene, transform, crs = self._load_single_scene(date)
            img[date_index] = single_scene

        return img, transform, crs

    @lru_cache(maxsize=128)
    def _load_single_scene(self, date: str) -> tuple[Tensor, rasterio.Affine, CRS]:
        """Load the imagery for a single date.

        Args:
            date: date of the imagery to load

        Returns:
            Tensor containing a single image tile, rasterio affine transform,
            mapping pixel coordinates to geo coordinates, and coordinate
            reference system of transform.
        """
        img = torch.zeros(
            len(self.bands), self.tile_height, self.tile_width, dtype=torch.float32
        )
        for band_index, band_name in enumerate(self.bands):
            filepath = os.path.join(
                self.root,
                'imagery',
                '00',
                f'00_{date}',
                f'00_{date}_{band_name}_10m.tif',
            )
            with rasterio.open(filepath) as src:
                transform = src.transform  # same transform for every band
                crs = src.crs
                array = src.read().astype(np.float32)
                img[band_index] = torch.from_numpy(array)

        return img, transform, crs

    @lru_cache
    def _load_mask(self, transform: rasterio.Affine) -> Tensor:
        """Rasterizes the dataset's labels (in geojson format)."""
        # Create a mask layer out of the geojson
        with open(os.path.join(self.root, 'labels', '00.geojson')) as f:
            geojson = json.load(f)

        labels = [
            (feature['geometry'], feature['properties']['class'])
            for feature in geojson['features']
        ]

        mask_data = rasterio.features.rasterize(
            labels,
            out_shape=(self.tile_height, self.tile_width),
            fill=0,  # nodata value
            transform=transform,
            all_touched=False,
            dtype=np.uint8,
        )

        mask = torch.from_numpy(mask_data).long()
        return mask

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, 'labels', '00.geojson')):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', self.url, self.root, '--recursive=true')

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        time_step: int = 0,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            time_step: time step at which to access image, beginning with 0
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

        image = np.rollaxis(sample['image'][time_step, rgb_indices].numpy(), 0, 3)
        image = np.clip(image / 3000, 0, 1)
        mask = sample['mask'].numpy()

        num_panels = 2
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy()
            num_panels += 1

        fig, axs = plt.subplots(ncols=num_panels, figsize=(4 * num_panels, 4))

        axs[0].imshow(image)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title(f't={time_step}')

        axs[1].imshow(mask, vmin=0, vmax=6, interpolation='none')
        axs[1].axis('off')
        if show_titles:
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=6, interpolation='none')
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
