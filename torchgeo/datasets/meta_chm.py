# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Meta Canopy Height Map (CHM) v2 dataset."""

import io
import urllib.request
from collections.abc import Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure

from .geo import RasterDataset
from .utils import Sample


class MetaCHM(RasterDataset):
    """Meta Canopy Height Map (CHM) v2 dataset.

    The `Meta CHMv2 (DINOv3) global canopy height map
    <https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/>`__ is a global,
    ~1.19 m/pixel estimate of tree canopy height derived from high-resolution satellite
    imagery.

    Dataset features:

    * canopy height in meters at ~1.19 m/pixel (uint8, 0 = no canopy or no data)
    * 213,109 tiles on a zoom-10 Web Mercator (EPSG:3857) grid

    Dataset format:

    * a STAC GeoParquet index with each tile's geometry, acquisition date, and COG URL
    * single-channel uint8 Cloud-Optimized GeoTIFFs whose pixel values are the canopy
      height in meters (0 = no canopy or no data)

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2603.06382

    .. versionadded:: 0.10
    """

    url = 'https://data.source.coop/tge-labs/meta-chm-v2/stac/items.parquet'

    is_image = False
    dtype = torch.float32
    all_bands = ('chm',)
    _res = (1.1943285669558463, 1.1943285669558463)

    def __init__(
        self, transforms: Callable[[Sample], Sample] | None = None, cache: bool = True
    ) -> None:
        """Initialize a new MetaCHM instance.

        Args:
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
        """
        self.paths = self.url
        self.transforms = transforms
        self.cache = cache
        self.bands = self.all_bands
        self.band_indexes = None
        self.time_series = False

        request = urllib.request.Request(self.url, headers={'User-Agent': 'torchgeo'})
        with urllib.request.urlopen(request) as response:
            buffer = io.BytesIO(response.read())
        columns = ['geometry', 'assets', 'datetime']
        gdf = gpd.read_parquet(str(buffer), columns=columns)
        gdf.to_crs('EPSG:3857', inplace=True)
        filepaths = (
            gdf['assets']
            .map(lambda asset: asset['chm']['href'])  # ty: ignore[not-subscriptable]
            .str.replace(
                's3://dataforgood-fb-data/',
                'https://dataforgood-fb-data.s3.amazonaws.com/',
                regex=False,
            )
        )
        datetimes = gdf['datetime'].dt.tz_localize(None)
        index = pd.IntervalIndex.from_arrays(
            datetimes, datetimes + pd.Timedelta(days=1), closed='both', name='datetime'
        )
        self.index = gpd.GeoDataFrame(
            {'filepath': filepaths.to_numpy()},
            index=index,
            geometry=gdf.geometry.to_numpy(),
            crs=gdf.crs,
        )

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample['mask'].numpy()
        ncols = 1

        showing_prediction = 'prediction' in sample
        if showing_prediction:
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs = [axs] if ncols == 1 else axs

        im = axs[0].imshow(mask, cmap='YlGn', vmin=0, vmax=40)
        axs[0].axis('off')
        fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label='Canopy height (m)')
        if show_titles:
            axs[0].set_title('Canopy Height')

        if showing_prediction:
            pred = sample['prediction'].numpy()
            im = axs[1].imshow(pred, cmap='YlGn', vmin=0, vmax=40)
            axs[1].axis('off')
            fig.colorbar(
                im, ax=axs[1], fraction=0.046, pad=0.04, label='Canopy height (m)'
            )
            if show_titles:
                axs[1].set_title('Prediction')

        if suptitle is not None:
            fig.suptitle(suptitle)
        return fig
