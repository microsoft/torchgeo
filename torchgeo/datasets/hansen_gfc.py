# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Hansen Global Forest Change dataset."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .geo import RasterDataset
from .utils import Sample


class HansenGlobalForestChange(RasterDataset):
    """Hansen Global Forest Change (GFC) dataset.

    The `Hansen Global Forest Change
    <https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13>`__
    dataset provides global 30 m maps of year-2000 tree cover percentage,
    annual forest loss (2001-2024, plus a partial 2025), the year of loss,
    forest gain (2000-2012), and a land/water/no-data mask, derived from
    time-series analysis of Landsat imagery.

    Dataset features:

    * Global coverage at 30 m per pixel resolution
    * Annual forest loss attribution via a per-pixel loss year

    Dataset format:

    * Single-band GeoTIFFs, one file per band, as produced by an Earth
      Engine export via :func:`ee.batch.Export.image.toDrive`

    Dataset bands:

    * treecover2000: percent tree cover in year 2000
    * loss: binary forest loss 2001-2024
    * lossyear: year of loss (1-24, corresponding to 2001-2024)
    * gain: binary forest gain 2000-2012
    * datamask: 1 = land, 2 = water, 0 = no data

    No download is provided; users export the bands they need from the
    Earth Engine catalog asset above (see this project's
    ``setup_data_loading.export_layers`` for an example export call) and
    point this dataset at the resulting directory.

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1126/science.1244693

    .. versionadded:: 0.8
    """

    filename_glob = 'hansen_*.tif'
    filename_regex = r"""
        ^hansen
        _(?P<band>[a-z0-9]+)
        _(?P<region>\w+)\.tif$
    """

    is_image = False
    separate_files = True

    all_bands = ('treecover2000', 'loss', 'lossyear', 'gain', 'datamask')

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
        mask = sample['mask'][0]

        showing_predictions = 'prediction' in sample
        ncols = 2 if showing_predictions else 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        if ncols == 1:
            axs = [axs]

        axs[0].imshow(mask, cmap='YlOrRd')
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('Ground Truth')

        if showing_predictions:
            pred = sample['prediction'][0]
            axs[1].imshow(pred, cmap='YlOrRd')
            axs[1].axis('off')
            if show_titles:
                axs[1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
