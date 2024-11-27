# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Airphen dataset."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .errors import RGBBandsMissingError
from .geo import RasterDataset
from .utils import percentile_normalization


class Airphen(RasterDataset):
    """Airphen dataset.

    `Airphen <https://ideol.sakura.ne.jp/img/20170123_HiphenAirphenKeyfeatures.pdf>`__
    is a multispectral scientific camera developed by agronomists and photonics
    engineers at `Hiphen <https://www.hiphen-plant.com/>`_ to match plant measurements
    needs and constraints.

    Main characteristics:

    * 6 Synchronized global shutter sensors
    * Sensor resolution 1280 x 960 pixels
    * Data format (.tiff, 12 bit)
    * SD card storage
    * Metadata information: Exif and XMP
    * Internal or external GPS
    * Synchronization with different sensors (TIR, RGB, others)

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34133/2021/9892647

    .. versionadded:: 0.6
    """

    # Each camera measures a custom set of spectral bands chosen at purchase time.
    # Hiphen offers 8 bands to choose from, sorted from short to long wavelength.
    all_bands = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8')
    rgb_bands = ('B4', 'B3', 'B1')

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
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

        image = sample['image'][rgb_indices].permute(1, 2, 0).float()
        image = percentile_normalization(image, axis=(0, 1))

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title('Image')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
