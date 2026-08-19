# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mixins for dataset classes."""

from typing import cast

from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .errors import RGBBandsMissingError
from .utils import (
    Sample,
    quantile_normalization,
    )

class PlottingMixin:
    """Mixin for dataset plotting.

    .. versionadded:: 0.10
    """

    #: Names of all available bands in the dataset
    all_bands: tuple[str, ...] = ()

    #: Names of RGB bands in the dataset
    rgb_bands: tuple[str, ...] = ()

    #: Color map for the dataset
    cmap: str | Colormap | None = None
    
    def plot(
        self, 
        sample: Sample, 
        show_titles: bool = True, 
        quantile_norm: bool = False, 
        time_step: int | None = None,
        suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.
        
        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            quantile_norm: flag indicating whether to apply quantile normalization
            time_step: time step at which to access image, beginning with 0
            suptitle: optional string to use as a suptitle
        
        Returns:
            a matplotlib Figure with the rendered sample
        
        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        
        .. versionadded:: 0.11
        """

        if time_step is not None:
            image = sample['image'][time_step]
        else:
            image = sample['image']
        
        if self.rgb_bands:
            rgb_indices = []
            for band in self.rgb_bands:
                if band in self.bands:
                    rgb_indices.append(self.bands.index(band))
                else:
                    raise RGBBandsMissingError()
            image = image[rgb_indices]
        else:
            image = image[:3]

        image = image.permute(1, 2, 0).float()
        
        if quantile_norm:
            image = quantile_normalization(image)

        mask_keys = [key for key in sample.keys() if key.startswith('mask') or key == ('prediction')]
        ncols = 1 + len(mask_keys)

        fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
        axs = axs[0]

        axs[0].imshow(image)
        axs[0].axis('off')
    
        if show_titles:
            title = ''
            if 'label' in sample:
                label = cast(int, sample['label'].item())
                if hasattr(self, 'classes'):
                    title = f'Label: {self.classes[label]}'
                else:
                    title = f'Label: {label}'
            elif time_step is not None:
                title += f' (t={time_step})'
            else:
                title = 'Image'
            axs[0].set_title(title)

        for i, key in enumerate(mask_keys, start=1):
            mask = sample[key]

            mask = mask.squeeze()

            axs[i].imshow(mask, interpolation='none')
            axs[i].axis('off')

            if show_titles:
                axs[i].set_title(key)
        
        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        
        return fig