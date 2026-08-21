# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mixins for dataset classes."""

from typing import cast

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from einops import rearrange

from .errors import RGBBandsMissingError
from .utils import Sample, quantile_normalization

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
        suptitle: str | None = None
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
        
        .. versionadded:: 0.11
        """

        image = sample['image']
        
        if self.rgb_bands:
            rgb_indices = []
            for band in self.rgb_bands:
                if band in self.bands:
                    rgb_indices.append(self.bands.index(band))
                else:
                    raise RGBBandsMissingError()
            image = image[rgb_indices]

        image = rearrange(image, 'c h w -> h w c')
        image = image.float()

        image = quantile_normalization(image)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            title = ''
            if 'label' in sample:
                label = cast(int, sample['label'].item())
                if hasattr(self, 'classes'):
                    title += f'Label: {self.classes[label]}'
                else:
                    title += f'Label: {label}'

                if 'prediction' in sample:
                    prediction = cast(int, sample['prediction'].item())
                    if hasattr(self, 'classes'):
                        title += f'\nPrediction: {self.classes[prediction]}'
                    else:
                        title += f'\nPrediction: {prediction}'

            else:
                title = 'Image'

            ax.set_title(title)
        
        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        
        return fig