# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mixins for dataset classes."""

from collections.abc import Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

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
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`NonGeoDataset.__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        dataset = getattr(self, 'dataset', self)

        rgb_indices = []
        for band in dataset.rgb_bands:
            if band in dataset.bands:
                rgb_indices.append(dataset.bands.index(band))
            else:
                raise RGBBandsMissingError()

        # Static -> time series
        images = sample['image']
        if images.dim() == 3:
            images = torch.unsqueeze(images, dim=0)

        ncols = len(images)
        if 'mask' in sample:
            ncols += 1
            if 'prediction' in sample:
                ncols += 1

        fig, ax = plt.subplots(ncols=ncols, squeeze=False)

        # Label
        title = 'Image'
        if 'label' in sample:
            if sample['label'].dim() == 0:
                # Multiclass classification
                label: Any = dataset.classes[sample['label']]
                if 'prediction' in sample:
                    prediction: Any = dataset.classes[sample['prediction']]
            else:
                # Multilabel classification
                label = sample['label'].numpy().nonzero()[0]
                if 'prediction' in sample:
                    prediction = sample['prediction'].numpy().nonzero()[0]

            title = f'Label: {label}'
            if 'prediction' in sample:
                title += f'\nPrediction: {prediction}'

        # Image
        if dataset.rgb_bands:
            images = images[:, rgb_indices]
            if set(dataset.rgb_bands) <= {'VV', 'VH', 'HH', 'HV'}:
                # SAR
                vv = images[:, 0]
                vh = images[:, 1]
                images = torch.stack([vv, vh, (vv + vh) / 2], dim=1)
                images = quantile_normalization(images)
        else:
            images = images[:, :3]

        images = quantile_normalization(images)
        images = rearrange(images, 't c h w -> t h w c')
        for i in range(len(images)):
            ax[0, i].imshow(images[i])
            ax[0, i].axis('off')
            if show_titles:
                ax[0, i].set_title(title)

        # Mask
        if 'mask' in sample:
            kwargs: dict[str, Any] = {'cmap': dataset.cmap}
            if hasattr(dataset, 'classes'):
                # Semantic segmentation
                kwargs |= {
                    'vmin': 0,
                    'vmax': len(cast(Sequence[object], dataset.classes)) - 1,
                    'interpolation': 'none',
                }
            mask = sample['mask']
            ax[0, i + 1].imshow(mask, **kwargs)
            ax[0, i + 1].axis('off')
            if show_titles:
                ax[0, i + 1].set_title('Mask')

            if 'prediction' in sample:
                prediction = sample['prediction']
                ax[0, i + 2].imshow(prediction, **kwargs)
                ax[0, i + 2].axis('off')
                if show_titles:
                    ax[0, i + 2].set_title('Prediction')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        return fig
