# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Biomass-S3 dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..errors import RGBBandsMissingError
from ..utils import Path, Sample, percentile_normalization, stack_samples
from .base import CopernicusBenchBase


class CopernicusBenchBiomassS3(CopernicusBenchBase):
    """Copernicus-Bench Biomass-S3 dataset.

    Biomass-S3 is a regression dataset based on Sentinel-3 OLCI images and CCI biomass.
    The biomass product is part of the European Space Agency's Climate Change Initiative
    (CCI) program and delivers global forest above-ground biomass at 100 m spatial
    resolution.

    This benchmark supports both static (1 image/location) and time series
    (1-4 images/location) modes, the former is used in the original benchmark.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://catalogue.ceda.ac.uk/uuid/02e1b18071ad45a19b4d3e8adafa2817/

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_biomass_s3/biomass_s3.zip'
    sha256 = '1d005b200d50f2e8b5f4482959bdfa6e2d7d05a8cd828d7f438c99a4e1cfbaef'
    zipfile = 'biomass_s3.zip'
    directory = 'biomass_s3'
    filename = 'static_fnames-{}.csv'
    dtype = torch.float
    filename_regex = r'S3[AB]_(?P<date>\d{8}T\d{6})'
    all_bands = (
        'Oa01_radiance',
        'Oa02_radiance',
        'Oa03_radiance',
        'Oa04_radiance',
        'Oa05_radiance',
        'Oa06_radiance',
        'Oa07_radiance',
        'Oa08_radiance',
        'Oa09_radiance',
        'Oa10_radiance',
        'Oa11_radiance',
        'Oa12_radiance',
        'Oa13_radiance',
        'Oa14_radiance',
        'Oa15_radiance',
        'Oa16_radiance',
        'Oa17_radiance',
        'Oa18_radiance',
        'Oa19_radiance',
        'Oa20_radiance',
        'Oa21_radiance',
    )
    rgb_bands = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')
    cmap = 'YlGn'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        mode: Literal['static', 'time-series'] = 'static',
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = True,
    ) -> None:
        """Initialize a new CopernicusBenchBiomassS3 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'static' or 'time-series'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, verify the checksum of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.mode = mode
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath, header=None)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        pid, file = self.files.iloc[index]
        match self.mode:
            case 'static':
                path = os.path.join(self.root, self.directory, 's3_olci', pid, file)
                sample = self._load_image(path)
            case 'time-series':
                paths = os.path.join(self.root, self.directory, 's3_olci', pid, '*.tif')
                samples = [self._load_image(path) for path in sorted(glob.glob(paths))]
                sample = stack_samples(samples)

        path = os.path.join(self.root, self.directory, 'biomass', f'{pid}.tif')
        sample |= self._load_mask(path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: Sample,
        show_titles: bool = True,
        suptitle: str | None = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.
            alpha: Opacity to use when rendering the mask (0 is transparent, 1 opaque).

        Returns:
            A matplotlib Figure with the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        try:
            rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        except ValueError as exc:
            raise RGBBandsMissingError() from exc

        image = sample['image'].detach().cpu()
        if image.dim() == 3:
            image = image.unsqueeze(0)

        rgb = image[:, rgb_indices].numpy()
        rgb = percentile_normalization(rgb)
        rgb = np.transpose(rgb, (0, 2, 3, 1))

        has_mask = 'mask' in sample
        has_prediction = 'prediction' in sample

        ncols = rgb.shape[0] + int(has_mask) + int(has_prediction)
        fig, axes = plt.subplots(
            nrows=1, ncols=ncols, figsize=(5 * ncols, 5), squeeze=False
        )
        axes_list = axes.flatten()

        for idx, img in enumerate(rgb):
            axes_list[idx].imshow(img)
            axes_list[idx].axis('off')
            if show_titles:
                title = 'Image'
                if rgb.shape[0] > 1:
                    title = f'Image {idx + 1}'
                axes_list[idx].set_title(title)

        current_col = rgb.shape[0]
        vmin = vmax = None
        if has_mask:
            mask = sample['mask'].detach().cpu().numpy().squeeze()
            vmin = float(np.nanmin(mask))
            vmax = float(np.nanmax(mask))
            mask_im = axes_list[current_col].imshow(
                mask, cmap=self.cmap, alpha=alpha, vmin=vmin, vmax=vmax
            )
            axes_list[current_col].axis('off')
            if show_titles:
                axes_list[current_col].set_title('Mask')
            fig.colorbar(mask_im, ax=axes_list[current_col], fraction=0.046, pad=0.04)
            current_col += 1

        if has_prediction:
            prediction = sample['prediction'].detach().cpu().numpy().squeeze()
            pred_im = axes_list[current_col].imshow(
                prediction,
                cmap=self.cmap,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
            axes_list[current_col].axis('off')
            if show_titles:
                axes_list[current_col].set_title('Prediction')
            fig.colorbar(pred_im, ax=axes_list[current_col], fraction=0.046, pad=0.04)

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
