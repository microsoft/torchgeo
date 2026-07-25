# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Landslide4Sense benchmark dataset."""

from collections.abc import Callable, Sequence
from pathlib import Path as pathlibPath
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .errors import RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import Path, Sample, lazy_import


class Landslide4Sense(NonGeoDataset):
    """Landslide4Sense benchmark dataset.

    The Landslide4Sense dataset is a benchmark for landslide detection from
    multisource satellite imagery. Each sample is a 14-channel image patch of
    shape ``128 x 128 x 14`` containing Sentinel-2 bands B1-B12 together with
    ALOS PALSAR-derived slope and DEM channels.

    The public release contains the following splits:

    * ``train``: images and masks
    * ``val``: images only
    * ``test``: images only

     .. note::

         In the public release, only the training split includes masks.

         The public dataset can be downloaded from:
         https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense

         Secondary source:
         https://github.com/iarai/Landslide4Sense-2022/tree/main

         Additional dependencies:

         * ``h5py`` for reading the dataset files.

     If you use this dataset in your research, please cite it using the following
     link:

     * https://arxiv.org/abs/2206.00515

    .. versionadded:: 0.10.0

    """

    splits = ('train', 'val', 'test')

    band_names = (
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B9',
        'B10',
        'B11',
        'B12',
        'Slope',
        'DEM',
    )
    rgb_bands = ('B4', 'B3', 'B2')

    def __init__(
        self,
        root: str | Path,
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        bands: Sequence[int] | None = None,
    ) -> None:
        """Initialize a new Landslide4Sense dataset instance.

        Args:
            root: Root directory where dataset files are stored.
            split: Dataset split to load.
            transforms: Transform function applied to each sample.
            bands: Optional subset of band indices to load from each image.

        Raises:
            ValueError: If ``split`` is not one of the supported split names.
            FileNotFoundError: If required split files cannot be found.

        """
        lazy_import('h5py')

        if split not in self.splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.splits}.")

        self.root = pathlibPath(root)
        self.split = split
        self.transforms = transforms
        self.bands = bands
        self.samples = self._load_samples()

    def _split_dirs(self) -> tuple[Path, Path | None]:
        """Return image and mask directories for the selected split."""
        mapping = {
            'train': ('TrainData/img', 'TrainData/mask'),
            'val': ('ValidData/img', None),
            'test': ('TestData/img', None),
        }

        img_rel, mask_rel = mapping[self.split]
        img_dir = self.root / img_rel

        if not img_dir.exists():
            raise FileNotFoundError(f'Image directory not found: {img_dir}')

        if mask_rel is None:
            return img_dir, None

        mask_dir = self.root / mask_rel
        if not mask_dir.exists():
            raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

        return img_dir, mask_dir

    def _load_samples(self) -> list[tuple[Path, Path | None]]:
        """Load sample file paths for the selected split."""
        img_dir, mask_dir = self._split_dirs()
        image_paths = sorted(img_dir.glob('*.h5'))

        if not image_paths:
            raise FileNotFoundError(f'No .h5 files found in image directory: {img_dir}')

        samples: list[tuple[Path, Path | None]] = []

        for image_path in image_paths:
            if mask_dir is None:
                samples.append((image_path, None))
                continue

            mask_name = image_path.name.replace('image_', 'mask_')
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                raise FileNotFoundError(f'Missing mask for image: {image_path}')

            samples.append((image_path, mask_path))

        return samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        """Return a single sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A sample dictionary containing an ``image`` tensor and, for the
            training split, a ``mask`` tensor.

        """
        h5py = lazy_import('h5py')

        image_path, mask_path = self.samples[index]

        with h5py.File(image_path, 'r') as f:
            image = f['img'][:]

        image = np.transpose(image, (2, 0, 1))

        if self.bands is not None:
            image = image[self.bands, :, :]

        sample: Sample = {'image': torch.from_numpy(image).float()}

        if mask_path is not None:
            with h5py.File(mask_path, 'r') as f:
                mask = f['mask'][:]

            sample['mask'] = torch.from_numpy(mask).unsqueeze(0).long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: Sample dictionary containing ``image`` and optional ``mask``
                and ``prediction`` tensors.
            show_titles: If ``True``, show subplot titles.
            suptitle: Optional overall figure title.

        Returns:
            The matplotlib figure containing the rendered sample.

        """
        if self.bands is None:
            available_band_indices = list(range(len(self.band_names)))
        else:
            available_band_indices = list(self.bands)

        rgb_indices = []
        for band in self.rgb_bands:
            full_index = self.band_names.index(band)
            if full_index in available_band_indices:
                rgb_indices.append(available_band_indices.index(full_index))
            else:
                raise RGBBandsMissingError()

        has_mask = 'mask' in sample
        has_prediction = 'prediction' in sample

        ncols = 1 + int(has_mask) + int(has_prediction)
        fig, axs = plt.subplots(1, ncols, figsize=(8 * ncols, 8))

        if ncols == 1:
            axs = [axs]

        image = sample['image'][rgb_indices].permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        axs[0].imshow(image)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('Image')

        col = 1
        if has_mask:
            axs[col].imshow(sample['mask'].squeeze(0))
            axs[col].axis('off')
            if show_titles:
                axs[col].set_title('Mask')
            col += 1

        if has_prediction:
            axs[col].imshow(sample['prediction'].squeeze(0))
            axs[col].axis('off')
            if show_titles:
                axs[col].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
