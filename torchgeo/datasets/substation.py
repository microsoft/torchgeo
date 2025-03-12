# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Substation segmentation dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_url, extract_archive


class Substation(NonGeoDataset):
    """Substation dataset.

    The `Substation <https://github.com/Lindsay-Lab/substation-seg>`__
    dataset is curated by TransitionZero and sourced from publicly
    available data repositories, including OpenSreetMap (OSM) and
    Copernicus Sentinel data. The dataset consists of Sentinel-2
    images from 27k+ locations; the task is to segment power-substations,
    which appear in the majority of locations in the dataset.
    Most locations have 4-5 images taken at different timepoints
    (i.e., revisits).

    Dataset Format:

    * .npz file for each datapoint

    Dataset Features:

    * 26,522 image-mask pairs stored as numpy files.
    * Data from 5 revisits for most locations.
    * Multi-temporal, multi-spectral images (13 channels) paired with masks,
      with a spatial resolution of 228x228 pixels

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2409.17363
    """

    directory = 'Substation'
    filename_images = 'image_stack.tar.gz'
    filename_masks = 'mask.tar.gz'
    url_for_images = 'https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/image_stack.tar.gz'
    url_for_masks = 'https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/mask.tar.gz'
    md5_images = '948706609864d0283f74ee7015f9d032'
    md5_masks = 'baa369ececdc2ff80e6ba2b4c7fe147c'

    def __init__(
        self,
        root: Path = 'data',
        bands: Sequence[int] = tuple(range(13)),
        mask_2d: bool = True,
        num_of_timepoints: int = 4,
        timepoint_aggregation: Literal['concat', 'median', 'first', 'random']
        | None = 'concat',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize the Substation.

        Args:
            root: Path to the directory containing the dataset.
            bands: Channels to use from the image.
            mask_2d: Whether to use a 2D mask.
            num_of_timepoints: Number of timepoints to use for each image.
            timepoint_aggregation: How to aggregate multiple timepoints.
            transforms: A transform takes input sample and returns a transformed version.
            download: Whether to download the dataset if it is not found.
            checksum: Whether to verify the dataset after downloading.
        """
        self.root = root
        self.bands = bands
        self.mask_2d = mask_2d
        self.num_of_timepoints = num_of_timepoints
        self.timepoint_aggregation = timepoint_aggregation
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.image_dir = os.path.join(root, 'image_stack')
        self.mask_dir = os.path.join(root, 'mask')
        self._verify()
        self.image_filenames = pd.Series(sorted(os.listdir(self.image_dir)))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Get an item from the dataset by index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A dictionary containing the image and corresponding mask.
        """
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, image_filename)

        image = np.load(image_path)['arr_0']

        # selecting channels
        image = image[:, self.bands, :, :]

        # handling multiple images across timepoints
        if image.shape[0] < self.num_of_timepoints:
            # Padding: cycle through existing timepoints
            padded_images = []
            for i in range(self.num_of_timepoints):
                padded_images.append(image[i % image.shape[0]])
            image = np.stack(padded_images)
        elif image.shape[0] > self.num_of_timepoints:
            # Removal: take the most recent timepoints
            image = image[-self.num_of_timepoints :]

        match self.timepoint_aggregation:
            case 'concat':
                # (num_of_timepoints*channels, h, w)
                image = np.reshape(image, (-1, image.shape[2], image.shape[3]))
            case 'median':
                image = np.median(image, axis=0)
            case 'first':
                image = image[0]
            case 'random':
                image = image[np.random.randint(image.shape[0])]

        mask = np.load(mask_path)['arr_0']
        mask[mask != 3] = 0
        mask[mask == 3] = 1

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()
        mask = mask.unsqueeze(dim=0)

        if self.mask_2d:
            mask_0 = 1.0 - mask
            mask = torch.concat([mask_0, mask], dim=0)
        mask = mask.squeeze()

        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.image_filenames)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            A matplotlib Figure containing the rendered sample.
        """
        ncols = 2
        shape_of_image = sample['image'].shape
        if len(shape_of_image) == 4:
            # Plot the first timepoint
            image = sample['image'][0][:3].permute(1, 2, 0).cpu().numpy()
        else:
            image = sample['image'][:3].permute(1, 2, 0).cpu().numpy()
        image = image / 255.0

        if self.mask_2d:
            mask = sample['mask'][0].squeeze(0).cpu().numpy()
        else:
            mask = sample['mask'].cpu().numpy()
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction'].cpu().numpy()
            if self.mask_2d:
                prediction = prediction[0]
            ncols = 3

        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, cmap='gray', interpolation='none')
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(prediction, cmap='gray', interpolation='none')
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle:
            fig.suptitle(suptitle)

        return fig

    def _extract(self) -> None:
        """Extract the dataset."""
        img_pathname = os.path.join(self.root, self.filename_images)
        extract_archive(img_pathname)

        mask_pathname = os.path.join(self.root, self.filename_masks)
        extract_archive(mask_pathname)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        image_path = os.path.join(self.image_dir, '*.npz')
        mask_path = os.path.join(self.mask_dir, '*.npz')
        if glob.glob(image_path) and glob.glob(mask_path):
            return

        # Check if the tar.gz files for images and masks have already been downloaded
        image_exists = os.path.exists(os.path.join(self.root, self.filename_images))
        mask_exists = os.path.exists(os.path.join(self.root, self.filename_masks))
        if image_exists and mask_exists:
            self._extract()
            return

        # If dataset files are missing and download is not allowed, raise an error
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        # Download and verify images
        download_url(
            self.url_for_images,
            self.root,
            filename=self.filename_images,
            md5=self.md5_images if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename_images), self.root)

        # Download and verify masks
        download_url(
            self.url_for_masks,
            self.root,
            filename=self.filename_masks,
            md5=self.md5_masks if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename_masks), self.root)
