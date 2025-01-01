# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module handles the Substation segmentation dataset."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class Substation(NonGeoDataset):
    """Class for Substation Dataset.

    This dataset is responsible for handling the loading and transformation of
    substation segmentation datasets. It extends NonGeoDataset, providing methods
    for dataset verification, downloading, and transformation.
    Dataset Format:
    * .npz file for each datapoint

    Dataset Features:

    * 26,522 image-mask pairs stored as numpy files.
    * Data from 5 revisits for most locations.
    * Multi-temporal, multi-spectral images (13 channels) paired with masks,
      with a spatial resolution of 228x228 pixels

    If you use this dataset in your research, please cite the following:
    * https://doi.org/10.48550/arXiv.2409.17363
    """

    directory = 'Substation'
    filename_images = 'image_stack.tar.gz'
    filename_masks = 'mask.tar.gz'
    url_for_images = 'https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/image_stack.tar.gz'
    url_for_masks = 'https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/mask.tar.gz'

    def __init__(
        self,
        root: str,
        bands: int,
        num_of_timepoints: int,
        use_timepoints: bool,
        mask_2d: bool,
        timepoint_aggregation: str,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize the Substation.

        Args:
            root: Path to the directory containing the dataset.
            bands: Number of channels to use from the image.
            use_timepoints: Whether to use multiple timepoints for each image.
            mask_2d: Whether to use a 2D mask.
            timepoint_aggregation: How to aggregate multiple timepoints.
            download: Whether to download the dataset if it is not found.
            checksum: Whether to verify the dataset after downloading.
        """
        self.root = root
        self.bands = bands
        self.use_timepoints = use_timepoints
        self.timepoint_aggregation = timepoint_aggregation
        self.mask_2d = mask_2d
        self.image_dir = os.path.join(root, 'substation', 'image_stack')
        self.mask_dir = os.path.join(root, 'substation', 'mask')
        self.num_of_timepoints = num_of_timepoints
        self.download = download
        self.checksum = checksum

        self._verify()
        self.load_image_filenames()

    def load_image_filenames(self) -> list[str]:
        """Load image filenames from the image directory."""
        self.image_filenames = os.listdir(self.image_dir)

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
        if self.use_timepoints:
            image = image[: self.num_of_timepoints, :, :, :]
            if self.timepoint_aggregation == 'concat':
                image = np.reshape(
                    image, (-1, image.shape[2], image.shape[3])
                )  # (num_of_timepoints*channels,h,w)
            elif self.timepoint_aggregation == 'median':
                image = np.median(image, axis=0)
        else:
            # image = np.median(image, axis=0)
            # image = image[0]
            if self.timepoint_aggregation == 'first':
                image = image[0]
            elif self.timepoint_aggregation == 'random':
                image = image[np.random.randint(image.shape[0])]

        mask = np.load(mask_path)['arr_0']
        mask[mask != 3] = 0
        mask[mask == 3] = 1

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(dim=0)

        if self.mask_2d:
            mask_0 = 1.0 - mask
            mask = torch.concat([mask_0, mask], dim=0)

        return {'image': image, 'mask': mask}

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

        image = sample['image'][:3].permute(1, 2, 0).cpu().numpy()
        image = image / 255.0  # Normalize image

        mask = sample['mask'][0].squeeze(0).cpu().numpy()

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction'][0].squeeze(0).cpu().numpy()
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
        if not getattr(self, 'download', True):
            raise FileNotFoundError(
                f'Dataset files not found in {self.root}. Enable downloading or provide the files.'
            )

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
            md5='INSERT_IMAGES_MD5_HASH' if self.checksum else None,
        )
        extract_archive(
            os.path.join(self.root, self.filename_images), self.root
        )

        # Download and verify masks
        download_url(
            self.url_for_masks,
            self.root,
            filename=self.filename_masks,
            md5='INSERT_MASKS_MD5_HASH' if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename_masks), self.root)
