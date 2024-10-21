# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Substation Segmentation Dataset."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class SubstationDataset(NonGeoDataset):
    """SubstationDataset is responsible for handling the loading and transformation of substation segmentation datasets.
    
    It extends NonGeoDataset, providing methods for dataset verification, downloading, and transformation.
    """
    directory = 'Substation'
    filename_images = 'image_stack.tar.gz'
    filename_masks = 'mask.tar.gz'
    url_for_images = ''
    url_for_masks = ''

    def __init__(self, args: object, image_files: list, geo_transforms: object = None, color_transforms: object = None,
                 image_resize: object = None, mask_resize: object = None) -> None:
        """Initialize the dataset with the provided parameters.

        Args:
            args: Arguments that contain configuration information such as data_dir, in_channels, and others.
            image_files: List of image filenames for the dataset.
            geo_transforms: Geometric transformations to apply on the images and masks.
            color_transforms: Color transformations to apply on the images.
            image_resize: Resize transformation for images.
            mask_resize: Resize transformation for masks.
        """
        self.data_dir = args.data_dir
        self.geo_transforms = geo_transforms
        self.color_transforms = color_transforms
        self.image_resize = image_resize
        self.mask_resize = mask_resize
        self.in_channels = args.in_channels
        self.use_timepoints = args.use_timepoints
        self.normalizing_type = args.normalizing_type
        self.normalizing_factor = args.normalizing_factor
        self.mask_2d = args.mask_2d
        self.model_type = args.model_type

        self.image_dir = os.path.join(self.data_dir, 'image_stack')
        self.mask_dir = os.path.join(self.data_dir, 'mask')
        self.image_filenames = image_files
        self.args = args

        # Check if the dataset is available or needs to be downloaded
        self._verify()

    def __getitem__(self, index: int) -> tuple:
        """Get an item from the dataset by index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A tuple containing the image and corresponding mask.
        """
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, image_filename)

        image = np.load(image_path)['arr_0']
        mask = np.load(mask_path)['arr_0']

        # Standardize images
        image = self._normalize_image(image)

        # Handle multiple channels and timepoints
        image = self._handle_channels_and_timepoints(image)

        # Process mask
        mask[mask != 3] = 0
        mask[mask == 3] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(dim=0)

        # Apply geo and color transformations
        image, mask = self._apply_transforms(image, mask)

        return image, mask

    def __len__(self) -> int:
        """Return the length of the dataset.
        
        Returns:
            Number of items in the dataset.
        """
        return len(self.image_filenames)

    def plot(self) -> None:
        """Plot a random image and mask from the dataset."""
        index = np.random.randint(0, self.__len__())
        image, mask = self.__getitem__(index)
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[1].imshow(image.permute(1, 2, 0))
        axs[1].imshow(mask.permute(1, 2, 0), alpha=0.5, cmap='gray')

    def _normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image based on the selected normalizing type.

        Args:
            image: The image to normalize.

        Returns:
            Normalized image as a torch tensor.
        """
        if self.normalizing_type == 'percentile':
            image = (image - self.normalizing_factor[:, 0].reshape((-1, 1, 1))) / self.normalizing_factor[:, 2].reshape((-1, 1, 1))
        elif self.normalizing_type == 'zscore':
            image = (image - self.args.means) / self.args.stds
        else:
            image = image / self.normalizing_factor
            image = np.clip(image, 0, 1)
        return torch.from_numpy(image)

    def _handle_channels_and_timepoints(self, image: np.ndarray) -> torch.Tensor:
        """Handle channels and timepoints in the image.

        Args:
            image: The image to process.

        Returns:
            Processed image as a torch tensor.
        """
        if self.in_channels == 3:
            image = image[:, [3, 2, 1], :, :]
        else:
            image = image[:4, :, :, :] if self.use_timepoints else image[0]
        return torch.from_numpy(image)

    def _apply_transforms(self, image: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Apply transformations to the image and mask.

        Args:
            image: The image tensor.
            mask: The mask tensor.

        Returns:
            A tuple containing the transformed image and mask.
        """
        if self.geo_transforms:
            combined = torch.cat((image, mask), 0)
            combined = self.geo_transforms(combined)
            image, mask = torch.split(combined, [image.shape[0], mask.shape[0]], 0)

        if self.color_transforms and self.in_channels >= 3:
            num_timepoints = image.shape[0] // self.in_channels
            for i in range(num_timepoints):
                image[i * self.in_channels:i * self.in_channels + 3, :, :] = self.color_transforms(image[i * self.in_channels:i * self.in_channels + 3, :, :])

        if self.image_resize:
            image = self.image_resize(image)
        if self.mask_resize:
            mask = self.mask_resize(mask)

        return image, mask

    def _verify(self) -> None:
        """Check if dataset exists, otherwise download it."""
        image_dir_exists = os.path.exists(self.image_dir)
        mask_dir_exists = os.path.exists(self.mask_dir)
        if not (image_dir_exists and mask_dir_exists):
            self._download()

    def _download(self) -> None:
        """Download images and masks if not already present."""
        print("Downloading images and masks...")
        download_url(self.url_for_images, self.data_dir, filename=self.filename_images)
        extract_archive(os.path.join(self.data_dir, self.filename_images), self.data_dir)

        download_url(self.url_for_masks, self.data_dir, filename=self.filename_masks)
        extract_archive(os.path.join(self.data_dir, self.filename_masks), self.data_dir)
