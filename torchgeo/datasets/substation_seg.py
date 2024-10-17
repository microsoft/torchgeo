import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import fiona
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor
from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_url, extract_archive


class SubstationDataset(NonGeoDataset):
    directory = 'Substation'
    filename_images = 'image_stack.tar.gz'
    filename_masks = 'mask.tar.gz'
    url_for_images = ''
    url_for_masks = ''

    def __init__(self, args, image_files, geo_transforms=None, color_transforms=None, image_resize=None, mask_resize=None):
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

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.image_filenames)

    def plot(self):
        index = np.random.randint(0, self.__len__())
        image, mask = self.__getitem__(index)
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[1].imshow(image.permute(1, 2, 0))
        axs[1].imshow(mask.permute(1, 2, 0), alpha=0.5, cmap='gray')

    def _normalize_image(self, image):
        if self.normalizing_type == 'percentile':
            image = (image - self.normalizing_factor[:, 0].reshape((-1, 1, 1))) / self.normalizing_factor[:, 2].reshape((-1, 1, 1))
        elif self.normalizing_type == 'zscore':
            image = (image - self.args.means) / self.args.stds
        else:
            image = image / self.normalizing_factor
            image = np.clip(image, 0, 1)
        return image

    def _handle_channels_and_timepoints(self, image):
        if self.in_channels == 3:
            image = image[:, [3, 2, 1], :, :]
        else:
            image = image[:4, :, :, :] if self.use_timepoints else image[0]
        return torch.from_numpy(image)

    def _apply_transforms(self, image, mask):
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

    def _verify(self):
        """Check if dataset exists, otherwise download it."""
        image_dir_exists = os.path.exists(self.image_dir)
        mask_dir_exists = os.path.exists(self.mask_dir)
        if not (image_dir_exists and mask_dir_exists):
            self._download()

    def _download(self):
        """Download images and masks if not already present."""
        print("Downloading images and masks...")
        download_url(self.url_for_images, self.data_dir, filename=self.filename_images)
        extract_archive(os.path.join(self.data_dir, self.filename_images), self.data_dir)

        download_url(self.url_for_masks, self.data_dir, filename=self.filename_masks)
        extract_archive(os.path.join(self.data_dir, self.filename_masks), self.data_dir)
