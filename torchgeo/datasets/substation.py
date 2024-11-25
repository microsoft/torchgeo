# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module handles the Substation segmentation dataset."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class SubstationDataset(NonGeoDataset):
    """Base class for Substation Dataset.
    
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

    directory: str = 'Substation'
    filename_images: str = 'image_stack.tar.gz'
    filename_masks: str = 'mask.tar.gz'
    url_for_images: str = 'https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/image_stack.tar.gz'
    url_for_masks: str = 'https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/mask.tar.gz'

    def __init__(
        self,
        args: Any,
        image_files: list[str],
        geo_transforms: Any | None = None,
        color_transforms: Any | None = None,
        image_resize: Any | None = None,
        mask_resize: Any | None = None,
    ) -> None:
        """Initialize the SubstationDataset.

        Args:
            args (Any): Arguments containing various dataset parameters such as `data_dir`, `in_channels`, etc.
            image_files (list[str]): A list of image file names.
            geo_transforms (Any | None): Geometric transformations to be applied to the images and masks. Defaults to None.
            color_transforms (Any | None): Color transformations to be applied to the images. Defaults to None.
            image_resize (Any | None): Transformation for resizing the images. Defaults to None.
            mask_resize (Any | None): Transformation for resizing the masks. Defaults to None.
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
        self.image_dir = os.path.join(args.data_dir, 'substation', 'image_stack')
        self.mask_dir = os.path.join(args.data_dir, 'substation', 'mask')
        self.image_filenames = image_files
        self.args = args

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
        # standardizing image
        if self.normalizing_type == 'percentile':
            image = (
                image - self.normalizing_factor[:, 0].reshape((-1, 1, 1))
            ) / self.normalizing_factor[:, 2].reshape((-1, 1, 1))
        elif self.normalizing_type == 'zscore':
            # means = np.array([1431, 1233, 1209, 1192, 1448, 2238, 2609, 2537, 2828, 884, 20, 2226, 1537]).reshape(-1, 1, 1)
            # stds = np.array([157, 254, 290, 420, 363, 457, 575, 606, 630, 156, 3, 554, 523]).reshape(-1, 1, 1)
            image = (image - self.args.means) / self.args.stds
        else:
            image = image / self.normalizing_factor
            # clipping image to 0,1 range
            image = np.clip(image, 0, 1)

        # selecting channels
        if self.in_channels == 3:
            image = image[:, [3, 2, 1], :, :]
        else:
            if self.model_type == 'swin':
                image = image[
                    :, [3, 2, 1, 4, 5, 6, 7, 10, 11], :, :
                ]  # swin only takes 9 channels
            else:
                image = image[:, : self.in_channels, :, :]

        # handling multiple images across timepoints
        if self.use_timepoints:
            image = image[:4, :, :, :]
            if self.args.timepoint_aggregation == 'concat':
                image = np.reshape(
                    image, (-1, image.shape[2], image.shape[3])
                )  # (4*channels,h,w)
            elif self.args.timepoint_aggregation == 'median':
                image = np.median(image, axis=0)
        else:
            # image = np.median(image, axis=0)
            # image = image[0]
            if self.args.timepoint_aggregation == 'first':
                image = image[0]
            elif self.args.timepoint_aggregation == 'random':
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

        if self.image_resize:
            image = self.image_resize(image)

        if self.mask_resize:
            mask = self.mask_resize(mask)

        return {'image': image, 'mask': mask}

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.image_filenames)

    def plot(self) -> None:
        """Plots a random image and mask from the dataset."""
        index = np.random.randint(0, self.__len__())
        data = self.__getitem__(index)
        image = data['image']
        mask = data['mask']

        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axs[1].imshow(image.permute(1, 2, 0).cpu().numpy())
        axs[1].imshow(mask.squeeze().cpu().numpy(), alpha=0.5, cmap='gray')

    def _verify(self) -> None:
        """Checks if dataset exists, otherwise download it."""
        image_dir_exists = os.path.exists(self.image_dir)
        mask_dir_exists = os.path.exists(self.mask_dir)
        if not (image_dir_exists and mask_dir_exists):
            self._download()

    def _download(self) -> None:
        """Download the dataset."""
        # Assuming self.url_for_images and self.url_for_masks are URLs for dataset components
        download_url(self.url_for_images, self.data_dir, filename=self.filename_images)
        extract_archive(
            os.path.join(self.data_dir, self.filename_images), self.data_dir
        )

        download_url(self.url_for_masks, self.data_dir, filename=self.filename_masks)
        extract_archive(os.path.join(self.data_dir, self.filename_masks), self.data_dir)
