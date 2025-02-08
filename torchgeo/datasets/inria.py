# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Inria Aerial Image Labeling Dataset."""

import glob
import os
import re
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    check_integrity,
    extract_archive,
    percentile_normalization,
)


class InriaAerialImageLabeling(NonGeoDataset):
    r"""Inria Aerial Image Labeling Dataset.

    The `Inria Aerial Image Labeling <https://project.inria.fr/aerialimagelabeling/>`__
    dataset is a building detection dataset over dissimilar settlements ranging from
    densely populated areas to alpine towns. Refer to the dataset homepage to download
    the dataset.

    Dataset features:

    * Coverage of 810 km\ :sup:`2`\  (405 km\ :sup:`2`\  for training and 405
      km\ :sup:`2`\  for testing)
    * Aerial orthorectified color imagery with a spatial resolution of 0.3 m
    * Number of images: 360 (train: 180, test: 180)
    * Train cities: Austin, Chicago, Kitsap, West Tyrol, Vienna
    * Test cities: Bellingham, Bloomington, Innsbruck, San Francisco, East Tyrol

    Dataset format:

    * Imagery - RGB aerial GeoTIFFs of shape 5000 x 5000
    * Labels - RGB aerial GeoTIFFs of shape 5000 x 5000

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2017.8127684

    .. versionadded:: 0.3

    .. versionchanged:: 0.5
       Added support for a *val* split.
    """

    directory = 'AerialImageDataset'
    filename = 'NEW2-AerialImageDataset.zip'
    md5 = '4b1acfe84ae9961edc1a6049f940380f'

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new InriaAerialImageLabeling Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` is invalid
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()
        self.files = self._load_files(root)

    def _load_files(self, root: Path) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        split = 'train' if self.split in ['train', 'val'] else 'test'
        root_dir = os.path.join(root, self.directory, split)
        pattern = re.compile(r'([A-Za-z]+)(\d+)')

        images = glob.glob(os.path.join(root_dir, 'images', '*.tif'))
        images = sorted(images)

        if split == 'train':
            labels = glob.glob(os.path.join(root_dir, 'gt', '*.tif'))
            labels = sorted(labels)

            for img, lbl in zip(images, labels):
                if match := pattern.search(img):
                    idx = int(match.group(2))
                    # For validation, use the first 5 images of every location
                    if self.split == 'train' and idx > 5:
                        files.append({'image': img, 'label': lbl})
                    elif self.split == 'val' and idx < 6:
                        files.append({'image': img, 'label': lbl})
        else:
            for img in images:
                files.append({'image': img})

        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rio.open(path) as img:
            array = img.read().astype(np.int32)
            tensor = torch.from_numpy(array).float()
            return tensor

    def _load_target(self, path: Path) -> Tensor:
        """Loads the target mask.

        Args:
            path: path to the mask

        Returns:
            the target mask
        """
        with rio.open(path) as img:
            array = img.read().astype(np.int32)
            array = np.clip(array, 0, 1)
            mask = torch.from_numpy(array[0]).long()
            return mask

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        img = self._load_image(files['image'])
        sample = {'image': img}
        if files.get('label'):
            mask = self._load_target(files['label'])
            sample['mask'] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Checks the integrity of the dataset structure."""
        if os.path.isdir(os.path.join(self.root, self.directory)):
            return

        archive_path = os.path.join(self.root, self.filename)
        md5_hash = self.md5 if self.checksum else None
        if not os.path.isfile(archive_path):
            raise DatasetNotFoundError(self)
        if not check_integrity(archive_path, md5_hash):
            raise RuntimeError('Dataset corrupted')
        print('Extracting...')
        extract_archive(archive_path)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample['image'][:3].numpy(), 0, 3)
        image = percentile_normalization(image, axis=(0, 1))

        ncols = 1
        show_mask = 'mask' in sample
        show_predictions = 'prediction' in sample

        if show_mask:
            mask = sample['mask'].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample['prediction'].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('Image')

        if show_mask:
            axs[1].imshow(mask, interpolation='none')
            axs[1].axis('off')
            if show_titles:
                axs[1].set_title('Label')

        if show_predictions:
            axs[2].imshow(prediction, interpolation='none')
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
