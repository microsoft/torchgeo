# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Substation segmentation dataset."""

import glob
import os
import struct
import zlib
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url, extract_archive


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
      with a spatial resolution of 228x228 pixels. When ``timepoint_aggregation``
      is None, images are returned as T x C x H x W tensors.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2409.17363

    """

    # Sentinel-2 true color: B04 (Red), B03 (Green), B02 (Blue) = indices 3, 2, 1
    rgb_bands = (3, 2, 1)

    directory = 'Substation'
    filenames_images = ('images.z01', 'images.z02', 'images.zip')
    filename_masks = 'mask.tar.gz'
    url = 'https://hf.co/datasets/neurograce/SubstationDataset/resolve/465090a85529932dfdc9b20b85fc287313ac02fb/{}'
    checksums: ClassVar[dict[str, str]] = {
        'images.z01': '25566e86a11483c144566d0999c915909c15133bdedbdf5c9699e51595f3b54f',
        'images.z02': 'c8b387fdc9e09c9384f156639b13a5ec4c3c5984ff4bb7233461d18d3b85043d',
        'images.zip': 'b4e0f644b0e1aedb1fc01756efc98c23d9593f06ecccd04a4791e13fa28fc5d6',
        'mask.tar.gz': 'e6d0c5b613373826f22b9c83af2c511a83d7a4a8adb90685c00ffeb78a79c66e',
    }

    def __init__(
        self,
        root: Path = 'data',
        bands: Sequence[int] = tuple(range(13)),
        mask_2d: bool = True,
        num_of_timepoints: int = 4,
        timepoint_aggregation: Literal['concat', 'median', 'first', 'random']
        | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
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
                If None, returns time-series as T x C x H x W.
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

    def __getitem__(self, index: int) -> Sample:
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
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        When the image is 4D (T x C x H x W), the first two timepoints are plotted.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            A matplotlib Figure containing the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        is_time_series = sample['image'].ndim == 4

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(list(self.bands).index(band))
            else:
                raise RGBBandsMissingError()

        if is_time_series:
            images = (
                torch.clamp(sample['image'][:, rgb_indices] / 4000, min=0, max=1)
                .cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
            )
            num_images = min(len(images), 2)
            ncols = num_images + 1
        else:
            image = (
                torch.clamp(sample['image'][rgb_indices] / 4000, min=0, max=1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            ncols = 2

        if self.mask_2d:
            mask = sample['mask'][1].squeeze(dim=0).cpu().numpy()
        else:
            mask = sample['mask'].cpu().numpy()
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction'].cpu().numpy()
            if self.mask_2d:
                prediction = prediction[0]
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))

        if is_time_series:
            for i in range(num_images):
                axs[i].imshow(images[i])
                axs[i].axis('off')
                if show_titles:
                    axs[i].set_title(f'Image {i}')
            axs[num_images].imshow(mask, cmap='gray', interpolation='none')
            axs[num_images].axis('off')
            if show_titles:
                axs[num_images].set_title('Mask')
            if showing_predictions:
                axs[num_images + 1].imshow(
                    prediction, cmap='gray', interpolation='none'
                )
                axs[num_images + 1].axis('off')
                if show_titles:
                    axs[num_images + 1].set_title('Prediction')
        else:
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

    def _extract_images(self) -> None:
        """Extract the images.

        The images are distributed as a multi-part (split) zip archive. Such archives
        record local file header offsets relative to the part they live in, so they can
        neither be opened by :class:`zipfile.ZipFile` nor repaired by concatenating the
        parts. The parts are instead read as a single stream and each entry is inflated
        directly into :attr:`image_dir`.
        """
        os.makedirs(self.image_dir, exist_ok=True)
        with ExitStack() as stack:
            files = [
                stack.enter_context(open(os.path.join(self.root, file), 'rb'))
                for file in self.filenames_images
            ]

            def read(size: int) -> bytes:
                # Exhausted parts return b'', so the next part is picked up transparently
                chunks = []
                for file in files:
                    while size > 0 and (chunk := file.read(size)):
                        chunks.append(chunk)
                        size -= len(chunk)
                return b''.join(chunks)

            read(4)  # multi-part archive spanning signature
            # Stop at the central directory, which has a different signature
            while (header := read(30))[:4] == b'PK\x03\x04':
                size, name_len, extra_len = struct.unpack('<18xL4x2H', header)
                name = read(name_len).decode()
                read(extra_len)
                if name.endswith('/'):
                    continue

                # Replace the directory stored in the archive with image_dir
                path = os.path.join(self.image_dir, os.path.basename(name))
                decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                with open(path, 'wb') as f:
                    f.write(decompressor.decompress(read(size)))
                    f.write(decompressor.flush())

    def _extract(self) -> None:
        """Extract the dataset."""
        self._extract_images()
        extract_archive(os.path.join(self.root, self.filename_masks), self.root)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        image_path = os.path.join(self.image_dir, '*.npz')
        mask_path = os.path.join(self.mask_dir, '*.npz')
        if glob.glob(image_path) and glob.glob(mask_path):
            return

        # Check if the archives have already been downloaded
        files = (*self.filenames_images, self.filename_masks)
        if all(os.path.exists(os.path.join(self.root, file)) for file in files):
            self._extract()
            return

        # If dataset files are missing and download is not allowed, raise an error
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for file, sha256 in self.checksums.items():
            download_url(
                self.url.format(file),
                self.root,
                sha256=sha256 if self.checksum else None,
            )
