# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD and LEVIR-CD+ datasets."""

import abc
import glob
import os
from collections.abc import Callable
from typing import ClassVar

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_and_extract_archive, percentile_normalization


class LEVIRCDBase(NonGeoDataset, abc.ABC):
    """Abstract base class for the LEVIRCD datasets.

    .. versionadded:: 0.6
    """

    splits: ClassVar[tuple[str, ...] | dict[str, dict[str, str]]]
    directories = ('A', 'B', 'label')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LEVIR-CD base dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self.files = self._load_files(self.root, self.split)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files['image1'])
        image2 = self._load_image(files['image2'])
        mask = self._load_target(files['mask'])
        sample = {'image': torch.stack([image1, image2]), 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('RGB'))
            tensor = torch.from_numpy(array).float()
            return einops.rearrange(tensor, 'h w c -> c h w')

    def _load_target(self, path: Path) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('L'))
            tensor = torch.from_numpy(array)
            tensor = torch.clamp(tensor, min=0, max=1)
            tensor = tensor.to(torch.long)
            # VideoSequential requires time dimension
            return einops.rearrange(tensor, 'h w -> () h w')

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
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        ncols = 3

        image1 = sample['image'][0].permute(1, 2, 0).numpy()
        image1 = percentile_normalization(image1, axis=(0, 1))

        image2 = sample['image'][1].permute(1, 2, 0).numpy()
        image2 = percentile_normalization(image2, axis=(0, 1))

        if 'prediction' in sample:
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 5))

        axs[0].imshow(image1)
        axs[0].axis('off')
        axs[1].imshow(image2)
        axs[1].axis('off')
        axs[2].imshow(sample['mask'][0], cmap='gray', interpolation='none')
        axs[2].axis('off')

        if 'prediction' in sample:
            axs[3].imshow(sample['prediction'][0], cmap='gray', interpolation='none')
            axs[3].axis('off')
            if show_titles:
                axs[3].set_title('Prediction')

        if show_titles:
            axs[0].set_title('Image 1')
            axs[1].set_title('Image 2')
            axs[2].set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    @abc.abstractmethod
    def _load_files(self, root: Path, split: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of image1, image2, mask
        """

    @abc.abstractmethod
    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """

    @abc.abstractmethod
    def _download(self) -> None:
        """Download the dataset and extract it."""


class LEVIRCD(LEVIRCDBase):
    """LEVIR-CD dataset.

    The `LEVIR-CD <https://github.com/justchenhao/STANet>`__
    dataset is a dataset for building change detection.

    Dataset features:

    * image pairs of 20 different urban regions across Texas between 2002-2018
    * binary change masks representing building change
    * three spectral bands - RGB
    * 637 image pairs with 50 cm per pixel resolution (~1024x1024 px)

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    1. no change
    2. change

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.3390/rs12101662

    .. versionadded:: 0.6
    """

    splits: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'url': 'https://drive.google.com/file/d/18GuoCuBn48oZKAlEo-LrNwABrFhVALU-',
            'filename': 'train.zip',
            'md5': 'a638e71f480628652dea78d8544307e4',
        },
        'val': {
            'url': 'https://drive.google.com/file/d/1BqSt4ueO7XAyQ_84mUjswUSJt13ZBuzG',
            'filename': 'val.zip',
            'md5': 'f7b857978524f9aa8c3bf7f94e3047a4',
        },
        'test': {
            'url': 'https://drive.google.com/file/d/1jj3qJD_grJlgIhUWO09zibRGJe0R4Tn0',
            'filename': 'test.zip',
            'md5': '07d5dd89e46f5c1359e2eca746989ed9',
        },
    }

    def _load_files(self, root: Path, split: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of image1, image2, mask
        """
        images1 = sorted(glob.glob(os.path.join(root, 'A', f'{split}*.png')))
        images2 = sorted(glob.glob(os.path.join(root, 'B', f'{split}*.png')))
        masks = sorted(glob.glob(os.path.join(root, 'label', f'{split}*.png')))

        files = []
        for image1, image2, mask in zip(images1, images2, masks):
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        return all(
            [
                os.path.exists(os.path.join(self.root, directory))
                for directory in self.directories
            ]
        )

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for split in self.splits:
            download_and_extract_archive(
                self.splits[split]['url'],
                self.root,
                filename=self.splits[split]['filename'],
                md5=self.splits[split]['md5'] if self.checksum else None,
            )


class LEVIRCDPlus(LEVIRCDBase):
    """LEVIR-CD+ dataset.

    The `LEVIR-CD+ <https://github.com/S2Looking/Dataset>`__
    dataset is a dataset for building change detection.

    Dataset features:

    * image pairs of 20 different urban regions across Texas between 2002-2020
    * binary change masks representing building change
    * three spectral bands - RGB
    * 985 image pairs with 50 cm per pixel resolution (~1024x1024 px)

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    1. no change
    2. change

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2107.09244
    """

    url = 'https://drive.google.com/file/d/1JamSsxiytXdzAIk6VDVWfc-OsX-81U81'
    md5 = '1adf156f628aa32fb2e8fe6cada16c04'
    filename = 'LEVIR-CD+.zip'
    directory = 'LEVIR-CD+'
    splits = ('train', 'test')

    def _load_files(self, root: Path, split: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of image1, image2, mask
        """
        files = []
        images = glob.glob(os.path.join(root, self.directory, split, 'A', '*.png'))
        images = sorted(os.path.basename(image) for image in images)
        for image in images:
            image1 = os.path.join(root, self.directory, split, 'A', image)
            image2 = os.path.join(root, self.directory, split, 'B', image)
            mask = os.path.join(root, self.directory, split, 'label', image)
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for filename in self.splits:
            filepath = os.path.join(self.root, self.directory, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
