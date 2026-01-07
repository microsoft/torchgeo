# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import (
    Path,
    download_and_extract_archive,
    download_url,
    draw_semantic_segmentation_masks,
    extract_archive,
    percentile_normalization,
    sort_sentinel2_bands,
)


class OSCD(NonGeoDataset):
    """OSCD dataset.

    The `Onera Satellite Change Detection <https://rcdaudt.github.io/oscd/>`_
    dataset addresses the issue of detecting changes between
    satellite images from different dates. Imagery comes from
    Sentinel-2 which contains varying resolutions per band.

    Dataset format:

    * images are 13-channel tifs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    0. no change
    1. change

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2018.8518015

    .. versionadded:: 0.2
    """

    urls: ClassVar[dict[str, str]] = {
        'Onera Satellite Change Detection dataset - Images.zip': (
            'https://huggingface.co/datasets/hkristen/oscd/resolve/main/'
            'Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip'
        ),
        'Onera Satellite Change Detection dataset - Train Labels.zip': (
            'https://huggingface.co/datasets/hkristen/oscd/resolve/main/'
            'Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip'
        ),
        'Onera Satellite Change Detection dataset - Test Labels.zip': (
            'https://huggingface.co/datasets/hkristen/oscd/resolve/main/'
            'Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip'
        ),
    }
    md5s: ClassVar[dict[str, str]] = {
        'Onera Satellite Change Detection dataset - Images.zip': (
            'c50d4a2941da64e03a47ac4dec63d915'
        ),
        'Onera Satellite Change Detection dataset - Train Labels.zip': (
            '4d2965af8170c705ebad3d6ee71b6990'
        ),
        'Onera Satellite Change Detection dataset - Test Labels.zip': (
            '8177d437793c522653c442aa4e66c617'
        ),
    }

    zipfile_glob = '*Onera*.zip'
    filename_glob = '*Onera*'
    splits = ('train', 'test')

    colormap = ('blue',)

    all_bands = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B10',
        'B11',
        'B12',
    )

    rgb_bands = ('B04', 'B03', 'B02')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new OSCD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        assert set(bands) <= set(self.all_bands)
        self.bands = bands
        self.all_band_indices = [self.all_bands.index(b) for b in self.bands]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        .. versionchanged:: 0.8
           Now returns a single T x C x H x W image.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files['images1'])
        image2 = self._load_image(files['images2'])
        mask = self._load_target(str(files['mask']))
        image = torch.stack(tensors=[image1, image2], dim=0)
        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            # FIXME: VideoSequential only works with a batch dimension
            sample['image'] = sample['image'].unsqueeze(0)
            sample['mask'] = sample['mask'].unsqueeze(0)
            sample = self.transforms(sample)
            sample['image'] = sample['image'].squeeze(0)
            sample['mask'] = sample['mask'].squeeze(0)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str | Sequence[str]]]:
        regions = []
        labels_root = os.path.join(
            self.root,
            f'Onera Satellite Change Detection dataset - {self.split.capitalize()} '
            + 'Labels',
        )
        images_root = os.path.join(
            self.root, 'Onera Satellite Change Detection dataset - Images'
        )
        folders = glob.glob(os.path.join(labels_root, '*/'))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, 'cm', 'cm.png')

            def get_image_paths(ind: int) -> list[str]:
                return sorted(
                    glob.glob(
                        os.path.join(images_root, region, f'imgs_{ind}_rect', '*.tif')
                    ),
                    key=sort_sentinel2_bands,
                )

            images1, images2 = get_image_paths(1), get_image_paths(2)
            images1 = [images1[i] for i in self.all_band_indices]
            images2 = [images2[i] for i in self.all_band_indices]

            with open(os.path.join(images_root, region, 'dates.txt')) as f:
                dates = tuple(
                    line.split()[-1] for line in f.read().strip().splitlines()
                )

            regions.append(
                dict(
                    region=region,
                    images1=images1,
                    images2=images2,
                    mask=mask,
                    dates=dates,
                )
            )

        return regions

    def _load_image(self, paths: Sequence[Path]) -> Tensor:
        """Load a single image.

        Args:
            paths: paths to each image band

        Returns:
            the image
        """
        images: list[np.typing.NDArray[np.int_]] = []
        for path in paths:
            with Image.open(path) as img:
                images.append(np.array(img))
        array: np.typing.NDArray[np.int_] = np.stack(images, axis=0).astype(np.int_)
        tensor = torch.from_numpy(array).float()
        return tensor

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

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, '**', self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            if not fname.endswith('.zip'):
                return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
        if glob.glob(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for filename in self.urls:
            download_url(
                self.urls[filename],
                self.root,
                filename=filename,
                md5=self.md5s[filename] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        ncols = 2

        try:
            rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        except ValueError as e:
            raise RGBBandsMissingError() from e

        def get_masked(img: Tensor) -> 'np.typing.NDArray[np.uint8]':
            rgb_img = img[rgb_indices].float().numpy()
            per02 = np.percentile(rgb_img, 2)
            per98 = np.percentile(rgb_img, 98)
            rgb_img = (np.clip((rgb_img - per02) / (per98 - per02), 0, 1) * 255).astype(
                np.uint8
            )
            array: np.typing.NDArray[np.uint8] = draw_semantic_segmentation_masks(
                torch.from_numpy(rgb_img),
                sample['mask'][0],
                alpha=alpha,
                colors=list(self.colormap),
            )
            return array

        image1 = get_masked(sample['image'][0])
        image2 = get_masked(sample['image'][1])
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis('off')
        axs[1].imshow(image2)
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title('Pre change')
            axs[1].set_title('Post change')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class OSCD100(NonGeoDataset):
    """Subset of OSCD with 100 RGB image pairs at 256x256 resolution.

    Intended for tutorials and demonstrations, not benchmarking.

    Dataset features:

    * 100 image pairs cropped from OSCD Sentinel-2 scenes
    * RGB bands only (B04, B03, B02) at 10m resolution
    * 256x256 pixel crops
    * Binary change masks

    Dataset classes:

    0. no change
    1. change

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2018.8518015

    .. versionadded:: 0.9
    """

    url = 'https://huggingface.co/datasets/hkristen/oscd100/resolve/main'
    splits: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'filename': 'oscd100_train.zip',
            'md5': '08bdfb95625bb8b3b9f3496dffbcda80',
        },
        'val': {
            'filename': 'oscd100_val.zip',
            'md5': 'dc9f97c5d98660ef29d1c285b75dfc92',
        },
        'test': {
            'filename': 'oscd100_test.zip',
            'md5': '430e03d6153774265cc61402ae42e153',
        },
    }
    directories = ('A', 'B', 'label')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new OSCD100 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
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

        self.files = self._load_files()

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

    def _load_files(self) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image1, image2, mask
        """
        images1 = sorted(glob.glob(os.path.join(self.root, 'A', f'{self.split}*.png')))
        images2 = sorted(glob.glob(os.path.join(self.root, 'B', f'{self.split}*.png')))
        masks = sorted(
            glob.glob(os.path.join(self.root, 'label', f'{self.split}*.png'))
        )

        files = []
        for image1, image2, mask in zip(images1, images2, masks):
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
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
        with Image.open(path) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('L'))
            tensor = torch.from_numpy(array)
            tensor = torch.clamp(tensor, min=0, max=1)
            tensor = tensor.to(torch.long)
            return einops.rearrange(tensor, 'h w -> () h w')

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        return all(
            os.path.exists(os.path.join(self.root, directory))
            for directory in self.directories
        )

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for split in self.splits:
            filename = self.splits[split]['filename']
            download_and_extract_archive(
                f'{self.url}/{filename}',
                self.root,
                filename=filename,
                md5=self.splits[split]['md5'] if self.checksum else None,
            )

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
        """
        ncols = 3

        image1 = sample['image'][0].permute(1, 2, 0).numpy()
        image1 = percentile_normalization(image1, axis=(0, 1))

        image2 = sample['image'][1].permute(1, 2, 0).numpy()
        image2 = percentile_normalization(image2, axis=(0, 1))

        if 'prediction' in sample:
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10))

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
