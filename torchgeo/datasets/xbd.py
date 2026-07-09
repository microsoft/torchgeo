# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""xBD dataset."""

import glob
import os
from collections.abc import Callable
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from typing_extensions import deprecated

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
)


class xBD(NonGeoDataset):
    """xBD dataset.

    The `xBD <https://xview2.org/dataset>`__
    dataset is a dataset for building disaster change detection. This dataset object
    uses the "Challenge training set (~7.8 GB)" and "Challenge test set (~2.6 GB)" data
    from the xView2 website as the train and test splits. Note, the xView2 website
    contains other data under the xView2 umbrella that are _not_ included here. E.g.
    the "Tier3 training data", the "Challenge holdout set", and the "full data".

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where the pixel values represent the class

    Dataset classes:

    0. background
    1. no damage
    2. minor damage
    3. major damage
    4. destroyed

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1911.09296

    .. versionadded:: 0.2
    """

    metadata: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'filename': 'train_images_labels_targets.tar.gz',
            'sha256': 'a5941b7a3e523eafc4aeaa740a1c83f1af6a18c894e7e8c62dd830a76921ecd4',
            'directory': 'train',
        },
        'test': {
            'filename': 'test_images_labels_targets.tar.gz',
            'sha256': '0fcdbfe3ee7d0842729dd2230217e74b2f12be35546ff666df4dae5388e2541c',
            'directory': 'test',
        },
    }
    classes = ('background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed')
    colormap = ('green', 'blue', 'orange', 'red')

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        checksum: bool = True,
    ) -> None:
        """Initialize a new xBD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, verify the checksum of the downloaded files (may be slow)

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in self.metadata
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files(root, split)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        .. versionchanged:: 0.8
           Now returns a single T x C x H x W image, change detection mask.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files['image1'])
        image2 = self._load_image(files['image2'])
        mask1 = self._load_target(files['mask1'])
        mask2 = self._load_target(files['mask2'])

        image = torch.stack(tensors=[image1, image2], dim=0)

        # Dataset consists of semantic segmentation masks before and after event
        # Convert to change detection by subtracting damage before from damage after
        # Clamp to avoid potential negative numbers
        mask = torch.clamp(mask2 - mask1, 0, 4)

        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(
        self, root: Path, split: Literal['train', 'test']
    ) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of images and masks
        """
        files = []
        directory = self.metadata[split]['directory']
        image_root = os.path.join(root, directory, 'images')
        mask_root = os.path.join(root, directory, 'targets')
        images = glob.glob(os.path.join(image_root, '*.png'))
        basenames = [os.path.basename(f) for f in images]
        basenames = ['_'.join(f.split('_')[:-2]) for f in basenames]
        for name in sorted(set(basenames)):
            image1 = os.path.join(image_root, f'{name}_pre_disaster.png')
            image2 = os.path.join(image_root, f'{name}_post_disaster.png')
            mask1 = os.path.join(mask_root, f'{name}_pre_disaster_target.png')
            mask2 = os.path.join(mask_root, f'{name}_post_disaster_target.png')
            files.append(
                {'image1': image1, 'image2': image2, 'mask1': mask1, 'mask2': mask2}
            )
        return files

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
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.float)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
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
            tensor = tensor.to(torch.long)
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        exists = []
        for split_info in self.metadata.values():
            for directory in ['images', 'targets']:
                exists.append(
                    os.path.exists(
                        os.path.join(self.root, split_info['directory'], directory)
                    )
                )

        if all(exists):
            return

        # Check if .tar.gz files already exists (if so then extract)
        exists = []
        for split_info in self.metadata.values():
            filepath = os.path.join(self.root, split_info['filename'])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(
                    filepath, sha256=split_info['sha256']
                ):
                    raise RuntimeError('Dataset found, but corrupted.')
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        raise DatasetNotFoundError(self)

    def plot(
        self,
        sample: Sample,
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
        """
        ncols = 2
        image1 = draw_semantic_segmentation_masks(
            sample['image'][0], sample['mask'], alpha=alpha, colors=list(self.colormap)
        )
        image2 = draw_semantic_segmentation_masks(
            sample['image'][1], sample['mask'], alpha=alpha, colors=list(self.colormap)
        )
        if 'prediction' in sample:
            ncols += 1
            image3 = draw_semantic_segmentation_masks(
                sample['image'][1],
                sample['prediction'],
                alpha=alpha,
                colors=list(self.colormap),
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis('off')
        axs[1].imshow(image2)
        axs[1].axis('off')
        if ncols > 2:
            axs[2].imshow(image3)
            axs[2].axis('off')

        if show_titles:
            axs[0].set_title('Pre disaster')
            axs[1].set_title('Post disaster')
            if ncols > 2:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


@deprecated('Use torchgeo.datasets.xBD instead')
class XView2(xBD):
    """Deprecated alias for the xBD dataset."""


class xBDDistShift(xBD):
    """xBD dataset with a custom, disaster-based train/test split.

    This subclass of :class:`xBD` reformulates the original train/test splits to enable
    domain adaptation and out-of-distribution (OOD) detection experiments, similar to
    :class:`EuroSATSpatial`. One disaster is used as the in-distribution (ID) training
    set and a different disaster as the out-of-distribution (OOD) test set, so that a
    model can be trained on one disaster type and evaluated on another. This tests the
    ability of models trained on one disaster to generalize to a different one.

    The multiclass damage masks are reformulated as a binary building segmentation task,
    where minor-damage and no-damage buildings are mapped to the *building* class and
    major-damage and destroyed buildings are mapped to the *background* class.

    .. versionadded:: 0.10
    """

    binary_classes = ('background', 'building')

    valid_disasters = (
        'hurricane-harvey',
        'socal-fire',
        'hurricane-matthew',
        'mexico-earthquake',
        'guatemala-volcano',
        'santa-rosa-wildfire',
        'palu-tsunami',
        'hurricane-florence',
        'hurricane-michael',
        'midwest-flooding',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'test'] = 'train',
        id_disaster: str = 'hurricane-matthew',
        id_pre_post: Literal['pre', 'post', 'both'] = 'post',
        ood_disaster: str = 'mexico-earthquake',
        ood_pre_post: Literal['pre', 'post', 'both'] = 'post',
        transforms: Callable[[Sample], Sample] | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new xBDDistShift dataset instance.

        Args:
            root: Root directory where the dataset can be found.
            split: One of "train" (in-distribution) or "test" (out-of-distribution).
            id_disaster: Name of the disaster used as the in-distribution (training)
                set. Must be one of *valid_disasters*.
            id_pre_post: Which imagery of the in-distribution disaster to use: "pre"
                (before the event), "post" (after the event), or "both".
            ood_disaster: Name of the disaster used as the out-of-distribution (test)
                set. Must be one of *valid_disasters*.
            ood_pre_post: Which imagery of the out-of-distribution disaster to use:
                "pre" (before the event), "post" (after the event), or "both".
            transforms: A function/transform that takes an input sample and returns a
                transformed version.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If *split* is invalid.
            ValueError: If *id_disaster* or *ood_disaster* is not one of
                *valid_disasters*.
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in ('train', 'test')
        for disaster in (id_disaster, ood_disaster):
            if disaster not in self.valid_disasters:
                raise ValueError(
                    f'Invalid disaster name: {disaster}. '
                    f'Valid options are: {", ".join(self.valid_disasters)}.'
                )

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        files = self._gather_files(root)
        self.split_files = {
            'train': self._filter_files(files, id_disaster, id_pre_post),
            'test': self._filter_files(files, ood_disaster, ood_pre_post),
        }

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and label at that index, with the mask reformulated as a binary
            building segmentation target.
        """
        files = self.split_files[self.split][index]
        image = self._load_image(files['image'])
        mask = self._load_target(files['mask'])

        # Reformulate the multiclass damage mask as a binary building mask
        mask[mask == 2] = 1  # minor-damage -> building
        mask[(mask == 3) | (mask == 4)] = 0  # major-damage, destroyed -> background

        sample = {'image': image, 'mask': mask}
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the current split.

        Returns:
            Length of the current split.
        """
        return len(self.split_files[self.split])

    def _gather_files(self, root: Path) -> list[dict[str, str]]:
        """Return the image and mask paths for every disaster across both splits.

        Args:
            root: Root directory of the dataset.

        Returns:
            List of dicts, each with the image path, mask path, disaster name, and
            pre/post type of a single sample.
        """
        files = []
        for split_info in self.metadata.values():
            directory = split_info['directory']
            image_root = os.path.join(root, directory, 'images')
            mask_root = os.path.join(root, directory, 'targets')
            for image in sorted(glob.glob(os.path.join(image_root, '*.png'))):
                basename = os.path.basename(image)
                disaster = basename.split('_')[0]
                pre_post = 'pre' if 'pre_disaster' in basename else 'post'
                mask = os.path.join(mask_root, basename.replace('.png', '_target.png'))
                files.append(
                    {
                        'image': image,
                        'mask': mask,
                        'disaster': disaster,
                        'pre_post': pre_post,
                    }
                )
        return files

    def _filter_files(
        self,
        files: list[dict[str, str]],
        disaster: str,
        pre_post: Literal['pre', 'post', 'both'],
    ) -> list[dict[str, str]]:
        """Select the files matching a disaster and pre/post type.

        Args:
            files: List of all files as returned by :meth:`_gather_files`.
            disaster: Disaster name to select.
            pre_post: Which imagery to select: "pre", "post", or "both".

        Returns:
            List of dicts with the image and mask paths of the matching samples.
        """
        return [
            {'image': file['image'], 'mask': file['mask']}
            for file in files
            if file['disaster'] == disaster and pre_post in ('both', file['pre_post'])
        ]
