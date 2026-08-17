# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Forest Change dataset."""

import json
import os
import textwrap
from collections.abc import Callable, Iterator
from random import randint
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_and_extract_archive, lazy_import

if TYPE_CHECKING:
    import tokenizers


class ForestChange(NonGeoDataset):
    """Forest change detection and captioning dataset.

    The `Forest-Change
    <https://huggingface.co/datasets/JimmyBrocko/Forest-Change>`__
    dataset is the first benchmark designed for joint forest change detection
    and captioning in remote sensing imagery.  It provides bi-temporal
    satellite image pairs from Google Earth (Landsat), pixel-level deforestation masks, and
    multi-granularity natural-language captions describing forest cover
    changes in tropical and subtropical regions.

    Dataset features:

    * 334 annotated bi-temporal RGB image pairs at ~30 m/pixel resolution
    * binary change masks (no change = 0, deforestation = 1)
    * five natural-language captions per image pair describing the change
    * geographic focus on tropical and subtropical deforestation fronts

    Dataset format:

    * images are three-channel PNGs under
      ``<root>/Forest-Change-dataset/images/<split>/A/`` and ``B/``
    * masks are single-channel PNGs under
      ``<root>/Forest-Change-dataset/images/<split>/label/``
    * raw captions are stored in
      ``<root>/Forest-Change-dataset/ForestChatcaptions.json``

    Dataset classes:

    0. no change
    1. deforestation

    If you use this dataset in your research, please cite:

    * https://www.sciencedirect.com/science/article/pii/S1574954126001470

    .. note::
       This dataset requires the following additional library to be installed:

       * `tokenizers <https://pypi.org/project/tokenizers/>`_ to tokenize the captions

    .. versionadded:: 0.11
    """

    splits = ('train', 'val', 'test')

    classes = ('no_change', 'deforestation')

    directories = ('A', 'B', 'label')

    directory = 'Forest-Change-dataset'

    captions_filename = 'ForestChatcaptions.json'

    url = 'https://hf.co/datasets/JimmyBrocko/Forest-Change/resolve/e8b25bf09c85ec85633d1b1b554f7bb23e47724d/Forest-Change-dataset.zip'
    sha256 = '424931a075f00f8cf21d4d2f622df688de559494844df4876b59bde13d3d855d'
    filename = 'Forest-Change-dataset.zip'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
        tokenizer: 'tokenizers.models.Model | None' = None,
    ) -> None:
        """Initialize a new ForestChange instance.

        Args:
            root: root directory where dataset can be found
            split: one of 'train', 'val', or 'test'
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the SHA256 of the downloaded files (may be slow)
            tokenizer: a pre-trained tokenizer
                (defaults to :class:`~tokenizers.models.BPE`).

        Raises:
            AssertionError: if *split* is invalid
            DatasetNotFoundError: if dataset is not found and *download* is False
            DependencyNotFoundError: if tokenizers is not installed
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

        captions_path = os.path.join(
            str(self.root), self.directory, self.captions_filename
        )
        with open(captions_path) as f:
            data: dict[str, Any] = json.load(f)

        self._captions_by_stem: dict[str, list[str]] = {}
        for img in data['images']:
            stem = os.path.splitext(img['filename'])[0]
            self._captions_by_stem[stem] = [
                sentence['raw'] for sentence in img['sentences'] if sentence['raw']
            ]

        self.files = self._load_files()

        if tokenizer is None:
            tokenizers = lazy_import('tokenizers')
            self.tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
            trainer = tokenizers.trainers.BpeTrainer()
            train_captions = self._caption_iterator('train')
            self.tokenizer.train_from_iterator(train_captions, trainer)
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

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
        f = self.files[index]
        image1 = self._load_image(f['image1'])
        image2 = self._load_image(f['image2'])
        mask = self._load_target(f['mask'])

        captions = self._captions_by_stem.get(f['name'], [])

        sample: Sample = {
            'image': torch.stack([image1, image2]),
            'mask': mask,
            'caption': self._load_tokens(captions, f['token_id']),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each
                panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 3
        if 'prediction' in sample:
            ncols += 1

        image1 = sample['image'][0].permute(1, 2, 0).numpy().astype(np.uint8)
        image2 = sample['image'][1].permute(1, 2, 0).numpy().astype(np.uint8)

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10))

        axs[0].imshow(image1)
        axs[0].axis('off')

        axs[1].imshow(image2)
        axs[1].axis('off')

        axs[2].imshow(sample['mask'][0], cmap='gray', interpolation='none')
        axs[2].axis('off')

        caption = sample['caption'].numpy()
        caption_text = textwrap.wrap(self.tokenizer.decode(caption))

        if 'caption_prediction' in sample:
            pred_caption = sample['caption_prediction'].numpy()
            caption_text += ['Predicted:'] + textwrap.wrap(
                self.tokenizer.decode(pred_caption)
            )

        if 'prediction' in sample:
            axs[3].imshow(sample['prediction'][0], cmap='gray', interpolation='none')
            axs[3].axis('off')

            if show_titles:
                axs[3].set_title('Prediction')

        if show_titles:
            axs[0].set_title('Image 1')
            axs[1].set_title('Image 2')
            axs[2].set_title('Mask')

        fig.text(
            0.5,
            0.01,
            'Captions:\n' + '\n'.join(caption_text),
            ha='center',
            va='bottom',
            wrap=True,
            fontsize=10,
        )

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout(rect=(0, 0.02, 1, 1))

        return fig

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the image directories and captions JSON are found, else False
        """
        captions_path = os.path.join(
            str(self.root), self.directory, self.captions_filename
        )
        if not os.path.exists(captions_path):
            return False
        for split in self.splits:
            for directory in self.directories:
                if not os.path.exists(
                    os.path.join(
                        str(self.root), self.directory, 'images', split, directory
                    )
                ):
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
            sha256=self.sha256 if self.checksum else None,
        )

    def _caption_iterator(self, split: str) -> Iterator[str]:
        """Yield every raw caption belonging to *split*.

        Args:
            split: split whose captions should be yielded

        Yields:
            individual raw caption strings
        """
        img_dir = os.path.join(str(self.root), self.directory, 'images', split, 'A')
        for name in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(name)[0]
            yield from self._captions_by_stem.get(stem, [])

    def _load_files(self) -> list[dict[str, Any]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image1, image2, mask,
            plus the caption index and stem name
        """
        img_dir = os.path.join(str(self.root), self.directory, 'images', self.split)
        names = sorted(os.listdir(os.path.join(img_dir, 'A')))

        files: list[dict[str, Any]] = []
        for name in names:
            stem = os.path.splitext(name)[0]
            # train uses a random caption each call, val/test always use the
            # first caption so evaluation is reproducible
            token_id = None if self.split == 'train' else 0
            files.append(
                {
                    'image1': os.path.join(img_dir, 'A', name),
                    'image2': os.path.join(img_dir, 'B', name),
                    'mask': os.path.join(img_dir, 'label', name),
                    'token_id': token_id,
                    'name': stem,
                }
            )
        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(str(path)) as img:
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
        with Image.open(str(path)) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('L'))
            tensor = torch.from_numpy(array)
            tensor = torch.clamp(tensor, min=0, max=1).to(torch.long)
            return tensor.unsqueeze(0)

    def _load_tokens(self, captions: list[str], token_id: int | None) -> Tensor:
        """Select and tokenize a single caption for a sample.

        Args:
            captions: raw caption strings for this sample
            token_id: index of the caption to encode. If None, a caption
                is chosen at random.

        Returns:
            encoded caption

        Raises:
            ValueError: if the sample contains no captions or if
                ``token_id`` is outside the valid caption range
        """
        n = len(captions)
        if n == 0:
            raise ValueError('No captions available for sample')

        if token_id is not None:
            if not 0 <= token_id < n:
                raise ValueError(
                    f'Caption index {token_id} out of range for sample '
                    f'with {n} captions'
                )
            j = token_id
        else:
            j = randint(0, n - 1)

        output = self.tokenizer.encode(captions[j])
        return torch.tensor(output.ids)
