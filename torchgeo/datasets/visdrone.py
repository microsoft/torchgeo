# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""VisDrone dataset."""

import glob
import os
from collections.abc import Callable
from typing import ClassVar, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class VisDrone(NonGeoDataset):
    """VisDrone object detection dataset.

    `VisDrone <https://github.com/VisDrone/VisDrone-Dataset>`__ contains images
    captured by drone-mounted cameras in 14 cities across China. This class supports
    the VisDrone2019-DET image object detection task.

    Dataset features:

    * 10,209 RGB images
    * over 540,000 bounding box annotations
    * 10 object classes

    Dataset format:

    * images are three-channel JPEGs
    * annotations are comma-separated text files with one object per line

    The dataset must be downloaded manually from the dataset website.

    .. versionadded:: 0.11
    """

    classes = (
        'pedestrian',
        'people',
        'bicycle',
        'car',
        'van',
        'truck',
        'tricycle',
        'awning-tricycle',
        'bus',
        'motor',
    )
    directories: ClassVar[dict[str, str]] = {
        'train': 'VisDrone2019-DET-train',
        'val': 'VisDrone2019-DET-val',
        'test': 'VisDrone2019-DET-test-dev',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        """Initialize a new VisDrone dataset instance.

        Args:
            root: root directory where the dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes an input sample and returns a
                transformed version

        Raises:
            AssertionError: if *split* is invalid
            DatasetNotFoundError: if the dataset is not found
        """
        assert split in self.directories

        self.root = root
        self.split = split
        self.transforms = transforms

        directory = os.path.join(self.root, self.directories[split])
        images = sorted(glob.glob(os.path.join(directory, 'images', '*.jpg')))
        if not images:
            raise DatasetNotFoundError(self)

        self.files = [
            (
                image,
                os.path.join(
                    directory,
                    'annotations',
                    f'{os.path.splitext(os.path.basename(image))[0]}.txt',
                ),
            )
            for image in images
        ]

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        image_path, annotation_path = self.files[index]
        with Image.open(image_path) as image:
            array = np.array(image.convert('RGB'))
            tensor = torch.from_numpy(array).permute(2, 0, 1).float()

        boxes, labels = self._load_target(annotation_path)
        sample = {'image': tensor, 'bbox_xyxy': boxes, 'label': labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_target(self, path: Path) -> tuple[Tensor, Tensor]:
        """Load bounding boxes and labels for a single image.

        Args:
            path: path to the annotation file

        Returns:
            bounding boxes in XYXY format and class labels
        """
        boxes = []
        labels = []
        with open(path) as file:
            for line in file:
                x, y, width, height, score, category, _, _ = map(
                    int, line.rstrip().split(',')
                )
                if score == 0 or not 1 <= category <= len(self.classes):
                    continue
                boxes.append([x, y, x + width, y + height])
                labels.append(category - 1)

        return (
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long),
        )
