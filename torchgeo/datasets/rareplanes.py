# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""RarePlanes dataset."""

import glob
import json
import os
from collections.abc import Callable
from typing import ClassVar, Literal, cast
from xml.etree import ElementTree

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.transform import Affine
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class RarePlanes(NonGeoDataset):
    """RarePlanes object detection dataset.

    `RarePlanes <https://github.com/jdc08161063/RarePlanes>`__ contains real
    WorldView-3 satellite imagery and computer-generated aerial imagery of aircraft.
    This class supports both the real tiled imagery and synthetic imagery.

    Dataset features:

    * 253 real WorldView-3 scenes and 50,000 synthetic images
    * over 14,700 real and 630,000 synthetic aircraft annotations
    * one object class: aircraft

    Dataset format:

    * real images are georeferenced PNG tiles with GeoJSON annotations
    * synthetic images are PNGs with XML annotations

    The dataset must be downloaded manually from the dataset website.

    .. versionadded:: 0.11
    """

    classes = ('aircraft',)
    image_directories: ClassVar[dict[str, str]] = {
        'real': 'PS-RGB_tiled',
        'synthetic': 'images',
    }
    annotation_directories: ClassVar[dict[str, str]] = {
        'real': 'geojson_aircraft_tiled',
        'synthetic': 'xmls',
    }
    annotation_extensions: ClassVar[dict[str, str]] = {
        'real': '.geojson',
        'synthetic': '.xml',
    }

    def __init__(
        self,
        root: Path = 'data',
        dataset_type: Literal['real', 'synthetic'] = 'real',
        split: Literal['train', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        """Initialize a new RarePlanes dataset instance.

        Args:
            root: root directory where the dataset can be found
            dataset_type: one of "real" or "synthetic"
            split: one of "train" or "test"
            transforms: a function/transform that takes an input sample and returns a
                transformed version

        Raises:
            AssertionError: if *dataset_type* or *split* is invalid
            DatasetNotFoundError: if the dataset is not found
        """
        assert dataset_type in self.image_directories
        assert split in {'train', 'test'}

        self.root = root
        self.dataset_type = dataset_type
        self.split = split
        self.transforms = transforms

        directory = os.path.join(self.root, dataset_type, split)
        images = sorted(
            glob.glob(
                os.path.join(directory, self.image_directories[dataset_type], '*.png')
            )
        )
        if not images:
            raise DatasetNotFoundError(self)

        annotation_directory = os.path.join(
            directory, self.annotation_directories[dataset_type]
        )
        extension = self.annotation_extensions[dataset_type]
        self.files = [
            (
                image,
                os.path.join(
                    annotation_directory,
                    f'{os.path.splitext(os.path.basename(image))[0]}{extension}',
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
        if self.dataset_type == 'real':
            image, transform = self._load_real_image(image_path)
            boxes = self._load_real_target(annotation_path, transform)
        else:
            image = self._load_synthetic_image(image_path)
            boxes = self._load_synthetic_target(annotation_path)

        labels = torch.zeros(len(boxes), dtype=torch.long)
        sample = {'image': image, 'bbox_xyxy': boxes, 'label': labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_real_image(self, path: Path) -> tuple[Tensor, Affine]:
        """Load a real image and its geotransform.

        Args:
            path: path to the image

        Returns:
            image tensor and geotransform
        """
        with rasterio.open(path) as source:
            return torch.from_numpy(source.read()).float(), source.transform

    def _load_synthetic_image(self, path: Path) -> Tensor:
        """Load a synthetic image.

        Args:
            path: path to the image

        Returns:
            image tensor
        """
        with Image.open(path) as image:
            array = np.array(image.convert('RGB'))
            return torch.from_numpy(array).permute(2, 0, 1).float()

    def _load_real_target(self, path: Path, transform: Affine) -> Tensor:
        """Load GeoJSON aircraft annotations.

        Args:
            path: path to the annotation file
            transform: image geotransform

        Returns:
            bounding boxes in XYXY format
        """
        with open(path) as file:
            annotations = json.load(file)

        inverse = ~transform
        boxes = []
        for feature in annotations['features']:
            coordinates = feature['geometry']['coordinates'][0]
            pixels = [inverse * (x, y) for x, y in coordinates]
            xs, ys = zip(*pixels)
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def _load_synthetic_target(self, path: Path) -> Tensor:
        """Load XML aircraft annotations.

        Args:
            path: path to the annotation file

        Returns:
            bounding boxes in XYXY format
        """
        root = ElementTree.parse(path).getroot()
        boxes = []
        for obj in root.findall('object'):
            if obj.findtext('category0') != 'Airplane':
                continue
            box = cast(ElementTree.Element, obj.find('bndbox2D'))
            boxes.append(
                [
                    int(cast(str, box.findtext('xmin'))),
                    int(cast(str, box.findtext('ymin'))),
                    int(cast(str, box.findtext('xmax'))),
                    int(cast(str, box.findtext('ymax'))),
                ]
            )

        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
