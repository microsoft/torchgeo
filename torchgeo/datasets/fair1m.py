# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FAIR1M dataset."""

import glob
import os
from collections.abc import Callable
from typing import ClassVar, cast
from xml.etree.ElementTree import Element, parse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, check_integrity, download_url, extract_archive


def parse_pascal_voc(path: Path) -> Sample:
    """Read a PASCAL VOC annotation file.

    Args:
        path: path to xml file

    Returns:
        dict of image filename, points, and class labels
    """
    et = parse(path)
    element = et.getroot()
    source = cast(Element, element.find('source'))
    filename = cast(Element, source.find('filename')).text
    labels, points = [], []
    objects = cast(Element, element.find('objects'))
    for obj in objects.findall('object'):
        elm_points = cast(Element, obj.find('points'))
        lis_points = elm_points.findall('point')
        str_points = []
        for point in lis_points:
            text = cast(str, point.text)
            str_points.append(text.split(','))
        tup_points = [(float(p1), float(p2)) for p1, p2 in str_points]
        possibleresult = cast(Element, obj.find('possibleresult'))
        name = cast(Element, possibleresult.find('name'))
        label = name.text
        labels.append(label)
        points.append(tup_points)
    return dict(filename=filename, points=points, labels=labels)


class FAIR1M(NonGeoDataset):
    """FAIR1M dataset.

    The `FAIR1M <https://www.gaofen-challenge.com/benchmark>`__
    dataset is a dataset for remote sensing fine-grained oriented object detection.

    Dataset features:

    * 15,000+ images with 0.3-0.8 m per pixel resolution (1,000-10,000 px)
    * 1 million object instances
    * 5 object categories, 37 object sub-categories
    * three spectral bands - RGB
    * images taken by Gaofen satellites and Google Earth

    Dataset format:

    * images are three-channel tiffs
    * labels are xml files with PASCAL VOC like annotations

    Dataset classes:

    0. Passenger Ship
    1. Motorboat
    2. Fishing Boat
    3. Tugboat
    4. other-ship
    5. Engineering Ship
    6. Liquid Cargo Ship
    7. Dry Cargo Ship
    8. Warship
    9. Small Car
    10. Bus
    11. Cargo Truck
    12. Dump Truck
    13. other-vehicle
    14. Van
    15. Trailer
    16. Tractor
    17. Excavator
    18. Truck Tractor
    19. Boeing737
    20. Boeing747
    21. Boeing777
    22. Boeing787
    23. ARJ21
    24. C919
    25. A220
    26. A321
    27. A330
    28. A350
    29. other-airplane
    30. Baseball Field
    31. Basketball Court
    32. Football Field
    33. Tennis Court
    34. Roundabout
    35. Intersection
    36. Bridge

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.isprsjprs.2021.12.004

    .. versionadded:: 0.2
    """

    classes: ClassVar[dict[str, Sample]] = {
        'Passenger Ship': {'id': 0, 'category': 'Ship'},
        'Motorboat': {'id': 1, 'category': 'Ship'},
        'Fishing Boat': {'id': 2, 'category': 'Ship'},
        'Tugboat': {'id': 3, 'category': 'Ship'},
        'other-ship': {'id': 4, 'category': 'Ship'},
        'Engineering Ship': {'id': 5, 'category': 'Ship'},
        'Liquid Cargo Ship': {'id': 6, 'category': 'Ship'},
        'Dry Cargo Ship': {'id': 7, 'category': 'Ship'},
        'Warship': {'id': 8, 'category': 'Ship'},
        'Small Car': {'id': 9, 'category': 'Vehicle'},
        'Bus': {'id': 10, 'category': 'Vehicle'},
        'Cargo Truck': {'id': 11, 'category': 'Vehicle'},
        'Dump Truck': {'id': 12, 'category': 'Vehicle'},
        'other-vehicle': {'id': 13, 'category': 'Vehicle'},
        'Van': {'id': 14, 'category': 'Vehicle'},
        'Trailer': {'id': 15, 'category': 'Vehicle'},
        'Tractor': {'id': 16, 'category': 'Vehicle'},
        'Excavator': {'id': 17, 'category': 'Vehicle'},
        'Truck Tractor': {'id': 18, 'category': 'Vehicle'},
        'Boeing737': {'id': 19, 'category': 'Airplane'},
        'Boeing747': {'id': 20, 'category': 'Airplane'},
        'Boeing777': {'id': 21, 'category': 'Airplane'},
        'Boeing787': {'id': 22, 'category': 'Airplane'},
        'ARJ21': {'id': 23, 'category': 'Airplane'},
        'C919': {'id': 24, 'category': 'Airplane'},
        'A220': {'id': 25, 'category': 'Airplane'},
        'A321': {'id': 26, 'category': 'Airplane'},
        'A330': {'id': 27, 'category': 'Airplane'},
        'A350': {'id': 28, 'category': 'Airplane'},
        'other-airplane': {'id': 29, 'category': 'Airplane'},
        'Baseball Field': {'id': 30, 'category': 'Court'},
        'Basketball Court': {'id': 31, 'category': 'Court'},
        'Football Field': {'id': 32, 'category': 'Court'},
        'Tennis Court': {'id': 33, 'category': 'Court'},
        'Roundabout': {'id': 34, 'category': 'Road'},
        'Intersection': {'id': 35, 'category': 'Road'},
        'Bridge': {'id': 36, 'category': 'Road'},
    }

    filename_glob: ClassVar[dict[str, str]] = {
        'train': os.path.join('train', '**', 'images', '*.tif'),
        'val': os.path.join('validation', 'images', '*.tif'),
        'test': os.path.join('test', 'images', '*.tif'),
    }
    directories: ClassVar[dict[str, tuple[str, ...]]] = {
        'train': (
            os.path.join('train', 'part1', 'images'),
            os.path.join('train', 'part1', 'labelXml'),
            os.path.join('train', 'part2', 'images'),
            os.path.join('train', 'part2', 'labelXml'),
        ),
        'val': (
            os.path.join('validation', 'images'),
            os.path.join('validation', 'labelXml'),
        ),
        'test': (os.path.join('test', 'images'),),
    }
    paths: ClassVar[dict[str, tuple[str, ...]]] = {
        'train': (
            os.path.join('train', 'part1', 'images.zip'),
            os.path.join('train', 'part1', 'labelXml.zip'),
            os.path.join('train', 'part2', 'images.zip'),
            os.path.join('train', 'part2', 'labelXmls.zip'),
        ),
        'val': (
            os.path.join('validation', 'images.zip'),
            os.path.join('validation', 'labelXmls.zip'),
        ),
        'test': (
            os.path.join('test', 'images0.zip'),
            os.path.join('test', 'images1.zip'),
            os.path.join('test', 'images2.zip'),
        ),
    }
    urls: ClassVar[dict[str, tuple[str, ...]]] = {
        'train': (
            'https://drive.google.com/file/d/1LWT_ybL-s88Lzg9A9wHpj0h2rJHrqrVf',
            'https://drive.google.com/file/d/1CnOuS8oX6T9JMqQnfFsbmf7U38G6Vc8u',
            'https://drive.google.com/file/d/1cx4MRfpmh68SnGAYetNlDy68w0NgKucJ',
            'https://drive.google.com/file/d/1RFVjadTHA_bsB7BJwSZoQbiyM7KIDEUI',
        ),
        'val': (
            'https://drive.google.com/file/d/1lSSHOD02B6_sUmr2b-R1iqhgWRQRw-S9',
            'https://drive.google.com/file/d/1sTTna1C5n3Senpfo-73PdiNilnja1AV4',
        ),
        'test': (
            'https://drive.google.com/file/d/1HtOOVfK9qetDBjE7MM0dK_u5u7n4gdw3',
            'https://drive.google.com/file/d/1iXKCPmmJtRYcyuWCQC35bk97NmyAsasq',
            'https://drive.google.com/file/d/1oUc25FVf8Zcp4pzJ31A1j1sOLNHu63P0',
        ),
    }
    md5s: ClassVar[dict[str, tuple[str, ...]]] = {
        'train': (
            'a460fe6b1b5b276bf856ce9ac72d6568',
            '80f833ff355f91445c92a0c0c1fa7414',
            'ad237e61dba304fcef23cd14aa6c4280',
            '5c5948e68cd0f991a0d73f10956a3b05',
        ),
        'val': ('dce782be65405aa381821b5f4d9eac94', '700b516a21edc9eae66ca315b72a09a1'),
        'test': (
            'fb8ccb274f3075d50ac9f7803fbafd3d',
            'dc9bbbdee000e97f02276aa61b03e585',
            '700b516a21edc9eae66ca315b72a09a1',
        ),
    }
    image_root: str = 'images'
    label_root: str = 'labelXml'

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new FAIR1M dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           Added *split* and *download* parameters.
        """
        assert split in self.directories
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()
        self.files = sorted(
            glob.glob(os.path.join(self.root, self.filename_glob[split]))
        )

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        path = self.files[index]

        image = self._load_image(path)
        sample: Sample = {'image': image}

        if self.split != 'test':
            label_path = str(path).replace(self.image_root, self.label_root)
            label_path = label_path.replace('.tif', '.xml')
            voc = parse_pascal_voc(label_path)
            boxes, labels = self._load_target(voc['points'], voc['labels'])
            sample = {'image': image, 'boxes': boxes, 'label': labels}

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
            path: path to image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('RGB'))
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(
        self, points: list[list[tuple[float, float]]], labels: list[str]
    ) -> tuple[Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            points: list of point tuple lists
            labels: list of class labels

        Returns:
            the target bounding boxes and labels
        """
        labels_list = [self.classes[label]['id'] for label in labels]
        boxes = torch.tensor(points).to(torch.float)
        labels_tensor = torch.tensor(labels_list)
        return boxes, labels_tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the directories already exist
        exists = []
        for directory in self.directories[self.split]:
            exists.append(os.path.exists(os.path.join(self.root, directory)))
        if all(exists):
            return

        # Check if .zip files already exists (if so extract)
        exists = []
        paths = self.paths[self.split]
        md5s = self.md5s[self.split]
        for path, md5 in zip(paths, md5s):
            filepath = os.path.join(self.root, path)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError('Dataset found, but corrupted.')
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        if self.download:
            self._download()
            return

        raise DatasetNotFoundError(self)

    def _download(self) -> None:
        """Download the dataset and extract it."""
        paths = self.paths[self.split]
        urls = self.urls[self.split]
        md5s = self.md5s[self.split]
        for directory in self.directories[self.split]:
            os.makedirs(os.path.join(self.root, directory), exist_ok=True)

        for path, url, md5 in zip(paths, urls, md5s):
            filepath = os.path.join(self.root, path)
            if not os.path.exists(filepath):
                download_url(
                    url=url,
                    root=os.path.dirname(filepath),
                    filename=os.path.basename(filepath),
                    md5=md5 if self.checksum else None,
                )
                extract_archive(filepath)

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
        image = sample['image'].permute((1, 2, 0)).numpy()

        ncols = 1
        if 'prediction_boxes' in sample:
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols < 2:
            axs = [axs]

        axs[0].imshow(image)
        axs[0].axis('off')

        if 'boxes' in sample:
            polygons = [
                patches.Polygon(points, color='r', fill=False)
                for points in sample['boxes'].numpy()
            ]
            for polygon in polygons:
                axs[0].add_patch(polygon)

        if show_titles:
            axs[0].set_title('Ground Truth')

        if ncols > 1:
            axs[1].imshow(image)
            axs[1].axis('off')
            polygons = [
                patches.Polygon(points, color='r', fill=False)
                for points in sample['prediction_boxes'].numpy()
            ]
            for polygon in polygons:
                axs[0].add_patch(polygon)

            if show_titles:
                axs[1].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
