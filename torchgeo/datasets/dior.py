# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DIOR dataset."""

import os
from collections.abc import Callable
from typing import Any, ClassVar, Literal
from xml.etree import ElementTree

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    download_url,
    extract_archive,
)


def parse_pascal_voc(path: Path) -> dict[str, Any]:
    """Read a PASCAL VOC annotation file.

    Args:
        path: path to xml file

    Returns:
        dict of image filename, bounding box coords, and class labels
    """
    et = ElementTree.parse(path)
    element = et.getroot()
    filename = element.find('filename').text  # type: ignore[union-attr]
    labels, bboxes = [], []

    for obj in element.findall('object'):
        bndbox = obj.find('bndbox')
        bbox = [
            int(bndbox.find('xmin').text),  # type: ignore[union-attr, arg-type]
            int(bndbox.find('ymin').text),  # type: ignore[union-attr, arg-type]
            int(bndbox.find('xmax').text),  # type: ignore[union-attr, arg-type]
            int(bndbox.find('ymax').text),  # type: ignore[union-attr, arg-type]
        ]
        label = obj.find('name').text  # type: ignore[union-attr]
        bboxes.append(bbox)
        labels.append(label)

    return dict(filename=filename, bboxes=bboxes, labels=labels)


class DIOR(NonGeoDataset):
    """DIOR dataset.

    `DIOR <https://arxiv.org/abs/1909.00133>`__ dataset contains horizontal bounding box
    annotations of Google Earth Aerial RGB imagery. The test split does not contain bounding
    box annotations and labels.

    Dataset features:

    * 20 classes
    * 192,472 manually annotated bounding box instances

    Dataset format:

    * Images are three channel .jpg files.
    * Annotations are in `Pascal VOC XML format
      <https://roboflow.com/formats/pascal-voc-xml>`_


    Classes:

    0. Airplane
    1. Airport
    2. Baseball Field
    3. Basketball Court
    4. Bridge
    5. Chimney
    6. Dam
    7. Expressway Service Area
    8. Expressway Toll Station
    9. Golf Field
    10. Ground Track Field
    11. Harbor
    12. Overpass
    13. Ship
    14. Stadium
    15. Storage Tank
    16. Tennis Court
    17. Train Station
    18. Vehicle
    19. Windmill


    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1909.00133


    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/torchgeo/dior/resolve/ec7be9567d2e08eb3d3401c15a52ee2145d0ef01/{}'

    files: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        'trainval': {
            'images': {
                'filename': 'Images_trainval.zip',
                'md5': '070e9314120403e5c965d12fe5321cb0',
            },
            'labels': {
                'filename': 'Annotations_trainval.zip',
                'md5': '90e045de37255c5919bbecf659b72c1a',
            },
        },
        'test': {
            'images': {
                'filename': 'Images_test.zip',
                'md5': '97f3cbc86de0867624a6a34190c694ae',
            }
        },
    }

    valid_splits = ('train', 'val', 'test')

    classes = (
        'airplane',
        'airport',
        'baseballfield',
        'basketballcourt',
        'bridge',
        'chimney',
        'dam',
        'expresswayservicearea',
        'expresswaytollstation',
        'golffield',
        'groundtrackfield',
        'harbor',
        'overpass',
        'ship',
        'stadium',
        'storagetank',
        'tenniscourt',
        'trainstation',
        'vehicle',
        'windmill',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DIOR dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of the dataset to use, one of 'train', 'val', 'test'
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found or corrupted and *download* is False.
            AssertionError: If *split* argument is invalid.
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        assert split in self.valid_splits, f'Split must be one of {self.valid_splits}.'
        self.split = split

        self._verify()

        self.sample_df = pd.read_csv(os.path.join(self.root, 'sample_df.csv'))

        self.sample_df = self.sample_df[
            self.sample_df['split'] == self.split
        ].reset_index(drop=True)

        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            idx: index to return

        Returns:
            data and label at that index
        """
        row = self.sample_df.iloc[idx]

        image = self._load_image(os.path.join(self.root, row['image_path']))

        sample: dict[str, Tensor] = {'image': image}

        if self.split != 'test':
            boxes, labels = self._load_target(
                os.path.join(self.root, row['label_path'])
            )
            sample['bbox_xyxy'] = boxes
            sample['label'] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('RGB'))
            tensor: Tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: Path) -> tuple[Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            path: path to the annotation file

        Returns:
            the target bounding boxes and labels
        """
        parsed = parse_pascal_voc(path)
        boxes = torch.tensor(parsed['bboxes'], dtype=torch.float32)
        labels = torch.tensor(
            [self.class_to_idx[label] for label in parsed['labels']]
        ).long()
        return boxes, labels

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        df_path = os.path.join(self.root, 'sample_df.csv')
        exists = []
        if os.path.exists(df_path):
            exists.append(True)
            df = pd.read_csv(df_path)
            df = df[df['split'] == self.split].reset_index(drop=True)
            for idx, row in df.iterrows():
                if os.path.exists(os.path.join(self.root, row['image_path'])):
                    exists.append(True)
                else:
                    exists.append(False)
        else:
            exists.append(False)

        if all(exists):
            return

        exists = []
        if self.split in ['train', 'val']:
            files = self.files['trainval']
        else:
            files = self.files['test']

        for key in files:
            filename = files[key]['filename']
            md5 = files[key]['md5']
            path = os.path.join(self.root, filename)
            if os.path.exists(path):
                if self.checksum and not check_integrity(path, md5):
                    raise RuntimeError('Dataset found, but corrupted.')
                extract_archive(path)
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self.split in ['train', 'val']:
            files = self.files['trainval']
        else:
            files = self.files['test']

        for key in files:
            filename = files[key]['filename']
            md5 = files[key]['md5']
            download_and_extract_archive(
                self.url.format(filename),
                self.root,
                filename=filename,
                md5=md5 if self.checksum else None,
            )

        # download the sample_df.csv file
        download_url(
            self.url.format('sample_df.csv'), self.root, filename='sample_df.csv'
        )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        box_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            box_alpha: alpha value for boxes

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image'].permute((1, 2, 0)).numpy()
        boxes = sample['bbox_xyxy'].numpy()
        labels = sample['label'].numpy()

        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))

        axs.imshow(image)
        axs.axis('off')

        cm = plt.get_cmap('gist_rainbow')

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

            # Horizontal box: [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=box_alpha,
                linestyle='solid',
                edgecolor=color,
                facecolor='none',
            )
            axs.add_patch(rect)
            # Add label above box
            axs.text(
                x1,
                y1 - 5,
                label,
                color='white',
                fontsize=8,
                bbox=dict(facecolor=color, alpha=box_alpha),
            )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
