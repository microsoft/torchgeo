# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""SentinelKilnDB dataset."""

import io
import os
from collections.abc import Callable
from typing import ClassVar, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url, quantile_normalization


class _FileInfo(TypedDict):
    filename: str
    sha256: str | None


class SentinelKilnDB(NonGeoDataset):
    """SentinelKilnDB brick kiln detection dataset.

    *SentinelKilnDB: A Large-Scale Dataset and Benchmark for OBB Brick Kiln Detection in South Asia Using Satellite Imagery.*
    Accepted at NeurIPS 2025.

    `SentinelKilnDB <https://huggingface.co/datasets/SustainabilityLabIITGN/SentinelKilnDB>`__
    is a dataset for brick kiln detection using Sentinel-2 satellite imagery from
    South Asia. It contains over 114K images (128x128 RGB) with bounding box annotations
    for three types of brick kilns.

    Dataset features:

    * 114K+ RGB images (128x128 pixels) from Sentinel-2
    * Multi-class object detection (3 kiln classes)
    * 62,671 total brick kiln instances
    * 41,068 negative samples (tiles with no kilns)
    * Axis-aligned (AA) and oriented (OBB) bounding box annotations in YOLO format

    Dataset classes:

    0. CFCBK (Clamp Fired Country Brick Kiln)
    1. FCBK (Fixed Chimney Bull Trench Kiln)
    2. Zigzag (Zigzag Kiln)

    If you use this dataset in your research, please cite the original paper:

    * https://openreview.net/forum?id=efGzsxVSEC

    .. note::

       This dataset requires the following additional library to be installed:

       * `pyarrow <https://pypi.org/project/pyarrow/>`_: to load parquet files

    .. versionadded:: 0.10
    """

    url = 'https://huggingface.co/datasets/SustainabilityLabIITGN/SentinelKilnDB/resolve/main/{0}/{0}.parquet'

    file_info: ClassVar[dict[str, _FileInfo]] = {
        'train': {
            'filename': 'train.parquet',
            'sha256': '90107b0e4e922f3d84c2996acd2bb371f36dbbd3cdd9b305057bd44e9a62c484',
        },
        'val': {
            'filename': 'val.parquet',
            'sha256': 'e8f92d816ebc88391d0bc6532a7bef775e242951f1d267553720fedd6f7be57e',
        },
        'test': {
            'filename': 'test.parquet',
            'sha256': '8a26ecf01f00ca45cffb741fbe4e8067ef601f6e33fb6905931d1f87c181cd4d',
        },
    }

    classes = ('CFCBK', 'FCBK', 'Zigzag')

    valid_splits = ('train', 'val', 'test')
    valid_orientations = ('horizontal', 'oriented')

    # Fixed image size for this dataset
    image_size = 128

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        bbox_orientation: Literal['horizontal', 'oriented'] = 'horizontal',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SentinelKilnDB dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of the dataset to use, one of ['train', 'val', 'test']
            bbox_orientation: bounding box orientation, one of ['horizontal', 'oriented'],
                where horizontal returns xyxy format and oriented returns
                x1y1x2y2x3y3x4y4 format
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if *split* or *bbox_orientation* argument is not valid
            DatasetNotFoundError: if dataset is not found and *download* is False
        """
        assert split in self.valid_splits, (
            f"Split '{split}' not supported, use one of {self.valid_splits}"
        )
        assert bbox_orientation in self.valid_orientations, (
            f'Bounding box orientation must be one of {self.valid_orientations}'
        )

        self.root = root
        self.split = split
        self.bbox_orientation = bbox_orientation
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        # Load parquet file
        parquet_path = os.path.join(self.root, self.file_info[self.split]['filename'])
        self.df = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.df)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        row = self.df.iloc[index]

        sample: Sample = {'image': self._load_image(row)}

        boxes, labels = self._load_target(row)

        if self.bbox_orientation == 'horizontal':
            sample['bbox_xyxy'] = boxes
        else:
            sample['bbox'] = boxes
        sample['labels'] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, row: pd.Series) -> Tensor:
        """Load image from parquet row.

        Args:
            row: a row from the parquet DataFrame

        Returns:
            image tensor of shape (C, H, W)
        """
        img_bytes = row['image']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()

    def _load_target(self, row: pd.Series) -> tuple[Tensor, Tensor]:
        """Load bounding box annotations from parquet row.

        Args:
            row: a row from the parquet DataFrame

        Returns:
            tuple of:
                boxes: tensor of shape (N, 4) for horizontal or (N, 8) for oriented
                labels: tensor of shape (N,) with class indices
        """
        if self.bbox_orientation == 'horizontal':
            label_array = row['yolo_aa_label']
            return self._parse_yolo_aa(label_array)
        else:
            label_array = row['yolo_obb_label']
            return self._parse_yolo_obb(label_array)

    def _parse_yolo_aa(
        self, label_array: np.ndarray[tuple[int], np.dtype[np.str_]]
    ) -> tuple[Tensor, Tensor]:
        """Parse YOLO axis-aligned bounding box labels.

        Format: class_id x_center y_center width height (normalized 0-1)
        Converts to: xmin, ymin, xmax, ymax (pixel coordinates)

        Args:
            label_array: array of YOLO AA label strings

        Returns:
            tuple of boxes (N, 4) and labels (N,)
        """
        boxes = []
        labels = []

        if len(label_array) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        for line in label_array:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * self.image_size
            y_center = float(parts[2]) * self.image_size
            width = float(parts[3]) * self.image_size
            height = float(parts[4]) * self.image_size

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.long
        )

    def _parse_yolo_obb(
        self, label_array: np.ndarray[tuple[int], np.dtype[np.str_]]
    ) -> tuple[Tensor, Tensor]:
        """Parse YOLO oriented bounding box labels.

        Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
        Converts to: x1 y1 x2 y2 x3 y3 x4 y4 (pixel coordinates)

        Args:
            label_array: array of YOLO OBB label strings

        Returns:
            tuple of boxes (N, 8) and labels (N,)
        """
        boxes = []
        labels = []

        if len(label_array) == 0:
            return (
                torch.zeros((0, 8), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        for line in label_array:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            class_id = int(parts[0])
            coords = [float(p) * self.image_size for p in parts[1:9]]

            boxes.append(coords)
            labels.append(class_id)

        if not boxes:
            return (
                torch.zeros((0, 8), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.long
        )

    def _verify(self) -> None:
        """Verify dataset integrity and download if needed."""
        parquet_path = os.path.join(self.root, self.file_info[self.split]['filename'])

        if os.path.exists(parquet_path):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)

        filename = self.file_info[self.split]['filename']

        if self.checksum:
            sha256 = self.file_info[self.split]['sha256']
            assert sha256 is not None
            download_url(
                url=self.url.format(self.split),
                root=self.root,
                filename=filename,
                sha256=sha256,
            )
        else:
            download_url(
                url=self.url.format(self.split), root=self.root, filename=filename
            )

    def plot(
        self,
        sample: Sample,
        show_titles: bool = True,
        suptitle: str | None = None,
        box_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by __getitem__
            show_titles: flag indicating whether to show titles
            suptitle: optional string to use as a suptitle
            box_alpha: alpha value for boxes

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = quantile_normalization(sample['image'].permute(1, 2, 0)).numpy()
        if self.bbox_orientation == 'horizontal':
            boxes = sample['bbox_xyxy'].cpu().numpy()
        else:
            boxes = sample['bbox'].cpu().numpy()
        labels = sample['labels'].cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.axis('off')

        # Create color map for classes
        cm = plt.get_cmap('gist_rainbow')

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

            if self.bbox_orientation == 'horizontal':
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
                ax.add_patch(rect)
                # Add label above box
                ax.text(
                    x1,
                    y1 - 2,
                    label,
                    color='white',
                    fontsize=6,
                    bbox=dict(facecolor=color, alpha=box_alpha),
                )
            else:
                # Oriented box: [x1,y1,x2,y2,x3,y3,x4,y4]
                vertices = box.reshape(4, 2)
                polygon = patches.Polygon(
                    vertices,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle='solid',
                    edgecolor=color,
                    facecolor='none',
                )
                ax.add_patch(polygon)
                # Add label at centroid
                centroid_x = vertices[:, 0].mean()
                centroid_y = vertices[:, 1].mean()
                ax.text(
                    centroid_x,
                    centroid_y,
                    label,
                    color='white',
                    fontsize=6,
                    bbox=dict(facecolor=color, alpha=box_alpha),
                    ha='center',
                    va='center',
                )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
