# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SODA datasets."""

import json
import os
from collections.abc import Callable
from typing import ClassVar

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
from .utils import Path, check_integrity, download_and_extract_archive, download_url


class SODAA(NonGeoDataset):
    """SODA-A dataset.

    The `SODA-A <https://shaunyuan22.github.io/SODA/>`_ dataset is a high resolution aerial imagery dataset for small object detection.

    Dataset features:

    * 2513 images
    * 872,069 annotations with oriented bounding boxes
    * 9 classes

    Dataset format:

    * Images are three channel .jpg files.
    * Annotations are in json files

    Classes:

    0. Airplane
    1. Helicopter
    2. Small vehicle
    3. Large vehicle
    4. Ship
    5. Container
    6. Storage tank
    7. Swimming-pool
    8. Windmill
    9. Other

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/10168277

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/torchgeo/soda-a/resolve/5ccad7b7147381b06fb969f95c3ffd8bf71208b9/{}'

    files: ClassVar[dict[str, dict[str, str]]] = {
        'images': {
            'filename': 'Images.zip',
            'md5sum': '8ee4ad7a306b0a0a900fa78a4f6aae68',
        },
        'labels': {
            'filename': 'Annotations.zip',
            'md5sum': '45b0d21209fc332d89b0144b308e57fa',
        },
    }

    classes = (
        'airplane',
        'helicopter',
        'small-vehicle',
        'large-vehicle',
        'ship',
        'container',
        'storage-tank',
        'swimming-pool',
        'windmill',
        'other',
    )

    valid_splits = ('train', 'val', 'test')

    valid_orientations = ('oriented', 'horizontal')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bbox_orientation: str = 'oriented',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new instance of SODA-A dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bbox_orientation: one of "oriented" or "horizontal"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` or ``bbox_orientation`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.valid_splits, f'split must be one of {self.valid_splits}'

        assert bbox_orientation in self.valid_orientations, (
            f'bbox_orientation must be one of {self.valid_orientations}'
        )

        self.root = root
        self.split = split
        self.bbox_orientation = bbox_orientation
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.sample_df = pd.read_parquet(os.path.join(self.root, 'sample_df.parquet'))
        self.sample_df = self.sample_df[
            self.sample_df['split'] == self.split
        ].reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the sample at the given index.

        Args:
            idx: index of the sample to return

        Returns:
            the sample at the given index
        """
        row = self.sample_df.iloc[idx]

        image = self._load_image(os.path.join(self.root, row['image_path']))
        boxes, labels = self._load_labels(os.path.join(self.root, row['label_path']))

        sample: dict[str, Tensor] = {'image': image, 'label': labels}

        if self.bbox_orientation == 'oriented':
            sample['boxes'] = boxes
        else:
            sample['boxes_xyxy'] = boxes

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> Tensor:
        """Load an image from disk.

        Args:
            path: path to the image file

        Returns:
            the image as a tensor
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('RGB'))
            tensor: Tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_labels(self, path: str) -> tuple[Tensor, Tensor]:
        """Load labels from disk.

        Args:
            path: path to the label file

        Returns:
            tuple of:
                - boxes: tensor of bounding boxes in XYXY format [N, 4]
                - labels: tensor of class labels [N]
        """
        with open(path) as f:
            data = json.load(f)

        boxes = []
        labels = []

        for ann in data['annotations']:
            # Extract polygon points
            coords = ann['poly']

            # Convert to axis-aligned bounding box
            if self.bbox_orientation == 'horizontal':
                x_coords = coords[::2]  # even indices (0,2,4,6)
                y_coords = coords[1::2]  # odd indices (1,3,5,7)
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                boxes.append(coords)
            labels.append(ann['category_id'])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return boxes_tensor, labels_tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        exists = []
        df_path = os.path.join(self.root, 'sample_df.parquet')

        if os.path.exists(df_path):
            exists.append(True)
            df = pd.read_parquet(df_path)
            df = df[df['split'] == self.split].reset_index(drop=True)
            for idx, row in df.iterrows():
                image_path = os.path.join(self.root, row['image_path'])
                label_path = os.path.join(self.root, row['label_path'])
                exists.append(os.path.exists(image_path) and os.path.exists(label_path))
        else:
            exists.append(False)

        if all(exists):
            return

        exists = []

        for file in self.files.values():
            archive_path = os.path.join(self.root, file['filename'])
            if os.path.exists(archive_path):
                if self.checksum and not check_integrity(archive_path, file['md5sum']):
                    raise RuntimeError('Dataset found, but corrupted.')
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for file in self.files.values():
            download_and_extract_archive(
                self.url.format(file['filename']),
                self.root,
                filename=file['filename'],
                md5=file['md5sum'] if self.checksum else None,
            )

        # also download the sample_df
        download_url(
            self.url.format('sample_df.parquet'),
            self.root,
            filename='sample_df.parquet',
        )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        box_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset with legend.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            box_alpha: alpha value for boxes

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image'].permute((1, 2, 0)).numpy()
        if self.bbox_orientation == 'horizontal':
            boxes = sample['boxes_xyxy'].numpy()
        else:
            boxes = sample['boxes'].numpy()
        labels = sample['label'].numpy()

        fig, ax = plt.subplots(ncols=1, figsize=(10, 10))

        ax.imshow(image)
        ax.axis('off')

        cm = plt.get_cmap('gist_rainbow')

        unique_labels = set()
        legend_elements = []

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

            if label not in unique_labels:
                legend_elements.append(
                    patches.Patch(facecolor=color, alpha=box_alpha, label=label)
                )
                unique_labels.add(label)

        ax.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=len(legend_elements),
            mode='expand',
        )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
