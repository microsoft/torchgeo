# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DOTA dataset."""

import os
import pandas as pd
from collections.abc import Callable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    download_url,
    lazy_import,
    percentile_normalization,
)


class DOTA(NonGeoDataset):
    """DOTA dataset.

    The `DOTA <https://captain-whu.github.io/DOTA/index.html>`_ is a large-scale object
    detection dataset for aerial imagery containing RGB and gray-scale imagery from Google Earth, GF-2 and JL-1 satellites
    as well as additional aerial imagery from CycloMedia.

    Dataset features:

    * multi-class object detection (15 classes in V1 and 18 classes in V2)
    * horizontal and oriented bounding boxes

    Dataset format:

    * images are three channel PNGs with various pixel sizes
    * annotations are text files with one line per bounding box

    Classes:

    * plane
    * ship
    * storage-tank
    * baseball-diamond
    * tennis-court
    * basketball-court
    * ground-track-field
    * harbor
    * bridge
    * large-vehicle
    * small-vehicle
    * helicopter
    * roundabout
    * soccer-ball-field
    * swimming-pool
    * container-crane
    * airport
    + helipad


    If you use this work in your research, please cite the following paper:

    * https://arxiv.org/abs/2102.12219
    * https://arxiv.org/abs/1711.10398

    .. versionadded:: 0.7
    """

    url = 'https://huggingface.co/datasets/torchgeo/dota/resolve/main/{}'

    file_info: ClassVar[dict[str, dict[str, dict[str, dict[str, str]]]]] = {
        'train': {
            'images': {
                '1.0': {'filename': 'dotav1_images_train.tar.gz', 'md5': ''},
                '2.0': {'filename': 'dotav2_images_train.tar.gz', 'md5': ''},
            },
            'annotations': {
                '1.0': {'filename': 'dotav1_annotations_train.tar.gz', 'md5': ''},
                '2.0': {'filename': 'dotav2_annotations_train.tar.gz', 'md5': ''},
            },
        },
        'val': {
            'images': {
                '1.0': {'filename': 'dotav1_images_val.tar.gz', 'md5': ''},
                '2.0': {'filename': 'dotav2_images_val.tar.gz', 'md5': ''},
            },
            'annotations': {
                '1.0': {'filename': 'dotav1_annotations_val.tar.gz', 'md5': ''},
                '2.0': {'filename': 'dotav2_annotations_val.tar.gz', 'md5': ''},
            },
        },
    }

    sample_df_path = 'samples.parquet'

    classes = (
        'plane',
        'ship',
        'storage-tank',
        'baseball-diamond',
        'tennis-court',
        'basketball-court',
        'ground-track-field',
        'harbor',
        'bridge',
        'large-vehicle',
        'small-vehicle',
        'helicopter',
        'roundabout',
        'soccer-ball-field',
        'swimming-pool',
        'container-crane',
        'airport',
        'helipad',
    )

    valid_splits = ('train', 'val')
    valid_versions = ('1.0', '2.0')

    valid_orientations = ('hbb', 'obb')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        version: str = '2.0',
        bbox_orientation: str = 'hbb',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DOTA dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of the dataset to use, one of ['train', 'val']
            version: version of the dataset to use, one of ['1.0', '2.0']
            bbox_orientation: bounding box orientation, one of ['hbb', 'obb'], meaning horizontal
                or oriented bounding boxes, hbb only available for v2.0
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if *split*, *version*, or *bbox_orientation* argument are not valid
            DatasetNotFoundError: if dataset is not found or corrupted
        """

        assert split in self.valid_splits, (
            f"Split '{split}' not supported, use one of {self.valid_splits}"
        )
        assert version in self.valid_versions, (
            f"Version '{version}' not supported, use one of {self.valid_versions}"
        )

        if version == '1.0':
            assert bbox_orientation == 'hbb', (
                "Bounding box orientation must be 'hbb' for version 1.0"
            )
        elif version == '2.0':
            assert bbox_orientation in self.valid_orientations, (
                f'Bounding box orientation must be one of {self.valid_orientations}'
            )

        self.root = root
        self.split = split
        self.version = version
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.bbox_orientation = bbox_orientation

        self._verify()

        self.sample_df = pd.read_parquet(os.path.join(self.root, 'samples.parquet'))
        self.sample_df = self.sample_df[self.sample_df['split'] == self.split]
        self.sample_df = self.sample_df[
            self.sample_df['version'] == self.version
        ].reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample_row = self.sample_df.iloc[index]

        sample = {'image': self._load_image(sample_row['image_path'])}

        if self.bbox_orientation == 'obb':
            boxes, labels = self._load_annotations(sample_row['annotation_path'])
        else:
            boxes, labels = self._load_annotations(sample_row['annotation_path'])

        sample['boxes'] = boxes
        sample['labels'] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> Tensor:
        """Load image.

        Args:
            path: path to image file

        Returns:
            image: image tensor
        """
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        return torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()

    def _load_annotations(self, path: str) -> tuple[Tensor, Tensor]:
        """Load DOTA annotations from text file.

        Format:
            x1 y1 x2 y2 x3 y3 x4 y4 class difficult

        Some files have 2 header lines that need to be skipped:
            imagesource:GoogleEarth
            gsd:0.146343590398

        Args:
            path: path to annotation file
        Returns:
            tuple of:
                boxes: tensor of shape (N, 8) with coordinates for obb
                    and (N, 4) for hbb
                labels: tensor of shape (N,) with class indices
        """
        with open(os.path.join(self.root, path), 'r') as f:
            lines = f.readlines()

        # Skip header if present
        start_idx = 0
        if lines and lines[0].startswith('imagesource'):
            start_idx = 2
        boxes = []
        labels = []

        for line in lines[start_idx:]:
            parts = line.strip().split(' ')

            # Always read 8 coordinates
            coords = [float(p) for p in parts[:8]]
            label = parts[8]

            labels.append(self.classes.index(label))

            if self.bbox_orientation == 'hbb':
                # Convert to [xmin, ymin, xmax, ymax] format
                x_coords = coords[::2]  # even indices (0,2,4,6)
                y_coords = coords[1::2]  # odd indices (1,3,5,7)
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                boxes.append(coords)

        if not boxes:
            return (
                torch.zeros((0, 4 if self.bbox_orientation == 'hbb' else 8)),
                torch.zeros(0, dtype=torch.long),
            )

        return torch.tensor(boxes), torch.tensor(labels)

    def _verify(self) -> None:
        """Verify dataset integrity and download/extract if needed."""
        # check if directories and sample file are present
        required_dirs = [
            os.path.join(self.root, self.split, 'images'),
            os.path.join(self.root, self.split, 'annotations'),
            os.path.join(self.root, self.sample_df_path),
        ]
        if all(os.path.exists(d) for d in required_dirs):
            return

        # Check for compressed files
        files_needed = [
            self.file_info[self.split]['images'][self.version]['filename'],
            self.file_info[self.split]['annotations'][self.version]['filename'],
        ]

        # For v2.0, also need v1.0 files, but only v2 annotations
        if self.version == '1.0':
            files_needed = [
                self.file_info[self.split]['images']['1.0']['filename'],
                self.file_info[self.split]['annotations']['1.0']['filename'],
            ]
        elif self.version == '2.0':
            files_needed = [
                self.file_info[self.split]['images']['1.0']['filename'],
                self.file_info[self.split]['images']['2.0']['filename'],
                self.file_info[self.split]['annotations']['2.0']['filename'],
            ]

        # Check if archives exist and verify checksums
        exists = []
        for filename in files_needed:
            filepath = os.path.join(self.root, filename)
            print('FILEPATH', filepath)
            if os.path.exists(filepath):
                if self.checksum:
                    md5 = self.file_info[self.split]['images'][self.version]['md5']
                    if not check_integrity(filepath, md5):
                        raise RuntimeError(f'Archive {filename} corrupted')
                exists.append(True)

            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract dataset
        for filename in files_needed:
            if not os.path.exists(os.path.join(self.root, filename)):
                md5 = self.file_info[self.split]['images'][self.version]['md5']
                download_and_extract_archive(
                    url=self.url.format(filename),
                    download_root=self.root,
                    filename=filename,
                    remove_finished=False,
                    md5=None if not self.checksum else md5,
                )
        # also download the metadata file
        if not os.path.exists(os.path.join(self.root, self.sample_df_path)):
            download_url(
                url=self.url.format(self.sample_df_path),
                root=self.root,
                filename=self.sample_df_path,
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
            sample: a sample returned by __getitem__
            show_titles: flag indicating whether to show titles
            suptitle: optional string to use as a suptitle
            box_alpha: alpha value for boxes

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = percentile_normalization(sample['image'].permute(1, 2, 0).numpy())
        boxes = sample['boxes'].cpu().numpy()
        labels = sample['labels'].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        # Create color map for classes
        cm = plt.get_cmap('gist_rainbow')

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

            if self.bbox_orientation == 'hbb':
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
                    y1 - 5,
                    label,
                    color='white',
                    fontsize=8,
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
                    fontsize=8,
                    bbox=dict(facecolor=color, alpha=box_alpha),
                    ha='center',
                    va='center',
                )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
