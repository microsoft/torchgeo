# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DOTA dataset."""

import os
from collections.abc import Callable
from typing import Any, ClassVar

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
from .utils import (
    Path,
    check_integrity,
    download_url,
    extract_archive,
    percentile_normalization,
)


class DOTA(NonGeoDataset):
    """DOTA dataset.

    The `DOTA <https://captain-whu.github.io/DOTA/index.html>`__ is a large-scale object
    detection dataset for aerial imagery containing RGB and gray-scale imagery from Google Earth, GF-2 and JL-1 satellites
    as well as additional aerial imagery from CycloMedia. There are three versions of the dataset: V1.0, V1.5, and V2.0, where,
    V1.0 and V1.5 have the same images but different annotations, and V2.0 extends both the images and annotations with more samples

    Dataset features:

    * 1869 samples in V1.0 and V.1.5 and 2423 samples in V2.0
    * multi-class object detection (15 classes in V1.0 and V1.5 and 18 classes in V2.0)
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
    * helipad


    If you use this work in your research, please cite the following papers:

    * https://arxiv.org/abs/2102.12219
    * https://arxiv.org/abs/1711.10398

    .. versionadded:: 0.7
    """

    url = 'https://huggingface.co/datasets/torchgeo/dota/resolve/672e63236622f7da6ee37fca44c50ac368b77cab/{}'

    file_info: ClassVar[dict[str, dict[str, dict[str, dict[str, str]]]]] = {
        'train': {
            'images': {
                '1.0': {
                    'filename': 'dotav1.0_images_train.tar.gz',
                    'md5': '363b472dc3c71e7fa2f4a60223b437ea',
                },
                '1.5': {
                    'filename': 'dotav1.0_images_train.tar.gz',
                    'md5': '363b472dc3c71e7fa2f4a60223b437ea',
                },
                '2.0': {
                    'filename': 'dotav2.0_images_train.tar.gz',
                    'md5': '91ae5212d170330ab9f65ccb6c675763',
                },
            },
            'annotations': {
                '1.0': {
                    'filename': 'dotav1.0_annotations_train.tar.gz',
                    'md5': 'f6788257bcc4d29018344a4128e3734a',
                },
                '1.5': {
                    'filename': 'dotav1.5_annotations_train.tar.gz',
                    'md5': '0da97e5623a87d7bec22e75f6978dbce',
                },
                '2.0': {
                    'filename': 'dotav2.0_annotations_train.tar.gz',
                    'md5': '04d3d626df2203053b7f06581b3b0667',
                },
            },
        },
        'val': {
            'images': {
                '1.0': {
                    'filename': 'dotav1.0_images_val.tar.gz',
                    'md5': '42293219ba61d61c417ae558bbe1f2ba',
                },
                '1.5': {
                    'filename': 'dotav1.0_images_val.tar.gz',
                    'md5': '42293219ba61d61c417ae558bbe1f2ba',
                },
                '2.0': {
                    'filename': 'dotav2.0_images_val.tar.gz',
                    'md5': '737f65edf54b5aa627b3d48b0e253095',
                },
            },
            'annotations': {
                '1.0': {
                    'filename': 'dotav1.0_annotations_val.tar.gz',
                    'md5': '28155c05b1dc3a0f5cb6b9bdfef85a13',
                },
                '1.5': {
                    'filename': 'dotav1.5_annotations_val.tar.gz',
                    'md5': '85bf945788784cf9b4f1c714453178fc',
                },
                '2.0': {
                    'filename': 'dotav2.0_annotations_val.tar.gz',
                    'md5': 'ec53c1dbcfc125d7532bd6a065c647ac',
                },
            },
        },
    }

    sample_df_path = 'samples.csv'

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
    valid_versions = ('1.0', '1.5', '2.0')

    valid_orientations = ('horizontal', 'oriented')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        version: str = '2.0',
        bbox_orientation: str = 'oriented',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DOTA dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of the dataset to use, one of ['train', 'val']
            version: version of the dataset to use, one of ['1.0', '2.0']
            bbox_orientation: bounding box orientation, one of ['horizontal', 'oriented'], where horizontal
                returnx xyxy format and oriented returns x1y1x2y2x3y3x4y4 format
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

        self.sample_df = pd.read_csv(os.path.join(self.root, 'samples.csv'))
        self.sample_df['version'] = self.sample_df['version'].astype(str)
        self.sample_df = self.sample_df[self.sample_df['split'] == self.split]
        self.sample_df = self.sample_df[
            self.sample_df['version'] == self.version
        ].reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
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

        boxes, labels = self._load_annotations(sample_row['annotation_path'])

        if self.bbox_orientation == 'horizontal':
            sample['bbox_xyxy'] = boxes
        else:
            sample['bbox'] = boxes
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
                boxes: tensor of shape (N, 8) with coordinates for oriented
                    and (N, 4) for horizontal
                labels: tensor of shape (N,) with class indices
        """
        with open(os.path.join(self.root, path)) as f:
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

            if self.bbox_orientation == 'horizontal':
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
                torch.zeros((0, 4 if self.bbox_orientation == 'horizontal' else 8)),
                torch.zeros(0, dtype=torch.long),
            )
        else:
            return torch.tensor(boxes), torch.tensor(labels)

    def _verify(self) -> None:
        """Verify dataset integrity and download/extract if needed."""
        # check if directories and sample file are present
        required_dirs = [
            os.path.join(self.root, self.split, 'images'),
            os.path.join(
                self.root, self.split, 'annotations', f'version{self.version}'
            ),
            os.path.join(self.root, self.sample_df_path),
        ]
        if all(os.path.exists(d) for d in required_dirs):
            return

        # Check for compressed files, v1.0 and v1.5 have the same images but different annotations
        files_needed = [
            (
                self.file_info[self.split]['images'][self.version]['filename'],
                self.file_info[self.split]['images'][self.version]['md5'],
            ),
            (
                self.file_info[self.split]['annotations'][self.version]['filename'],
                self.file_info[self.split]['annotations'][self.version]['md5'],
            ),
        ]
        # For v2.0, also need v1.0 image files, but only v2 annotations
        if self.version == '2.0':
            files_needed.append(
                (
                    self.file_info[self.split]['images']['1.0']['filename'],
                    self.file_info[self.split]['images']['1.0']['md5'],
                )
            )

        # Check if archives exist and verify checksums
        exists = []
        for filename, md5 in files_needed:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                if self.checksum:
                    if not check_integrity(filepath, md5):
                        raise RuntimeError(f'Archive {filename} corrupted')
                exists.append(True)
                if not os.path.exists(os.path.join(self.root, filename)):
                    self._extract([(filename, md5)])
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # also download the metadata file
        self._download(files_needed)
        self._extract(files_needed)

    def _download(self, files_needed: list[tuple[str, str]]) -> None:
        """Download the dataset.

        Args:
            files_needed: list of files to download for the particular version
        """
        for filename, md5 in files_needed:
            if not os.path.exists(os.path.join(self.root, filename)):
                download_url(
                    url=self.url.format(filename),
                    root=self.root,
                    filename=filename,
                    md5=None if not self.checksum else md5,
                )

        if not os.path.exists(os.path.join(self.root, self.sample_df_path)):
            download_url(
                url=self.url.format(self.sample_df_path),
                root=self.root,
                filename=self.sample_df_path,
            )

    def _extract(self, files_needed: list[tuple[str, str]]) -> None:
        """Extract the dataset.

        Args:
            files_needed: list of files to extract for the particular version
        """
        for filename, _ in files_needed:
            filepath = os.path.join(self.root, filename)
            extract_archive(filepath, self.root)

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
        if self.bbox_orientation == 'horizontal':
            boxes = sample['bbox_xyxy'].cpu().numpy()
        else:
            boxes = sample['bbox'].cpu().numpy()
        labels = sample['labels'].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
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
