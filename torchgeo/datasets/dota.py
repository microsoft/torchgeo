# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""DOTA dataset."""

import os
from collections.abc import Callable
from typing import ClassVar, Literal

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
    Sample,
    check_integrity,
    download_url,
    extract_archive,
    quantile_normalization,
)


class DOTA(NonGeoDataset):
    """DOTA dataset.

    `DOTA <https://captain-whu.github.io/DOTA/index.html>`__ is a large-scale object
    detection dataset for aerial imagery containing RGB and gray-scale imagery
    from Google Earth, GF-2 and JL-1 satellites as well as additional aerial imagery
    from CycloMedia. There are three versions of the dataset: v1.0, v1.5, and v2.0, where,
    v1.0 and v1.5 have the same images but different annotations,
    and v2.0 extends both the images and annotations with more samples

    Dataset features:

    * 1869 samples in v1.0 and v1.5 and 2423 samples in v2.0
    * multi-class object detection (15 classes in v1.0 and v1.5 and 18 classes in v2.0)
    * horizontal and oriented bounding boxes

    Dataset format:

    * images are three channel PNGs with various pixel sizes
    * annotations are text files with one line per bounding box

    Classes:

    0. plane
    1. ship
    2. storage-tank
    3. baseball-diamond
    4. tennis-court
    5. basketball-court
    6. ground-track-field
    7. harbor
    8. bridge
    9. large-vehicle
    10. small-vehicle
    11. helicopter
    12. roundabout
    13. soccer-ball-field
    14. swimming-pool
    15. container-crane (v1.5+)
    16. airport (v2.0+)
    17. helipad (v2.0+)

    If you use this work in your research, please cite the following papers:

    * https://arxiv.org/abs/2102.12219
    * https://arxiv.org/abs/1711.10398

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/isaaccorley/dota/resolve/672e63236622f7da6ee37fca44c50ac368b77cab/{}'

    file_info: ClassVar[dict[str, dict[str, dict[str, dict[str, str]]]]] = {
        'train': {
            'images': {
                '1.0': {
                    'filename': 'dotav1.0_images_train.tar.gz',
                    'sha256': '2fd69eeb9ba0c775db007c7fe11d7708fb905a67ca3a663874147434db68e7ac',
                },
                '1.5': {
                    'filename': 'dotav1.0_images_train.tar.gz',
                    'sha256': '2fd69eeb9ba0c775db007c7fe11d7708fb905a67ca3a663874147434db68e7ac',
                },
                '2.0': {
                    'filename': 'dotav2.0_images_train.tar.gz',
                    'sha256': 'ceccafe50e4e49c5f1ad6b0bce917e356c4df29321fd79037c77912bad594989',
                },
            },
            'annotations': {
                '1.0': {
                    'filename': 'dotav1.0_annotations_train.tar.gz',
                    'sha256': '79096d14f5065e1582af47817b5d7c2d1c1611cadc7f1ccc3f868ded1c41a9f6',
                },
                '1.5': {
                    'filename': 'dotav1.5_annotations_train.tar.gz',
                    'sha256': '0c8fe411f50331dcb0d1aeef90606045e24d6f34b225630f8b98af3044c469fc',
                },
                '2.0': {
                    'filename': 'dotav2.0_annotations_train.tar.gz',
                    'sha256': 'f142824f6eafef3b1922f7ff1583375e4d59d2a841d4670c312be854652631a6',
                },
            },
        },
        'val': {
            'images': {
                '1.0': {
                    'filename': 'dotav1.0_images_val.tar.gz',
                    'sha256': 'f7ab9c570b3aba66d07d6ec91801f76c840fde2ffc12b4f8efa02c90aeb32c83',
                },
                '1.5': {
                    'filename': 'dotav1.0_images_val.tar.gz',
                    'sha256': 'f7ab9c570b3aba66d07d6ec91801f76c840fde2ffc12b4f8efa02c90aeb32c83',
                },
                '2.0': {
                    'filename': 'dotav2.0_images_val.tar.gz',
                    'sha256': '76005d65bc1d0e8ee2dd4b4c2922b47fddc73d07dd93a87eb5f022d22727a0fd',
                },
            },
            'annotations': {
                '1.0': {
                    'filename': 'dotav1.0_annotations_val.tar.gz',
                    'sha256': '879e8f013a5231cb52c3dc95b5ade7d4aeb3a0d3e600cb658a5a4d13ea7116c2',
                },
                '1.5': {
                    'filename': 'dotav1.5_annotations_val.tar.gz',
                    'sha256': 'd9a5c8412f0094fcba088c8b41e1f214e5dbc22da415beba32289d6b9f32e2a5',
                },
                '2.0': {
                    'filename': 'dotav2.0_annotations_val.tar.gz',
                    'sha256': '66cde2e76f131a55243904884b68ef450a73b1404668001df92bc94dde4a2e25',
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
        split: Literal['train', 'val'] = 'train',
        version: Literal['1.0', '1.5', '2.0'] = '2.0',
        bbox_orientation: Literal['horizontal', 'oriented'] = 'oriented',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DOTA dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of the dataset to use, one of ['train', 'val']
            version: version of the dataset to use, one of ['1.0', '1.5', '2.0']
            bbox_orientation: bounding box orientation, one of ['horizontal', 'oriented'], where horizontal
                returnx xyxy format and oriented returns x1y1x2y2x3y3x4y4 format
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files (may be slow)

        Raises:
            AssertionError: if *split*, *version*, or *bbox_orientation* argument are not valid
            DatasetNotFoundError: if dataset is not found or corrupted, and *download* is False
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

    def __getitem__(self, index: int) -> Sample:
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
                self.file_info[self.split]['images'][self.version]['sha256'],
            ),
            (
                self.file_info[self.split]['annotations'][self.version]['filename'],
                self.file_info[self.split]['annotations'][self.version]['sha256'],
            ),
        ]
        # For v2.0, also need v1.0 image files, but only v2 annotations
        if self.version == '2.0':
            files_needed.append(
                (
                    self.file_info[self.split]['images']['1.0']['filename'],
                    self.file_info[self.split]['images']['1.0']['sha256'],
                )
            )

        # Check if archives exist and verify checksums if requested
        exists = []
        for filename, sha256 in files_needed:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                if self.checksum and not check_integrity(filepath, sha256=sha256):
                    raise RuntimeError(f'Archive {filename} corrupted')
                exists.append(True)
                self._extract([(filename, sha256)])
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
        for filename, sha256 in files_needed:
            if not os.path.exists(os.path.join(self.root, filename)):
                download_url(
                    url=self.url.format(filename),
                    root=self.root,
                    filename=filename,
                    sha256=None if not self.checksum else sha256,
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
        image = quantile_normalization(sample['image'].permute(1, 2, 0))
        if self.bbox_orientation == 'horizontal':
            boxes = sample['bbox_xyxy']
        else:
            boxes = sample['bbox']
        labels = sample['labels']

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
                    bbox={'facecolor': color, 'alpha': box_alpha},
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
                    bbox={'facecolor': color, 'alpha': box_alpha},
                    ha='center',
                    va='center',
                )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
