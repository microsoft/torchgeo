# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""NWPU VHR-10 dataset."""

import json
import os
from collections import defaultdict
from collections.abc import Callable
from typing import ClassVar, Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image
from rasterio.features import rasterize
from shapely import Polygon

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    check_integrity,
    download_and_extract_archive,
    download_url,
    quantile_normalization,
)


class VHR10(NonGeoDataset):
    """NWPU VHR-10 dataset.

    Northwestern Polytechnical University (NWPU) very-high-resolution ten-class (VHR-10)
    remote sensing image dataset.

    Consists of 800 VHR optical remote sensing images, where 715 color images were
    acquired from Google Earth with the spatial resolution ranging from 0.5 to 2 m,
    and 85 pansharpened color infrared (CIR) images were acquired from Vaihingen data
    with a spatial resolution of 0.08 m.

    The data set is divided into two sets:

    * Positive image set (650 images) which contains at least one target in an image
    * Negative image set (150 images) does not contain any targets

    The positive image set consists of objects from ten classes:

    1. Airplanes (757)
    2. Ships (302)
    3. Storage tanks (655)
    4. Baseball diamonds (390)
    5. Tennis courts (524)
    6. Basketball courts (159)
    7. Ground track fields (163)
    8. Harbors (224)
    9. Bridges (124)
    10. Vehicles (477)

    Includes object detection bounding boxes from original paper and instance
    segmentation masks from follow-up publications. If you use this dataset in your
    research, please cite the following papers:

    * https://doi.org/10.1016/j.isprsjprs.2014.10.002
    * https://doi.org/10.1109/IGARSS.2019.8898573
    * https://doi.org/10.3390/rs12060989
    """

    image_meta: ClassVar[dict[str, str]] = {
        'url': 'https://hf.co/datasets/isaaccorley/vhr10/resolve/60ecc4be33609184e2224606858cd00b7daba8df/NWPU%20VHR-10%20dataset.zip',
        'filename': 'NWPU VHR-10 dataset.zip',
        'md5': '6add6751469c12dd8c8d6223064c6c4d',
    }
    target_meta: ClassVar[dict[str, str]] = {
        'url': 'https://hf.co/datasets/isaaccorley/vhr10/resolve/7e7968ad265dadc4494e0ca4a079e0b63dc6f3f8/annotations.json',
        'filename': 'annotations.json',
        'md5': '7c76ec50c17a61bb0514050d20f22c08',
    }

    categories = (
        'background',
        'airplane',
        'ship',
        'storage tank',
        'baseball diamond',
        'tennis court',
        'basketball court',
        'ground track field',
        'harbor',
        'bridge',
        'vehicle',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['positive', 'negative'] = 'positive',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new VHR-10 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "positive" or "negative"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in {'positive', 'negative'}

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        if split == 'positive':
            path = os.path.join(self.root, 'NWPU VHR-10 dataset', 'annotations.json')
            with open(path) as f:
                annotations = json.load(f)

                # Gather image shapes
                out_shapes = []
                for image in annotations['images']:
                    out_shapes.append((image['height'], image['width']))

                self.labels = defaultdict(list)
                self.boxes = defaultdict(list)
                self.masks = defaultdict(list)
                for annotation in annotations['annotations']:
                    i = annotation['image_id']
                    self.labels[i].append(annotation['category_id'])

                    # Convert box format
                    x1, y1, w, h = annotation['bbox']
                    self.boxes[i].append([x1, y1, x1 + w, y1 + h])

                    # Rasterize segmentation mask
                    segmentation = annotation['segmentation']  # [[x1, y1, x2, y2, ...]]
                    xs = segmentation[0][::2]  # [x1, x2, ...]
                    ys = segmentation[0][1::2]  # [y1, y2, ...]
                    coords = list(zip(xs, ys))  # [(x1, y1), (x2, y2), ...]
                    shapes = [(Polygon(coords), 1)]
                    mask = rasterize(shapes, out_shapes[i], dtype=np.uint8)
                    self.masks[i].append(torch.from_numpy(mask))

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample = {}

        # Both 'positive' and 'negative' splits have an image
        split = f'{self.split} image set'
        file = f'{index + 1:03d}.jpg'
        path = os.path.join(self.root, 'NWPU VHR-10 dataset', split, file)
        with Image.open(path) as f:
            tensor = torch.from_numpy(np.array(f)).float()
            tensor = einops.rearrange(tensor, 'h w c -> c h w')
            sample['image'] = tensor

        # Only 'positive' split has target labels
        if self.split == 'positive':
            sample['label'] = torch.tensor(self.labels[index])
            sample['bbox_xyxy'] = torch.tensor(self.boxes[index])
            sample['mask'] = torch.stack(self.masks[index])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        if self.split == 'positive':
            return 650
        else:
            return 150

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        image: bool = check_integrity(
            os.path.join(self.root, self.image_meta['filename']),
            self.image_meta['md5'] if self.checksum else None,
        )

        # Annotations only needed for "positive" image set
        target = True
        if self.split == 'positive':
            target = check_integrity(
                os.path.join(
                    self.root, 'NWPU VHR-10 dataset', self.target_meta['filename']
                ),
                self.target_meta['md5'] if self.checksum else None,
            )

        return image and target

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # Download images
        download_and_extract_archive(
            self.image_meta['url'],
            self.root,
            filename=self.image_meta['filename'],
            md5=self.image_meta['md5'] if self.checksum else None,
        )

        # Annotations only needed for "positive" image set
        if self.split == 'positive':
            # Download annotations
            download_url(
                self.target_meta['url'],
                os.path.join(self.root, 'NWPU VHR-10 dataset'),
                self.target_meta['filename'],
                self.target_meta['md5'] if self.checksum else None,
            )

    def plot(
        self,
        sample: Sample,
        show_titles: bool = True,
        suptitle: str | None = None,
        show_feats: Literal['boxes', 'masks', 'both'] = 'both',
        box_alpha: float = 0.7,
        mask_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            show_feats: optional string to pick features to be shown: boxes, masks, both
            box_alpha: alpha value of box
            mask_alpha: alpha value of mask

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid

        .. versionadded:: 0.4
        """
        assert show_feats in {'boxes', 'masks', 'both'}

        cm = plt.get_cmap('gist_rainbow')
        ncols = 2 if 'prediction_label' in sample else 1
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 10))

        # Image
        image = einops.rearrange(sample['image'], 'c h w -> h w c')
        image = quantile_normalization(image)
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')

        if 'label' in sample:
            labels = sample['label']
            boxes = sample['bbox_xyxy']
            masks = sample['mask']
            for i in range(len(labels)):
                class_num = labels[i]
                color = cm(class_num / len(self.categories))

                # Boxes
                if show_feats in {'boxes', 'both'}:
                    # Add rectangle
                    x1, y1, x2, y2 = boxes[i]
                    r = Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle='dashed',
                        edgecolor=color,
                        facecolor='none',
                    )
                    axs[0, 0].add_patch(r)

                    # Add label
                    label = self.categories[class_num]
                    caption = label
                    axs[0, 0].text(
                        x1,
                        y1 - 8,
                        caption,
                        color='white',
                        size=11,
                        backgroundcolor='none',
                    )

                # Masks
                if show_feats in {'masks', 'both'}:
                    mask = masks[i]
                    alpha = mask * mask_alpha
                    mask = mask * class_num
                    axs[0, 0].imshow(mask, cmap=cm, vmin=0, vmax=10, alpha=alpha)

            if show_titles:
                axs[0, 0].set_title('Ground Truth')

        if 'prediction_label' in sample:
            # Image
            axs[0, 1].imshow(image)
            axs[0, 1].axis('off')

            scores = sample['prediction_score']
            labels = sample['prediction_label']
            boxes = sample['prediction_bbox_xyxy']
            for i in range(len(labels)):
                score = scores[i]
                if score < 0.5:
                    continue

                class_num = labels[i]
                color = cm(class_num / len(self.categories))

                # Boxes
                if show_feats in {'boxes', 'both'}:
                    # Add rectangle
                    x1, y1, x2, y2 = boxes[i]
                    r = Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle='dashed',
                        edgecolor=color,
                        facecolor='none',
                    )
                    axs[0, 1].add_patch(r)

                    # Add label
                    label = self.categories[class_num]
                    caption = f'{label} {score:.3f}'
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color='white',
                        size=11,
                        backgroundcolor='none',
                    )

                # Masks
                if 'prediction_mask' in sample and show_feats in {'masks', 'both'}:
                    masks = sample['prediction_mask']
                    mask = masks[i]
                    alpha = mask * mask_alpha
                    mask = mask * class_num
                    axs[0, 1].imshow(mask, cmap=cm, vmin=0, vmax=10, alpha=alpha)

            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
