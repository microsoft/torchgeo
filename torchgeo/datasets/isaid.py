# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""iSAID dataset."""

import os
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    lazy_import,
)


def convert_coco_poly_to_mask(
    segmentations: list[int], height: int, width: int
) -> Tensor:
    """Convert coco polygons to mask tensor.

    Args:
        segmentations (List[int]): polygon coordinates
        height (int): image height
        width (int): image width

    Returns:
        Tensor: Mask tensor

    Raises:
        DependencyNotFoundError: If pycocotools is not installed.
    """
    pycocotools = lazy_import('pycocotools')
    masks = []
    for polygons in segmentations:
        rles = pycocotools.mask.frPyObjects(polygons, height, width)
        mask = pycocotools.mask.decode(rles)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
        masks_tensor = torch.stack(masks, dim=0)

    return masks_tensor


class ConvertCocoAnnotations:
    """Callable for converting the boxes, masks and labels into tensors.

    This is a modified version of ConvertCocoPolysToMask() from torchvision found in
    https://github.com/pytorch/vision/blob/v0.14.0/references/detection/coco_utils.py
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Converts MS COCO fields (boxes, masks & labels) from list of ints to tensors.

        Args:
            sample: Sample

        Returns:
            Processed sample
        """
        image = sample['image']
        _, h, w = image.size()
        target = sample['label']

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        bboxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        categories = [obj['category_id'] for obj in anno]
        classes = torch.tensor(categories, dtype=torch.int64)

        segmentations = [obj['segmentation'] for obj in anno]

        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {'boxes': boxes, 'labels': classes, 'image_id': image_id}
        if masks.nelement() > 0:
            masks = masks[keep]
            target['masks'] = masks

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])
        iscrowd = torch.tensor([obj['iscrowd'] for obj in anno])
        target['area'] = area
        target['iscrowd'] = iscrowd
        return {'image': image, 'label': target}


class ISAID(NonGeoDataset):
    """iSAID dataset.

    The `iSAID <https://captain-whu.github.io/iSAID/>`_ dataset is a large-scale instance segmentation dataset for aerial imagery.
    It builds upon the DOTA V1 dataset, but includes instance-level annotations for 15 object categories.

    Dataset features:

    * multi-class instance segmentation
    * multi-class object detection
    * aerial imagery over various GSDs

    Dataset format:

    * images are three channel RGB PNGs with various pixel dimensions
    * labels are annotaitons in json MSCOCO format

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

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1905.12886
    * https://arxiv.org/abs/1711.10398

    .. versionadded:: 0.7

    .. note::

       This dataset requires the following additional library to be installed:

       * `pycocotools <https://pypi.org/project/pycocotools/>`_ to load the
         annotations
    """

    img_url = 'https://huggingface.co/datasets/torchgeo/dota/tree/main/{}'

    img_files: ClassVar[dict[str, dict[str, str]]] = {
        'train': {'filename': 'dotav1_images_train.tar.gz', 'md5': ''},
        'val': {'filename': 'dotav1_images_val.tar.gz', 'md5': ''},
    }

    label_url = 'https://huggingface.co/datasets/torchgeo/isaid/tree/main/{}'

    label_files: ClassVar[dict[str, dict[str, str]]] = {
        'train': {'filename': 'isaid_annotations_train.tar.gz', 'md5': ''},
        'val': {'filename': 'isaid_annotations_val.tar.gz', 'md5': ''},
    }

    # Retrieved from self.coco.loadCats(self.coco.getCatIds())
    classes: ClassVar[dict[int, str]] = {
        1: 'storage_tank',
        2: 'Large_Vehicle',
        3: 'Small_Vehicle',
        4: 'plane',
        5: 'ship',
        6: 'Swimming_pool',
        7: 'Harbor',
        8: 'tennis_court',
        9: 'Ground_Track_Field',
        10: 'Soccer_ball_field',
        11: 'baseball_diamond',
        12: 'Bridge',
        13: 'basketball_court',
        14: 'Roundabout',
        15: 'Helicopter',
    }

    valid_splits = ('train', 'val')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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
            DependencyNotFoundError: if pycocotools is
                not installed.
        """
        assert split in self.valid_splits, (
            f"Invalid split '{split}', please use one of {self.valid_splits}"
        )

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        pc = lazy_import('pycocotools.coco')
        self.coco = pc.COCO(
            os.path.join(
                self.root, self.split, 'Annotations', f'iSAID_{self.split}.json'
            )
        )
        self.coco_convert = ConvertCocoAnnotations()
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = index % len(self) + 1

        sample: dict[str, Any] = {
            'image': self._load_image(id_),
            'label': self._load_mask(id_),
        }

        sample = self.coco_convert(sample)
        sample['labels'] = sample['label']['labels']
        sample['boxes'] = sample['label']['boxes']
        sample['masks'] = sample['label']['masks']
        del sample['label']

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_mask(self, id_: int) -> dict[str, Any]:
        """Load mask.

        Args:
            id_: image ID for coco

        Returns:
            instance mask tensor with unique IDs
        """
        annot = self.coco.loadAnns(self.coco.getAnnIds(id_ - 1))

        target = dict(image_id=id_, annotations=annot)
        return target

    def _load_image(self, id_: int) -> Tensor:
        """Load an image from a given path.

        Args:
            id_: image ID for coco

        Returns:
            image tensor
        """
        filename = os.path.join(
            self.root, self.split, 'images', self.coco.imgs[id_ - 1]['file_name']
        )
        image = Image.open(filename).convert('RGB')
        return torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # check presence of directories
        dirs = ['images', 'Annotations', 'Instance_masks', 'Semantic_masks']
        exists = [
            os.path.exists(os.path.join(self.root, self.split, dir)) for dir in dirs
        ]

        if all(exists):
            return

        # check compressed files
        exists = []
        files = [
            self.img_files[self.split]['filename'],
            self.label_files[self.split]['filename'],
        ]
        md5s = [self.img_files[self.split]['md5'], self.label_files[self.split]['md5']]
        for file, md5 in zip(files, md5s):
            if os.path.exists(os.path.join(self.root, file)):
                if self.checksum and not check_integrity(
                    os.path.join(self.root, file), md5
                ):
                    raise RuntimeError(f'Archive {file} is found but corrupted')
                exists.append(True)
                extract_archive(os.path.join(self.root, file), self.root)
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # download the dataset
        for file in files:
            download_and_extract_archive(
                self.img_url.format(file), self.root, md5=md5 if self.checksum else None
            )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """

        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']
        masks = sample.get('masks', None)

        # Convert image tensor (C, H, W) to a numpy array (H, W, C).
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            # If the values are normalized [0,1], convert to [0,255]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        cmap = plt.get_cmap('tab10')

        # Plot bounding boxes and class labels.
        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            color = cmap(i % 10)
            rect = Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            label_id = int(labels[i].cpu().item())
            class_name = self.classes.get(label_id, str(label_id))
            ax.text(
                x1, y1, class_name, color=color, fontsize=12, backgroundcolor='white'
            )

        # Overlay instance segmentation masks if available.
        if masks is not None:
            masks = masks.cpu().numpy()  # shape: (N, H, W)
            for i in range(masks.shape[0]):
                mask = masks[i]
                color = cmap(i % 10)
                # Create an RGBA image for the mask.
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[..., :3] = color[:3]
                # Use the binary mask to set an alpha value.
                colored_mask[..., 3] = 0.4 * mask
                ax.imshow(colored_mask)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
