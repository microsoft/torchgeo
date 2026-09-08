# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Functional Map of the World dataset."""

import glob
import json
import os
from collections.abc import Callable
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class FMoW(NonGeoDataset):
    """Functional Map of the World (fMoW) dataset.

    The `Functional Map of the World <https://arxiv.org/abs/1711.07846>`_ (fMoW)
    dataset contains satellite imagery for satellite building and land use classification.
    This loader supports the RGB image distribution of fMoW for train and val splits.

    Dataset features:

    * 62 land use and building categories
    * ~200 GB RGB image distribution
    * Bounding box spatial annotations in [x1, y1, x2, y2] format

    Dataset format:

    * Images are RGB .jpg files in nested category/sequence directories.
    * Paired metadata files are in UTF-8 JSON format containing bounding box coordinates.

    Dataset layout:

    .. code-block:: text

        <root>/
        ├── train/
        │   └── <category>/
        │       └── <sequence>/
        │           ├── <image>_rgb.jpg
        │           └── <image>_rgb.json
        └── val/
            └── <category>/
                └── <sequence>/
                    ├── <image>_rgb.jpg
                    └── <image>_rgb.json

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1711.07846

    .. versionadded:: 0.11
    """

    valid_splits = ('train', 'val')

    classes: ClassVar[tuple[str, ...]] = (
        'airport',
        'airport_hangar',
        'airport_terminal',
        'amusement_park',
        'aquaculture',
        'archaeological_site',
        'barn',
        'border_checkpoint',
        'burial_site',
        'car_dealership',
        'construction_site',
        'crop_field',
        'dam',
        'debris_or_rubble',
        'educational_institution',
        'electric_substation',
        'factory_or_powerplant',
        'fire_station',
        'flooded_road',
        'fountain',
        'gas_station',
        'golf_course',
        'ground_transportation_station',
        'helipad',
        'hospital',
        'impoverished_settlement',
        'interchange',
        'lake_or_pond',
        'lighthouse',
        'military_facility',
        'multi-unit_residential',
        'nuclear_powerplant',
        'office_building',
        'oil_or_gas_facility',
        'park',
        'parking_lot_or_garage',
        'place_of_worship',
        'police_station',
        'port',
        'prison',
        'race_track',
        'railway_bridge',
        'recreational_facility',
        'road_bridge',
        'runway',
        'shipyard',
        'shopping_mall',
        'single-unit_residential',
        'smokestack',
        'solar_farm',
        'space_facility',
        'stadium',
        'storage_tank',
        'surface_mine',
        'swimming_pool',
        'toll_booth',
        'tower',
        'tunnel_opening',
        'waste_disposal',
        'water_treatment_facility',
        'wind_farm',
        'zoo',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        """Initialize a new FMoW dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of dataset to use, one of 'train' or 'val'
            transforms: optional function/transform taking input sample and returning transformed sample

        Raises:
            AssertionError: If *split* argument is invalid.
            DatasetNotFoundError: If dataset directory is missing or contains no matching images.
        """
        assert split in self.valid_splits, f'Split must be one of {self.valid_splits}.'

        self.root = root
        self.split = split
        self.transforms = transforms
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        pattern = os.path.join(self.root, self.split, '*', '*', '*_rgb.jpg')
        image_paths = glob.glob(pattern)
        self.image_paths = sorted(
            path
            for path in image_paths
            if os.path.basename(os.path.dirname(os.path.dirname(path)))
            in self.class_to_idx
        )

        if not self.image_paths:
            raise DatasetNotFoundError(self)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Sample:
        """Return a sample at the given index.

        Args:
            index: index of sample to fetch

        Returns:
            sample containing image, label index, and bounding boxes
        """
        image_path = self.image_paths[index]

        category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        label = torch.tensor(self.class_to_idx[category])

        image = self._load_image(image_path)
        bbox_xyxy = self._load_bounding_boxes(image_path)

        sample: Sample = {'image': image, 'label': label, 'bbox_xyxy': bbox_xyxy}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: Path) -> Tensor:
        """Load an image as a C x H x W float tensor.

        Args:
            path: path to image file

        Returns:
            image tensor
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.uint8] = np.array(img.convert('RGB'))
            tensor: Tensor = torch.from_numpy(array).permute(2, 0, 1).float()
            return tensor

    def _load_bounding_boxes(self, image_path: Path) -> Tensor:
        """Load and convert bounding box annotations for an image.

        Args:
            image_path: path to image file

        Returns:
            tensor of bounding box corner coordinates shaped (N, 4)
        """
        json_path = os.path.splitext(image_path)[0] + '.json'
        boxes: list[list[float]] = []

        with open(json_path, encoding='utf-8') as f:
            metadata = json.load(f)

        for annotation in metadata['bounding_boxes']:
            x, y, w, h = annotation['box']
            boxes.append([float(x), float(y), float(x + w), float(y + h)])

        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by __getitem__
            show_titles: flag indicating whether to draw titles
            suptitle: optional figure title

        Returns:
            Matplotlib Figure containing plotted sample
        """
        image = sample['image'].permute(1, 2, 0)
        if image.is_floating_point() and image.max() > 1:
            image = image / 255
        image = image.numpy()

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')

        for box in sample['bbox_xyxy']:
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                )
            )

        if show_titles:
            title = ''
            if 'label' in sample:
                label_idx = int(sample['label'].item())
                title += f'Label: {self.classes[label_idx]}'
            if 'prediction' in sample:
                pred_idx = int(sample['prediction'].item())
                if title:
                    title += '\n'
                title += f'Prediction: {self.classes[pred_idx]}'
            if title:
                ax.set_title(title)

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
