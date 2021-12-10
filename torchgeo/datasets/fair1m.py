# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FAIR1M dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional, Tuple
from xml.etree import ElementTree

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets.utils import dataset_split
from .geo import VisionDataset

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


def read_pascal_voc(path: str) -> Dict[str, Any]:
    """Read a PASCAL VOC annotation file.

    Args:
        path: path to xml file

    Returns:
        dict of image filename, points, and class labels
    """
    et = ElementTree.parse(path)
    element = et.getroot()
    filename = element.find("source").find("filename").text  # type: ignore[union-attr]
    labels, points = [], []
    for obj in element.find("objects").findall("object"):  # type: ignore[union-attr]
        obj_points = [
            p for p in obj.find("points").findall("point")  # type: ignore[union-attr]
        ]
        obj_points = [p.text.split(",") for p in obj_points]  # type: ignore[union-attr]
        obj_points = [(float(p1), float(p2)) for p1, p2 in obj_points]  # type: ignore[arg-type]
        label = obj.find("possibleresult").find("name").text  # type: ignore[union-attr]
        labels.append(label)
        points.append(obj_points)
    return dict(filename=filename, points=points, labels=labels)


class FAIR1M(VisionDataset):
    """FAIR1M dataset.

    The `FAIR1M <http://gaofen-challenge.com/benchmark>`_
    dataset is a dataset for remote sensing fine-grained oriented object detection.

    Dataset features:

    * 15,000+ images with 0.3-0.8m m per pixel resolution (1,000-10,000 px)
    * 1 million object instances
    * 5 object categories, 37 object sub-categories
    * three spectral bands - RGB
    * images taken by Gaofen satellites and Google Earth

    Dataset format:

    * images are three-channel tiffs
    * labels are xml files with PASCAL VOC annotations

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

    * https://arxiv.org/abs/2103.05569

    """

    classes = {
        "Passenger Ship": {"id": 0, "category": "Ship"},
        "Motorboat": {"id": 1, "category": "Ship"},
        "Fishing Boat": {"id": 2, "category": "Ship"},
        "Tugboat": {"id": 3, "category": "Ship"},
        "other-ship": {"id": 4, "category": "Ship"},
        "Engineering Ship": {"id": 5, "category": "Ship"},
        "Liquid Cargo Ship": {"id": 6, "category": "Ship"},
        "Dry Cargo Ship": {"id": 7, "category": "Ship"},
        "Warship": {"id": 8, "category": "Ship"},
        "Small Car": {"id": 9, "category": "Vehicle"},
        "Bus": {"id": 10, "category": "Vehicle"},
        "Cargo Truck": {"id": 11, "category": "Vehicle"},
        "Dump Truck": {"id": 12, "category": "Vehicle"},
        "other-vehicle": {"id": 13, "category": "Vehicle"},
        "Van": {"id": 14, "category": "Vehicle"},
        "Trailer": {"id": 15, "category": "Vehicle"},
        "Tractor": {"id": 16, "category": "Vehicle"},
        "Excavator": {"id": 17, "category": "Vehicle"},
        "Truck Tractor": {"id": 18, "category": "Vehicle"},
        "Boeing737": {"id": 19, "category": "Airplane"},
        "Boeing747": {"id": 20, "category": "Airplane"},
        "Boeing777": {"id": 21, "category": "Airplane"},
        "Boeing787": {"id": 22, "category": "Airplane"},
        "ARJ21": {"id": 23, "category": "Airplane"},
        "C919": {"id": 24, "category": "Airplane"},
        "A220": {"id": 25, "category": "Airplane"},
        "A321": {"id": 26, "category": "Airplane"},
        "A330": {"id": 27, "category": "Airplane"},
        "A350": {"id": 28, "category": "Airplane"},
        "other-airplane": {"id": 29, "category": "Airplane"},
        "Baseball Field": {"id": 30, "category": "Court"},
        "Basketball Court": {"id": 31, "category": "Court"},
        "Football Field": {"id": 32, "category": "Court"},
        "Tennis Court": {"id": 33, "category": "Court"},
        "Roundabout": {"id": 34, "category": "Road"},
        "Intersection": {"id": 35, "category": "Road"},
        "Bridge": {"id": 36, "category": "Road"},
    }

    image_root: str = "images"
    labels_root: str = "labelXmls"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new FAIR1M dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.transforms = transforms

        files = sorted(glob.glob(os.path.join(self.root, self.labels_root, "*.xml")))
        self.files = [read_pascal_voc(f) for f in files]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        boxes, labels = self._load_target(index)
        sample = {"image": image, "bbox": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        filename = self.files[index]["filename"]
        path = os.path.join(self.root, self.image_root, filename)
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target bounding boxes and labels
        """
        boxes_list = self.files[index]["points"]
        labels_list = [
            self.classes[label]["id"] for label in self.files[index]["labels"]
        ]
        boxes = torch.tensor(boxes_list).to(torch.float)  # type: ignore[attr-defined]
        labels = torch.tensor(labels_list)  # type: ignore[attr-defined]
        return boxes, labels

    def plot(self, index: int) -> None:
        """Plot a data sample.

        Args:
            index: the index of the sample to plot
        """
        sample = self[index]
        image = sample["image"].permute((1, 2, 0)).numpy()
        polygons = [
            patches.Polygon(points, color="r", fill=False)
            for points in sample["bbox"].numpy()
        ]
        ax = plt.axes()
        ax.imshow(image)
        ax.axis("off")
        ax.figure.set_size_inches(10, 10)
        ax.figure.tight_layout()
        for polygon in polygons:
            ax.add_patch(polygon)
        plt.show()
        plt.close()


class FAIR1MDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the FAIR1M dataset."""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        unsupervised_mode: bool = False,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for FAIR1M based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the FAIR1M Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unsupervised_mode = unsupervised_mode
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        if not self.unsupervised_mode:
            dataset = FAIR1M(self.root_dir, transforms=transforms)
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
            )
        else:
            self.train_dataset = FAIR1M(  # type: ignore[assignment]
                self.root_dir, transforms=transforms
            )
            self.val_dataset, self.test_dataset = None, None  # type: ignore[assignment]

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        if self.unsupervised_mode or self.val_split_pct == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        if self.unsupervised_mode or self.test_split_pct == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
