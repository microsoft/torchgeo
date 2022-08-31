# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FAIR1M dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from xml.etree.ElementTree import Element, parse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive


def parse_pascal_voc(path: str) -> Dict[str, Any]:
    """Read a PASCAL VOC annotation file.

    Args:
        path: path to xml file

    Returns:
        dict of image filename, points, and class labels
    """
    et = parse(path)
    element = et.getroot()
    source = cast(Element, element.find("source"))
    filename = cast(Element, source.find("filename")).text
    labels, points = [], []
    objects = cast(Element, element.find("objects"))
    for obj in objects.findall("object"):
        elm_points = cast(Element, obj.find("points"))
        lis_points = elm_points.findall("point")
        str_points = []
        for point in lis_points:
            text = cast(str, point.text)
            str_points.append(text.split(","))
        tup_points = [(float(p1), float(p2)) for p1, p2 in str_points]
        possibleresult = cast(Element, obj.find("possibleresult"))
        name = cast(Element, possibleresult.find("name"))
        label = name.text
        labels.append(label)
        points.append(tup_points)
    return dict(filename=filename, points=points, labels=labels)


class FAIR1M(NonGeoDataset):
    """FAIR1M dataset.

    The `FAIR1M <http://gaofen-challenge.com/benchmark>`__
    dataset is a dataset for remote sensing fine-grained oriented object detection.

    Dataset features:

    * 15,000+ images with 0.3-0.8 m per pixel resolution (1,000-10,000 px)
    * 1 million object instances
    * 5 object categories, 37 object sub-categories
    * three spectral bands - RGB
    * images taken by Gaofen satellites and Google Earth

    Dataset format:

    * images are three-channel tiffs
    * labels are xml files with PASCAL VOC like annotations

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

    .. versionadded:: 0.2
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
    filenames = ["images.zip", "labelXmls.zip"]
    md5s = ["a460fe6b1b5b276bf856ce9ac72d6568", "ca8666dc43a553f8d65e5dc671a8ac3c"]

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new FAIR1M dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self._verify()
        self.files = sorted(
            glob.glob(os.path.join(self.root, self.labels_root, "*.xml"))
        )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        path = self.files[index]
        parsed = parse_pascal_voc(path)
        image = self._load_image(parsed["filename"])
        boxes, labels = self._load_target(parsed["points"], parsed["labels"])
        sample = {"image": image, "boxes": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to image

        Returns:
            the image
        """
        path = os.path.join(self.root, self.image_root, path)
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(
        self, points: List[List[Tuple[float, float]]], labels: List[str]
    ) -> Tuple[Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            points: list of point tuple lists
            labels: list of class labels

        Returns:
            the target bounding boxes and labels
        """
        labels_list = [self.classes[label]["id"] for label in labels]
        boxes = torch.tensor(points).to(torch.float)
        labels_tensor = torch.tensor(labels_list)
        return boxes, labels_tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not found
        """
        # Check if the files already exist
        exists = []
        for directory in [self.image_root, self.labels_root]:
            exists.append(os.path.exists(os.path.join(self.root, directory)))
        if all(exists):
            return

        # Check if .zip files already exists (if so extract)
        exists = []
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        raise RuntimeError(
            "Dataset not found in `root` directory, "
            "specify a different `root` directory."
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"].permute((1, 2, 0)).numpy()

        ncols = 1
        if "prediction_boxes" in sample:
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols < 2:
            axs = [axs]

        axs[0].imshow(image)
        axs[0].axis("off")
        polygons = [
            patches.Polygon(points, color="r", fill=False)
            for points in sample["boxes"].numpy()
        ]
        for polygon in polygons:
            axs[0].add_patch(polygon)

        if show_titles:
            axs[0].set_title("Ground Truth")

        if ncols > 1:
            axs[1].imshow(image)
            axs[1].axis("off")
            polygons = [
                patches.Polygon(points, color="r", fill=False)
                for points in sample["prediction_boxes"].numpy()
            ]
            for polygon in polygons:
                axs[0].add_patch(polygon)

            if show_titles:
                axs[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
