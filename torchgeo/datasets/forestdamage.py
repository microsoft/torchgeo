# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Forest Damage dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from xml.etree import ElementTree

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import VisionDataset


def parse_pascal_voc(path: str) -> Dict[str, Any]:
    """Read a PASCAL VOC annotation file.

    Args:
        path: path to xml file

    Returns:
        dict of image filename, points, and class labels
    """
    et = ElementTree.parse(path)
    element = et.getroot()
    filename = element.find("filename").text  # type: ignore[union-attr]
    labels, bboxes = [], []
    for obj in element.findall("object"):
        bndbox = obj.find("bndbox")
        bbox = [
            int(bndbox.find("xmin").text),  # type: ignore[union-attr, arg-type]
            int(bndbox.find("ymin").text),  # type: ignore[union-attr, arg-type]
            int(bndbox.find("xmax").text),  # type: ignore[union-attr, arg-type]
            int(bndbox.find("ymax").text),  # type: ignore[union-attr, arg-type]
        ]
        label = obj.find("damage").text  # type: ignore[union-attr]
        bboxes.append(bbox)
        labels.append(label)
    return dict(filename=filename, bboxes=bboxes, labels=labels)


class ForestDamage(VisionDataset):
    """Forest Damage dataset.

    The `ForestDamage
    <https://lila.science/datasets/forest-damages-larch-casebearer/>`_
    dataset contains drone imagery that can be used for tree identification,
    as well as tree damage classification for larch trees.

    Dataset features:

    * 1543 images
    * 101,878 tree annotations
    * subset of 840 images contain 44,522 annotations about tree health
      (Healthy (H), Light Damage (LD), High Damage (HD)), all other
      images have "other" as damage level

    Dataset format:

    * images are three-channel jpgs
    * annotations are in `Pascal VOC XML format
      <https://roboflow.com/formats/pascal-voc-xml#w-tabs-0-data-w-pane-3>`_n

    If you use this dataset in your research, please use the following citation:

    * Swedish Forest Agency (2021): Forest Damages – Larch Casebearer 1.0.
      National Forest Data Lab. Dataset.

    .. versionadded:: 0.3
    """

    classes = ["H", "LD", "HD", "other"]

    data_dir = "Data_Set_Larch_Casebearer"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new ForestDamage dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self.files = self._load_files(self.root)

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        parsed = parse_pascal_voc(files["annotation"])
        image = self._load_image(files["image"])

        boxes, labels = self._load_target(parsed["bboxes"], parsed["labels"])

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

    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image, audio, label
        """
        images = sorted(
            glob.glob(os.path.join(root, self.data_dir, "**", "Images", "*.JPG"))
        )
        annotations = sorted(
            glob.glob(os.path.join(root, self.data_dir, "**", "Annotations", "*.xml"))
        )

        files = [
            dict(image=image, annotation=annotation)
            for image, annotation in zip(images, annotations)
        ]

        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(
        self, bboxes: List[List[int]], labels: List[str]
    ) -> Tuple[Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            bboxes: list of bbox coordinats [xmin, ymin, xmax, ymax]
            labels: list of class labels

        Returns:
            the target bounding boxes and labels
        """
        labels_list = [self.class_to_idx[label] for label in labels]
        boxes = torch.tensor(bboxes).to(torch.float)  # type: ignore[attr-defined]
        labels = torch.tensor(labels_list)  # type: ignore[attr-defined]
        return boxes, cast(Tensor, labels)

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
        showing_predictions = "prediction_boxes" in sample
        if showing_predictions:
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if not showing_predictions:
            axs = [axs]

        axs[0].imshow(image)
        axs[0].axis("off")

        bboxes = [
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            for bbox in sample["boxes"].numpy()
        ]
        for bbox in bboxes:
            axs[0].add_patch(bbox)

        if show_titles:
            axs[0].set_title("Ground Truth")

        if showing_predictions:
            axs[1].imshow(image)
            axs[1].axis("off")

            pred_bboxes = [
                patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                for bbox in sample["prediction_boxes"].numpy()
            ]
            for bbox in pred_bboxes:
                axs[1].add_patch(bbox)

            if show_titles:
                axs[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
