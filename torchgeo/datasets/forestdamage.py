# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Forest Damage dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive, extract_archive


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

        label_var = obj.find("damage")
        if label_var is not None:
            label = label_var.text
        else:
            label = "other"
        bboxes.append(bbox)
        labels.append(label)
    return dict(filename=filename, bboxes=bboxes, labels=labels)


class ForestDamage(NonGeoDataset):
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
      <https://roboflow.com/formats/pascal-voc-xml#w-tabs-0-data-w-pane-3>`_

    Dataset Classes:

    0. other
    1. healthy
    2. light damage
    3. high damage

    If the download fails or stalls, it is recommended to try azcopy
    as suggested `here <https://lila.science/faq>`__. It is expected that the
    downloaded data file with name ``Data_Set_Larch_Casebearer``
    can be found in ``root``.

    If you use this dataset in your research, please use the following citation:

    * Swedish Forest Agency (2021): Forest Damages - Larch Casebearer 1.0.
      National Forest Data Lab. Dataset.

    .. versionadded:: 0.3
    """

    classes = ["other", "H", "LD", "HD"]
    url = (
        "https://lilablobssc.blob.core.windows.net/larch-casebearer/"
        "Data_Set_Larch_Casebearer.zip"
    )
    data_dir = "Data_Set_Larch_Casebearer"
    md5 = "907815bcc739bff89496fac8f8ce63d7"

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

        self._verify()

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
            list of dicts containing paths for each pair of image, annotation
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
            tensor: Tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(
        self, bboxes: List[List[int]], labels_list: List[str]
    ) -> Tuple[Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            bboxes: list of bbox coordinats [xmin, ymin, xmax, ymax]
            labels_list: list of class labels

        Returns:
            the target bounding boxes and labels
        """
        labels = torch.tensor([self.class_to_idx[label] for label in labels_list])
        boxes = torch.tensor(bboxes).to(torch.float)
        return boxes, labels

    def _verify(self) -> None:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories are found, else False
        """
        filepath = os.path.join(self.root, self.data_dir)
        if os.path.isdir(filepath):
            return

        filepath = os.path.join(self.root, self.data_dir + ".zip")
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory, either specify a different"
                + " `root` directory or manually download "
                + "the dataset to this directory."
            )

        # else download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum does not match
        """
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.data_dir + ".zip",
            md5=self.md5 if self.checksum else None,
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
