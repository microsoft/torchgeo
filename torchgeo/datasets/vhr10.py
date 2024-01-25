# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NWPU VHR-10 dataset."""

import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    DatasetNotFoundError,
    check_integrity,
    download_and_extract_archive,
    download_url,
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
    """
    from pycocotools import mask as coco_mask  # noqa: F401

    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
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
        image = sample["image"]
        _, h, w = image.size()
        target = sample["label"]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        bboxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        categories = [obj["category_id"] for obj in anno]
        classes = torch.tensor(categories, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]

        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {"boxes": boxes, "labels": classes, "image_id": image_id}
        if masks.nelement() > 0:
            masks = masks[keep]
            target["masks"] = masks

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return {"image": image, "label": target}


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

    .. note::

       This dataset requires the following additional libraries to be installed:

       * `pycocotools <https://pypi.org/project/pycocotools/>`_ to load the
         ``annotations.json`` file for the "positive" image set
       * `rarfile <https://pypi.org/project/rarfile/>`_ to extract the dataset,
         which is stored in a RAR file
    """

    image_meta = {
        "url": "https://drive.google.com/file/d/1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE",
        "filename": "NWPU VHR-10 dataset.rar",
        "md5": "d30a7ff99d92123ebb0b3a14d9102081",
    }
    target_meta = {
        "url": "https://raw.githubusercontent.com/chaozhong2010/VHR-10_dataset_coco/ce0ba0f5f6a0737031f1cbe05e785ddd5ef05bd7/NWPU%20VHR-10_dataset_coco/annotations.json",  # noqa: E501
        "filename": "annotations.json",
        "md5": "7c76ec50c17a61bb0514050d20f22c08",
    }

    categories = [
        "background",
        "airplane",
        "ships",
        "storage tank",
        "baseball diamond",
        "tennis court",
        "basketball court",
        "ground track field",
        "harbor",
        "bridge",
        "vehicle",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "positive",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new VHR-10 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "postive" or "negative"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            ImportError: if ``split="positive"`` and pycocotools is not installed
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in ["positive", "negative"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        if split == "positive":
            # Must be installed to parse annotations file
            try:
                from pycocotools.coco import COCO  # noqa: F401
            except ImportError:
                raise ImportError(
                    "pycocotools is not installed and is required to use this dataset"
                )

            self.coco = COCO(
                os.path.join(
                    self.root, "NWPU VHR-10 dataset", self.target_meta["filename"]
                )
            )

            self.coco_convert = ConvertCocoAnnotations()
            self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = index % len(self) + 1

        sample: dict[str, Any] = {
            "image": self._load_image(id_),
            "label": self._load_target(id_),
        }

        if sample["label"]["annotations"]:
            sample = self.coco_convert(sample)
            sample["labels"] = sample["label"]["labels"]
            sample["boxes"] = sample["label"]["boxes"]
            sample["masks"] = sample["label"]["masks"]
            del sample["label"]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        if self.split == "positive":
            return len(self.ids)
        else:
            return 150

    def _load_image(self, id_: int) -> Tensor:
        """Load a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = os.path.join(
            self.root,
            "NWPU VHR-10 dataset",
            self.split + " image set",
            f"{id_:03d}.jpg",
        )
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            tensor = torch.from_numpy(array)
            tensor = tensor.float()
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, id_: int) -> dict[str, Any]:
        """Load the annotations for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the annotations
        """
        # Images in the "negative" image set have no annotations
        annot = []
        if self.split == "positive":
            annot = self.coco.loadAnns(self.coco.getAnnIds(id_ - 1))

        target = dict(image_id=id_, annotations=annot)

        return target

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        image: bool = check_integrity(
            os.path.join(self.root, self.image_meta["filename"]),
            self.image_meta["md5"] if self.checksum else None,
        )

        # Annotations only needed for "positive" image set
        target = True
        if self.split == "positive":
            target = check_integrity(
                os.path.join(
                    self.root, "NWPU VHR-10 dataset", self.target_meta["filename"]
                ),
                self.target_meta["md5"] if self.checksum else None,
            )

        return image and target

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        # Download images
        download_and_extract_archive(
            self.image_meta["url"],
            self.root,
            filename=self.image_meta["filename"],
            md5=self.image_meta["md5"] if self.checksum else None,
        )

        # Annotations only needed for "positive" image set
        if self.split == "positive":
            # Download annotations
            download_url(
                self.target_meta["url"],
                os.path.join(self.root, "NWPU VHR-10 dataset"),
                self.target_meta["filename"],
                self.target_meta["md5"] if self.checksum else None,
            )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        show_feats: Optional[str] = "both",
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
            ImportError: if plotting masks and scikit-image is not installed

        .. versionadded:: 0.4
        """
        assert show_feats in {"boxes", "masks", "both"}

        if self.split == "negative":
            fig, axs = plt.subplots(squeeze=False)
            axs[0, 0].imshow(sample["image"].permute(1, 2, 0))
            axs[0, 0].axis("off")

            if suptitle is not None:
                plt.suptitle(suptitle)
            return fig

        if show_feats != "boxes":
            try:
                from skimage.measure import find_contours  # noqa: F401
            except ImportError:
                raise ImportError(
                    "scikit-image is not installed and is required to plot masks."
                )

        image = sample["image"].permute(1, 2, 0).numpy()
        boxes = sample["boxes"].cpu().numpy()
        labels = sample["labels"].cpu().numpy()
        if "masks" in sample:
            masks = [mask.squeeze().cpu().numpy() for mask in sample["masks"]]

        n_gt = len(boxes)

        ncols = 1
        show_predictions = "prediction_labels" in sample

        if show_predictions:
            show_pred_boxes = False
            show_pred_masks = False
            prediction_labels = sample["prediction_labels"].numpy()
            prediction_scores = sample["prediction_scores"].numpy()
            if "prediction_boxes" in sample:
                prediction_boxes = sample["prediction_boxes"].numpy()
                show_pred_boxes = True
            if "prediction_masks" in sample:
                prediction_masks = sample["prediction_masks"].numpy()
                show_pred_masks = True

            n_pred = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 13))
        axs[0, 0].imshow(image)
        axs[0, 0].axis("off")

        cm = plt.get_cmap("gist_rainbow")
        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / len(self.categories))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            if show_feats in {"boxes", "both"}:
                r = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle="dashed",
                    edgecolor=color,
                    facecolor="none",
                )
                axs[0, 0].add_patch(r)

            # Add labels
            label = self.categories[class_num]
            caption = label
            axs[0, 0].text(
                x1, y1 - 8, caption, color="white", size=11, backgroundcolor="none"
            )

            # Add masks
            if show_feats in {"masks", "both"} and "masks" in sample:
                mask = masks[i]
                contours = find_contours(mask, 0.5)  # type: ignore[no-untyped-call]
                for verts in contours:
                    verts = np.fliplr(verts)
                    p = patches.Polygon(
                        verts, facecolor=color, alpha=mask_alpha, edgecolor="white"
                    )
                    axs[0, 0].add_patch(p)

            if show_titles:
                axs[0, 0].set_title("Ground Truth")

        if show_predictions:
            axs[0, 1].imshow(image)
            axs[0, 1].axis("off")
            for i in range(n_pred):
                score = prediction_scores[i]
                if score < 0.5:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(self.categories))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                    axs[0, 1].add_patch(r)

                    # Add labels
                    label = self.categories[class_num]
                    caption = f"{label} {score:.3f}"
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color="white",
                        size=11,
                        backgroundcolor="none",
                    )

                # Add masks
                if show_pred_masks:
                    mask = prediction_masks[i]
                    contours = find_contours(mask, 0.5)  # type: ignore[no-untyped-call]
                    for verts in contours:
                        verts = np.fliplr(verts)
                        p = patches.Polygon(
                            verts, facecolor=color, alpha=mask_alpha, edgecolor="white"
                        )
                        axs[0, 1].add_patch(p)

            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
