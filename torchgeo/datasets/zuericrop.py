# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ZueriCrop dataset."""

import os
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url


class ZueriCrop(VisionDataset):
    """ZueriCrop dataset.

    The `ZueriCrop <https://github.com/0zgur0/ms-convSTAR>`_
    dataset is a dataset for time-series instance segmentation of crops.

    Dataset features:

    * Sentinel-2 multispectral imagery
    * instance masks of 48 crop categories
    * nine multispectral bands
    * 116k images with 10 m per pixel resolution (24x24 px)
    * ~28k time-series containing 142 images each

    Dataset format:

    * single hdf5 dataset containing images, semantic masks, and instance masks
    * data is parsed into images and instance masks, boxes, and labels
    * one mask per time-series

    Dataset classes:

    * 48 fine-grained hierarchical crop
      `categories <https://github.com/0zgur0/ms-convSTAR/blob/master/labels.csv>`_

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2021.112603

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset
    """

    urls = [
        "https://polybox.ethz.ch/index.php/s/uXfdr2AcXE3QNB6/download",
        "https://raw.githubusercontent.com/0zgur0/ms-convSTAR/master/labels.csv",
    ]
    md5s = ["1635231df67f3d25f4f1e62c98e221a4", "5118398c7a5bbc246f5f6bb35d8d529b"]
    filenames = ["ZueriCrop.hdf5", "labels.csv"]

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new ZueriCrop dataset instance.

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
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, "ZueriCrop.hdf5")

        self._verify()

        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError(
                "h5py is not installed and is required to use this dataset"
            )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample containing image, mask, bounding boxes, and target label
        """
        image = self._load_image(index)
        mask, boxes, label = self._load_target(index)

        sample = {"image": image, "mask": mask, "boxes": boxes, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        import h5py

        with h5py.File(self.filepath, "r") as f:
            length: int = f["data"].shape[0]
        return length

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        import h5py

        with h5py.File(self.filepath, "r") as f:
            array = f["data"][index, ...]

        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        # Convert from TxHxWxC to TxCxHxW
        tensor = tensor.permute((0, 3, 1, 2))
        return tensor

    def _load_target(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask and label for each mask
        """
        import h5py

        with h5py.File(self.filepath, "r") as f:
            mask_array = f["gt"][index, ...]
            instance_array = f["gt_instance"][index, ...]

        mask_tensor = torch.from_numpy(mask_array)  # type: ignore[attr-defined]
        instance_tensor = torch.from_numpy(instance_array)  # type: ignore[attr-defined]

        # Convert from HxWxC to CxHxW
        mask_tensor = mask_tensor.permute((2, 0, 1))
        instance_tensor = instance_tensor.permute((2, 0, 1))

        # Convert instance mask of N instances to N binary instance masks
        instance_ids = torch.unique(instance_tensor)  # type: ignore[attr-defined]
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]
        instance_ids = instance_ids[:, None, None]
        masks: Tensor = instance_tensor == instance_ids

        # Parse labels for each instance
        labels_list = []
        for mask in masks:
            label = mask_tensor[mask[None, :, :]]
            label = torch.unique(label)[0]  # type: ignore[attr-defined]
            labels_list.append(label)

        # Get bounding boxes for each instance
        boxes_list = []
        for mask in masks:
            pos = torch.where(mask)  # type: ignore[attr-defined]
            xmin = torch.min(pos[1])  # type: ignore[attr-defined]
            xmax = torch.max(pos[1])  # type: ignore[attr-defined]
            ymin = torch.min(pos[0])  # type: ignore[attr-defined]
            ymax = torch.max(pos[0])  # type: ignore[attr-defined]
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.to(torch.uint8)  # type: ignore[attr-defined]
        boxes = torch.tensor(boxes_list).to(torch.float)  # type: ignore[attr-defined]
        labels = torch.tensor(labels_list).to(torch.long)  # type: ignore[attr-defined]

        return masks, boxes, labels

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            exists.append(os.path.exists(filepath))

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for url, filename, md5 in zip(self.urls, self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                download_url(
                    url,
                    self.root,
                    filename=filename,
                    md5=md5 if self.checksum else None,
                )
