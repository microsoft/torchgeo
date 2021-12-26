# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""US3D dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .geo import VisionDataset

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class US3D(VisionDataset):
    """US3D dataset.

    The `Urban Semantic 3D (US3D) <https://spacenet.ai/the-ussocom-urban-3d-competition/>`_
    dataset is a dataset for remote sensing fine-grained oriented object detection.

    Dataset features:

    * 15,000+ images with 0.5 m per pixel resolution (2,048-2,048 px)
    * 1 million object instances
    * three spectral bands - RGB
    * images taken by Maxar WorldView 2-3 satellites

    Dataset classes:

    0. Background
    1. Building

    Dataset format:

    * images are three-channel tiffs
    * digital surface and terrain models are one-channel tiffs
    * instance and binary masks are one-channel tiffs

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/AIPR.2017.8457973

    .. versionadded:: 0.2
    """

    directories = {
        "train": "01-Provisional_Train",
        "val": "02-Provisional_Test",
        "test": "03-Sequestered_Test",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new Urban3D dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        rgb = self._load_image(files["rgb"])
        dtm = self._load_image(files["dtm"])
        dsm = self._load_image(files["dsm"])
        image = torch.cat([rgb, dtm, dsm], dim=0)
        mask = self._load_target(files["instance_mask"])
        sample = {"image": image, "mask": mask}

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
        with rasterio.open(path) as f:
            array = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.float)  # type: ignore[attr-defined]
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load a single target.

        Args:
            path: path to target

        Returns:
            the target
        """
        with rasterio.open(path) as f:
            array = f.read(out_dtype="int32")
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
            return tensor

    def _load_files(self) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each sample
        """
        image_root = os.path.join(self.root, self.directories[self.split], "Inputs")
        target_root = os.path.join(self.root, self.directories[self.split], "GT")
        basenames = [
            os.path.basename(f) for f in glob.glob(os.path.join(image_root, "*.tif"))
        ]
        prefixes = set([os.path.splitext(f)[0].rsplit("_", 1)[0] for f in basenames])

        files = []
        for prefix in prefixes:
            files.append(
                dict(
                    rgb=os.path.join(image_root, f"{prefix}_RGB.tif"),
                    dtm=os.path.join(image_root, f"{prefix}_DTM.tif"),
                    dsm=os.path.join(image_root, f"{prefix}_DSM.tif"),
                    binary_mask=os.path.join(target_root, f"{prefix}_GTL.tif"),
                    instance_mask=os.path.join(target_root, f"{prefix}_GTI.tif"),
                )
            )
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if the dataset is not found
        """
        # Check if the files already exist
        exists = []
        for directory in self.directories.values():
            exists.append(os.path.exists(os.path.join(self.root, directory)))
        if all(exists):
            return

        raise RuntimeError(
            "Dataset not found in `root` directory, "
            "specify a different `root` directory."
        )
