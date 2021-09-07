# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat dataset."""

import os
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from .geo import VisionDataset
from .utils import check_integrity


class So2Sat(VisionDataset):
    """So2Sat dataset.

    The `So2Sat <https://doi.org/10.1109/MGRS.2020.2964708>`_ dataset consists of
    corresponding synthetic aperture radar and multispectral optical image data
    acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and a
    corresponding local climate zones (LCZ) label. The dataset is distributed over
    42 cities across different continents and cultural regions of the world, and comes
    with a split into fully independent, non-overlapping training, validation,
    and test sets.

    This implementation focuses on the *2nd* version of the dataset as described in
    the author's github repository https://github.com/zhu-xlab/So2Sat-LCZ42 and hosted
    at https://mediatum.ub.tum.de/1483140. This version is identical to the first
    version of the dataset but includes the test data. The splits are defined as
    follows:

    * Training: 42 cities around the world
    * Validation: western half of 10 other cities covering 10 cultural zones
    * Testing: eastern half of the 10 other citie

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/MGRS.2020.2964708

    .. note::

       This dataset can be automatically downloaded using the following bash script:

       .. code-block: bash

          for split in train validation testing
          do
              wget ftp://m1483140:m1483140@dataserv.ub.tum.de/$split.h5
          done

       or manually downloaded from https://dataserv.ub.tum.de/index.php/s/m1483140
       This download will likely take several hours.
    """

    filenames = {
        "train": "training.h5",
        "validation": "validation.h5",
        "test": "testing.h5",
    }
    md5s = {
        "train": "702bc6a9368ebff4542d791e53469244",
        "validation": "71cfa6795de3e22207229d06d6f8775d",
        "test": "e81426102b488623a723beab52b31a8a",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new So2Sat dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "validation", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if data is not found in ``root``, or checksums don't match
        """
        import h5py

        assert split in ["train", "validation", "test"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        self.fn = os.path.join(self.root, self.filenames[split])

        with h5py.File(self.fn, "r") as f:
            self.size = int(f["label"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        import h5py

        with h5py.File(self.fn, "r") as f:
            s1 = f["sen1"][index].astype(np.float64)  # convert from <f8 to float64
            s2 = f["sen2"][index].astype(np.float64)  # convert from <f8 to float64
            label = int(  # convert one-hot encoding to int64 then Python int
                f["label"][index].argmax()
            )

            s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
            s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format

            s1 = torch.from_numpy(s1)  # type: ignore[attr-defined]
            s2 = torch.from_numpy(s2)  # type: ignore[attr-defined]

        sample = {
            "image": torch.cat([s1, s2]),  # type: ignore[attr-defined]
            "label": label,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return self.size

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for split_name, filename in self.filenames.items():
            filepath = os.path.join(self.root, filename)
            md5 = self.md5s[split_name]
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True
