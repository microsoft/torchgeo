# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat dataset."""

import os
from typing import Any, Callable, Dict, Optional, cast

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from .geo import VisionDataset
from .utils import check_integrity

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


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
    * Testing: eastern half of the 10 other cities

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/MGRS.2020.2964708

    .. note::

       This dataset can be automatically downloaded using the following bash script:

       .. code-block:: bash

          for split in training validation testing
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


class So2SatDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [
            -3.591224256609313e-05,
            -7.658561276843396e-06,
            5.9373857475971184e-05,
            2.5166231537121083e-05,
            0.04420110659759328,
            0.25761027084996196,
            0.0007556743372573258,
            0.0013503466830024448,
            0.12375696117681859,
            0.1092774636368323,
            0.1010855203267882,
            0.1142398616114001,
            0.1592656692023089,
            0.18147236008771792,
            0.1745740312291377,
            0.19501607349635292,
            0.15428468872076637,
            0.10905050699570007,
        ]
    ).reshape(18, 1, 1)

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [
            0.17555201137417686,
            0.17556463274968204,
            0.45998793417834255,
            0.455988755730148,
            2.8559909213125763,
            8.324800606439833,
            2.4498757382563103,
            1.4647352984509094,
            0.03958795985905458,
            0.047778262752410296,
            0.06636616706371974,
            0.06358874912497474,
            0.07744387147984592,
            0.09101635085921553,
            0.09218466562387101,
            0.10164581233948201,
            0.09991773043519253,
            0.08780632509122865,
        ]
    ).reshape(18, 1, 1)

    # this reorders the bands to put S2 RGB first, then remainder of S2, then S1
    reindex_to_rgb_first = [
        10,
        9,
        8,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        # 0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
    ]

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        bands: str = "rgb",
        unsupervised_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for So2Sat based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the So2Sat Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            bands: Either "rgb" or "s2"
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bands = bands
        self.unsupervised_mode = unsupervised_mode

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image

        Returns:
            preprocessed sample
        """
        # sample["image"] = (sample["image"] - self.band_means) / self.band_stds
        sample["image"] = sample["image"].float()
        sample["image"] = sample["image"][self.reindex_to_rgb_first, :, :]

        if self.bands == "rgb":
            sample["image"] = sample["image"][:3, :, :]

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        So2Sat(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_transforms = Compose([self.preprocess])
        val_test_transforms = self.preprocess

        if not self.unsupervised_mode:

            self.train_dataset = So2Sat(
                self.root_dir, split="train", transforms=train_transforms
            )

            self.val_dataset = So2Sat(
                self.root_dir, split="validation", transforms=val_test_transforms
            )

            self.test_dataset = So2Sat(
                self.root_dir, split="test", transforms=val_test_transforms
            )

        else:

            temp_train = So2Sat(
                self.root_dir, split="train", transforms=train_transforms
            )

            self.val_dataset = So2Sat(
                self.root_dir, split="validation", transforms=train_transforms
            )

            self.test_dataset = So2Sat(
                self.root_dir, split="test", transforms=train_transforms
            )

            self.train_dataset = cast(
                So2Sat, temp_train + self.val_dataset + self.test_dataset
            )

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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
