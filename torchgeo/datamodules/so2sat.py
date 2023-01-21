# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat datamodule."""

from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from ..datasets import So2Sat


class So2SatDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(
        [
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
    )

    band_stds = torch.tensor(
        [
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
    )

    # this reorders the bands to put S2 RGB first, then remainder of S2
    reindex_to_rgb_first = [2, 1, 0, 3, 4, 5, 6, 7, 8, 9]

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        band_set: str = "rgb",
        unsupervised_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for So2Sat based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            band_set: Collection of So2Sat bands to use
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.So2Sat`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.band_set = band_set
        self.unsupervised_mode = unsupervised_mode
        self.kwargs = kwargs

        self.norm = Normalize(self.band_means, self.band_stds)

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] = self.norm(sample["image"])
        sample["image"] = sample["image"][self.reindex_to_rgb_first, :, :]

        if self.band_set == "rgb":
            sample["image"] = sample["image"][:3, :, :]

        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        bands = So2Sat.BAND_SETS["s2"]
        train_transforms = Compose([self.preprocess])
        val_test_transforms = self.preprocess

        if not self.unsupervised_mode:
            self.train_dataset = So2Sat(
                split="train", bands=bands, transforms=train_transforms, **self.kwargs
            )

            self.val_dataset = So2Sat(
                split="validation",
                bands=bands,
                transforms=val_test_transforms,
                **self.kwargs,
            )

            self.test_dataset = So2Sat(
                split="test", bands=bands, transforms=val_test_transforms, **self.kwargs
            )

        else:

            temp_train = So2Sat(
                split="train", bands=bands, transforms=train_transforms, **self.kwargs
            )

            self.val_dataset = So2Sat(
                split="validation",
                bands=bands,
                transforms=train_transforms,
                **self.kwargs,
            )

            self.test_dataset = So2Sat(
                split="test", bands=bands, transforms=train_transforms, **self.kwargs
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

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.So2Sat.plot`.

        .. versionadded:: 0.4
        """
        return self.test_dataset.plot(*args, **kwargs)
