# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced datamodule."""

from typing import Any, Optional

from kornia.augmentation import Normalize, Resize

from ..datasets import UCMerced
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class UCMercedDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the UC Merced dataset.

    Uses random train/val/test splits.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for UCMerced based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.UCMerced`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            Normalize(mean=0.0, std=255.0), Resize(size=256), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            UCMerced(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = UCMerced(split="train", **self.kwargs)
        self.val_dataset = UCMerced(split="val", **self.kwargs)
        self.test_dataset = UCMerced(split="test", **self.kwargs)
