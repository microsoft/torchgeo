# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom trainer for the ETCI2021 dataset."""

from typing import Any, Dict, cast

import kornia.augmentation as K
import torch
from torch import Tensor

from .segmentation import SemanticSegmentationTask


# TODO: move this functionality into SemanticSegmentationTask and remove this class
class ETCI2021SemanticSegmentationTask(SemanticSegmentationTask):
    """LightningModule for training on ETCI2021 with data augmentation.

    .. deprecated:: 0.2
       Use :class:`SemanticSegmentationTask` instead.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function."""
        super().__init__(**kwargs)

        self.train_augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            data_keys=["input", "mask"],
        )

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        with torch.no_grad():
            # Kornia expects masks to be [B, C, H, W] format with float values while
            # torch losses like cross entropy expect masks to be in [B, H, W] format
            # with long values. We'll just assume that samples are in a format that
            # torch expects and adjust accordingly if we need to do augmentations.
            x, y = self.train_augmentations(x, y.unsqueeze(1).float())
        y = y.long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)
