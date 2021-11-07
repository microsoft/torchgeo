# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom trainer for the RESISC45 dataset."""

from typing import Any, Dict, cast

import kornia.augmentation as K
import torch
from torch import Tensor

from .classification import ClassificationTask


# TODO: move this functionality into ClassificationTask and remove this class
class RESISC45ClassificationTask(ClassificationTask):
    """LightningModule for training on RESISC45 with data augmentation.

    .. deprecated:: 0.1
       Use :class:`ClassificationTask` instead.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            classification_model: Name of the classification model use
            loss: Name of the loss function
            weights: Either "random", "imagenet_only", "imagenet_and_random", or
                "random_rgb"
        """
        super().__init__(**kwargs)

        self.train_augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["input"],
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
        y = batch["label"]
        with torch.no_grad():
            x = self.train_augmentations(x)
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)
