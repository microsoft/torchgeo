# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Segmentation tasks."""

from typing import Any, Dict, cast

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]

from .segmentation import SemanticSegmentationTask


# TODO: move this functionality into SemanticSegmentationTask and remove this class
class LandCoverAISegmentationTask(SemanticSegmentationTask):
    """LightningModule for training models on the Landcover.AI Dataset.

    .. deprecated: 0.1
       Use :class:`SemanticSegmentationTask` instead.
    """

    # TODO: move this to LandCoverAIDataModule
    train_augmentations = K.AugmentationSequential(
        K.RandomRotation(p=0.5, degrees=90),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomSharpness(p=0.5),
        K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
            x, y = self.train_augmentations(x, y)
        y = y.long().squeeze()

        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU.

        Logs the first 10 validation samples to tensorboard as images with 3 subplots
        showing the image, mask, and predictions.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10 and self.hparams["verbose"]:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                x[0].cpu().numpy(), 0, 3
            )
            mask = y[0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=5)
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=5)
            axs[2].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)
