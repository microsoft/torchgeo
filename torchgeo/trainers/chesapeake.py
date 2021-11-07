# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Segmentation tasks."""

from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]

from ..datasets import Chesapeake7
from .segmentation import SemanticSegmentationTask

# TODO: move the color maps to a dataset object
CMAP_7 = matplotlib.colors.ListedColormap(
    [np.array(Chesapeake7.cmap[i]) / 255.0 for i in range(7)]
)
CMAP_5 = matplotlib.colors.ListedColormap(
    np.array(
        [
            (0, 0, 0, 0),
            (0, 197, 255, 255),
            (38, 115, 0, 255),
            (163, 255, 115, 255),
            (156, 156, 156, 255),
        ]
    )
    / 255.0
)


# TODO: move this functionality into SemanticSegmentationTask and remove this class
class ChesapeakeCVPRSegmentationTask(SemanticSegmentationTask):
    """LightningModule for training models on the Chesapeake CVPR Land Cover dataset.

    .. deprecated: 0.1
       Use :class:`SemanticSegmentationTask` instead.
    """

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
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            cmap = None
            if self.hparams["num_classes"] == 5:
                cmap = CMAP_5
            else:
                cmap = CMAP_7
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0].cpu().numpy(), 0, 3
            )
            mask = batch["mask"][0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img[:, :, :3])
            axs[0].axis("off")
            axs[1].imshow(
                mask,
                vmin=0,
                vmax=self.hparams["num_classes"] - 1,
                cmap=cmap,
                interpolation="none",
            )
            axs[1].axis("off")
            axs[2].imshow(
                pred,
                vmin=0,
                vmax=self.hparams["num_classes"] - 1,
                cmap=cmap,
                interpolation="none",
            )
            axs[2].axis("off")
            plt.tight_layout()

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()
