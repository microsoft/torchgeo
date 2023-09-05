# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base class for all :mod:`torchgeo` trainers."""

from typing import Any

from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseTask(LightningModule):
    """Base class for all TorchGeo trainers.

    .. versionadded:: 0.5
    """

    #: Model to train
    model: Any

    #: Performance metric to monitor in learning rate scheduler and callbacks
    monitor = "val_loss"

    def __init__(self) -> None:
        """Initialize a new BaseTask instance."""
        super().__init__()
        self.save_hyperparameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams["patience"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
