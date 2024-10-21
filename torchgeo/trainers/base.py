# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` trainers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import lightning
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseTask(LightningModule, ABC):
    """Abstract base class for all TorchGeo trainers.

    .. versionadded:: 0.5
    """

    #: Parameters to ignore when saving hyperparameters.
    ignore: Sequence[str] | str | None = 'weights'

    #: Model to train.
    model: Any

    #: Performance metric to monitor in learning rate scheduler and callbacks.
    monitor = 'val_loss'

    #: Whether the goal is to minimize or maximize the performance metric to monitor.
    mode = 'min'

    def __init__(self) -> None:
        """Initialize a new BaseTask instance.

        Args:
            ignore: Arguments to skip when saving hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=self.ignore)
        self.configure_models()
        self.configure_losses()
        self.configure_metrics()

    @abstractmethod
    def configure_models(self) -> None:
        """Initialize the model."""

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""

    def configure_optimizers(
        self,
    ) -> 'lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig':
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams['lr'])
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams['patience'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor},
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)
