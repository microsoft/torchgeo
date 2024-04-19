# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for I/O benchmarking."""

from typing import Any

import lightning
import torch
from torch import Tensor
from torch.optim import SGD

from .base import BaseTask


class IOBenchTask(BaseTask):
    """I/O benchmarking.

    .. versionadded:: 0.6
    """

    def configure_models(self) -> None:
        """No-op."""

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer.

        Returns:
            Optimizer.
        """
        optimizer = SGD([torch.tensor(0.0, requires_grad=True)], lr=0)
        return {"optimizer": optimizer}

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """No-op.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Zero.
        """
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
