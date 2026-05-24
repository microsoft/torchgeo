# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for temporal regression."""

from typing import Any, Literal

import einops
from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import Sample
from ..models import LTAE
from .base import BaseTask


class TemporalRegressionTask(BaseTask):
    """Trainer for sequence-to-sequence temporal regression.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        model: Literal['ltae'] = 'ltae',
        in_features: int = 1,
        out_features: int = 1,
        loss: Literal['mae', 'mse'] = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new TemporalRegressionTask instance.

        Args:
            model: Name of the model architecture.
            in_features: Number of input features per time step
                (the *C* dimension of the *(B, T, C)* input tensor).
            out_features: Number of output features times the number of time steps
                (the *T x C* dimension of the *(B, T, C)* target tensor).
            loss: Loss function.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            **kwargs: Additional keyword arguments passed to the model constructor.

        .. versionadded:: 0.10
        """
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        match self.hparams['model']:
            case 'ltae':
                ltae = LTAE(in_channels=self.hparams['in_features'], **self.kwargs)
                linear = nn.Linear(ltae.n_neurons[-1], self.hparams['out_features'])
                self.model = nn.Sequential(ltae, linear)

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        match self.hparams['loss']:
            case 'mse':
                self.criterion: nn.Module = nn.MSELoss()
            case 'mae':
                self.criterion = nn.L1Loss()

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.MeanSquaredError` (``MSE``) and its square root
          (``RMSE``). Lower is better.
        * :class:`~torchmetrics.MeanAbsoluteError` (``MAE``). Lower is better.
        """
        metrics = MetricCollection(
            {
                'RMSE': MeanSquaredError(squared=False),
                'MSE': MeanSquaredError(squared=True),
                'MAE': MeanAbsoluteError(),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _unnormalise(
        self, y_hat: Tensor, y: Tensor, batch: Sample, H: int
    ) -> tuple[Tensor, Tensor]:
        """Optionally map predictions and targets back to the original scale.

        When the datamodule supplies per-feature ``'mean'`` and ``'std'``
        tensors in the batch (as :class:`~torchgeo.datamodules.AirQualityDataModule`
        does), both ``y_hat`` and ``y`` are de-normalised before metric
        computation so that reported values are interpretable in physical
        units.

        If neither key is present this method is a **no-op**, returning
        ``y_hat`` and ``y`` unchanged.
        Override this method to implement a different normalisation
        convention (e.g. min-max scaling, per-sample stats).

        Args:
            y_hat: Model predictions ``(B, num_outputs)`` in normalised space.
            y: Flattened ground-truth targets ``(B, H*C)`` in normalised space.
            batch: The full batch dict; may contain ``'mean'`` and ``'std'``.
            H: Number of future time steps being predicted.

        Returns:
            ``(y_hat_orig, y_orig)`` — tensors in the original scale, or the
            inputs unchanged when no normalisation stats are available.
        """
        mean: Tensor | None = batch.get('mean')  # type: ignore[assignment]
        std: Tensor | None = batch.get('std')  # type: ignore[assignment]

        if mean is None or std is None:
            return y_hat, y

        mean_rep = mean.repeat(H).to(y_hat)  # (H*C,)
        std_rep = std.repeat(H).to(y_hat)  # (H*C,)

        y_hat_orig = y_hat * std_rep + mean_rep
        y_orig = y * std_rep + mean_rep
        return y_hat_orig, y_orig

    def _shared_step(self, batch: Sample, batch_idx: int, stage: str) -> Tensor:
        """Forward pass, loss computation, and metric update for all splits.

        Args:
            batch: Output of the DataLoader.  Must contain at least
                :attr:`input_key` and :attr:`target_key`.  Optionally
                contains ``'mean'`` / ``'std'`` for unnormalised metrics.
            batch_idx: Index of this batch within the epoch.
            stage: One of ``'train'``, ``'val'``, or ``'test'``.

        Returns:
            Scalar loss tensor.
        """
        x = batch['input']
        y = batch['target']
        y = einops.rearrange(y, 'b t c -> b (t c)')
        batch_size = x.shape[0]

        y_hat = self.model(x)  # (B, out_features)

        loss = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)

        # Denormalize before computing metrics
        datamodule = self.trainer.datamodule
        y = y * datamodule.target_mean + datamodule.target_std
        y_hat = y_hat * datamodule.target_mean + datamodule.target_std

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)
        self.log_dict(metrics, batch_size=batch_size)

        return loss

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, batch_idx, 'test')

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute predicted values, optionally unnormalised.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Predicted values of shape ``(B, num_outputs)``.  When the batch
            contains ``'mean'`` / ``'std'`` the predictions are returned in
            the *original* (unnormalised) scale via :meth:`_unnormalise`.
        """
        x = batch['input']

        y_hat = self.model(x)

        # Denormalize before returning predictions
        datamodule = self.trainer.datamodule
        y_hat = y_hat * datamodule.target_mean + datamodule.target_std

        return y_hat
