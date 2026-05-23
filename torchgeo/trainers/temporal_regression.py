# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for temporal regression."""

from typing import Any

import torch.nn as nn
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import Sample
from ..models import LTAE
from .base import BaseTask


class TemporalRegressionTask(BaseTask):
    """Trainer for sequence-to-value temporal regression.

    Accepts a fixed-length window of past observations ``(B, T, C)`` and
    predicts one or more future values ``(B, num_outputs)``.  See
    :meth:`__init__` for constructor arguments, :meth:`_prepare_targets`
    for batch key conventions, and :meth:`configure_metrics` for logged
    metrics.

    .. versionadded:: 0.10
    """

    #: Batch key for the model input tensor ``(B, T, C)``.
    input_key: str = 'input'

    #: Batch key for the regression target tensor ``(B, H, C)``.
    target_key: str = 'target'

    def __init__(
        self,
        model: str = 'ltae',
        in_channels: int = 12,
        num_outputs: int = 1,
        loss: str = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        **model_kwargs: Any,
    ) -> None:
        """Initialize a new TemporalRegressionTask instance.

        Args:
            model: Name of the model architecture.  Currently ``'ltae'`` is
                supported out of the box; add further architectures by
                extending :meth:`configure_models`.
            in_channels: Number of input features per time step (the ``C``
                dimension of the ``(B, T, C)`` input tensor).
            num_outputs: Total number of scalar values to predict per sample.
                For a multi-step, multi-channel forecast this should be
                ``num_future_steps x num_channels``.
            loss: Loss function.  One of ``'mse'`` or ``'mae'``.
            lr: Learning rate for the AdamW optimiser.
            patience: Number of epochs without improvement after which
                the learning rate is reduced (ReduceLROnPlateau).
            **model_kwargs: Additional keyword arguments forwarded verbatim
                to the model constructor inside :meth:`configure_models`.

                **L-TAE defaults** (used when ``model='ltae'`` and no
                override is provided):

                * ``n_head`` (*int*) — number of attention heads (default 16).
                * ``d_k`` (*int*) — key / query dimension (default 8).
                * ``d_model`` (*int*) — projection dimension (default 256).
                * ``n_neurons`` (*tuple[int, ...]*) — MLP widths; the first
                  element must equal *d_model* (default ``(256, 128)``).
                * ``dropout`` (*float*) — dropout rate (default 0.2).
                * ``len_max_seq`` (*int*) — maximum sequence length for the
                  positional encoding table (default 24).
                * ``T`` (*int*) — period for sinusoidal positional encoding
                  (default 1000).

        Raises:
            ValueError: If *model* or *loss* is not a recognised value.

        .. versionadded:: 0.10
        """
        super().__init__()

    # Configuration hooks (called by BaseTask.__init__ via super())

    def configure_models(self) -> None:
        """Initialise the model.

        Reads ``self.hparams['model']`` and builds the corresponding
        architecture.  Model-specific hyper-parameters are retrieved from
        ``self.hparams`` (they were stored there by
        :meth:`~lightning.LightningModule.save_hyperparameters` via
        ``**model_kwargs``).

        Raises:
            ValueError: If :attr:`hparams` ``['model']`` is not a supported
                architecture name.
        """
        model: str = self.hparams['model']

        match model:
            case 'ltae':
                n_head = self.hparams.get('n_head', 16)
                d_k = self.hparams.get('d_k', 8)
                d_model = self.hparams.get('d_model', 256)
                n_neurons = self.hparams.get('n_neurons', (256, 128))
                dropout = self.hparams.get('dropout', 0.2)
                len_max_seq = self.hparams.get('len_max_seq', 24)
                T = self.hparams.get('T', 1000)

                self.model = nn.Sequential(
                    LTAE(
                        in_channels=self.hparams['in_channels'],
                        n_head=n_head, d_k=d_k, d_model=d_model,
                        n_neurons=n_neurons, dropout=dropout,
                        len_max_seq=len_max_seq, T=T,
                    ),
                    nn.Linear(n_neurons[-1], self.hparams['num_outputs']),
                )

            case _:
                raise ValueError(
                    f"Model '{model}' is not supported. "
                    "Currently only 'ltae' is available. "
                    'Extend configure_models() to add further architectures.'
                )

    def configure_losses(self) -> None:
        """Initialise the loss criterion.

        Raises:
            ValueError: If :attr:`hparams` ``['loss']`` is not valid.
        """
        loss: str = self.hparams['loss']
        match loss:
            case 'mse':
                self.criterion: nn.Module = nn.MSELoss()
            case 'mae':
                self.criterion = nn.L1Loss()
            case _:
                raise ValueError(
                    f"Loss type '{loss}' is not valid. "
                    "Currently supports 'mse' or 'mae'."
                )

    def configure_metrics(self) -> None:
        """Initialise the performance metrics.

        The following metrics are tracked for every split
        (``train_``, ``val_``, ``test_``):

        * :class:`~torchmetrics.MeanSquaredError` (``MSE``) and its square
          root (``RMSE``).  Lower is better.
        * :class:`~torchmetrics.MeanAbsoluteError` (``MAE``).  Lower is
          better.

        When the datamodule passes ``'mean'`` / ``'std'`` in the batch the
        metrics are evaluated in the unnormalised space via
        :meth:`_unnormalise`.
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

    def _prepare_targets(self, batch: Sample) -> tuple[Tensor, Tensor, Tensor]:
        """Extract and reshape input / target tensors from a batch.

        This method encapsulates the assumptions the training loop makes
        about *batch structure* and *model output shape*:

        1. The model input is found at :attr:`input_key`.
        2. The regression target is found at :attr:`target_key`.
        3. The target is flattened to ``(B, H*C)`` so it aligns with the
           flat ``(B, num_outputs)`` vector produced by sequence-to-value
           models such as L-TAE.

        Args:
            batch: Output of the DataLoader.

        Returns:
            A three-tuple ``(x, y_flat, y_raw)`` where

            * ``x``: input tensor ``(B, T, C)``
            * ``y_flat``: flattened target ``(B, H*C)`` used for loss
              computation and metric updates.
            * ``y_raw``: original target ``(B, H, C)``; ``H`` is extracted 
                and passed to :meth:`_unnormalise`.
        """
        x: Tensor = batch[self.input_key]  # (B, T, C)
        y_raw: Tensor = batch[self.target_key]  # (B, H, C)
        y_flat = y_raw.reshape(x.shape[0], -1)  # (B, H*C)
        return x, y_flat, y_raw

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
        std_rep = std.repeat(H).to(y_hat)    # (H*C,)

        y_hat_orig = y_hat * std_rep + mean_rep
        y_orig = y * std_rep + mean_rep
        return y_hat_orig, y_orig

    def _common_step(self, batch: Sample, batch_idx: int, stage: str) -> Tensor:
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
        x, y, y_raw = self._prepare_targets(batch)
        batch_size: int = x.shape[0]

        y_hat: Tensor = self.model(x)  # (B, num_outputs)

        loss: Tensor = self.criterion(y_hat, y)
        self.log(
            f'{stage}_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size
        )

        # Compute metrics in original space when normalisation stats are
        # available; fall back to normalised space otherwise (no-op).
        H: int = y_raw.shape[1]
        y_hat_m, y_m = self._unnormalise(y_hat, y, batch, H)

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat_m, y_m)
        self.log_dict(metrics, batch_size=batch_size)

        return loss

    # Lightning step methods

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
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._common_step(batch, batch_idx, 'test')

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
        x: Tensor = batch[self.input_key]
        y_hat: Tensor = self.model(x)
        H: int = self.hparams['num_outputs'] // self.hparams['in_channels']
        # y is unused in predict context; passing y_hat as a placeholder
        y_hat, _ = self._unnormalise(y_hat, y_hat, batch, H)
        return y_hat