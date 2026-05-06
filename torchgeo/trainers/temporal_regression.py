# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for temporal regression."""

from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import Sample
from ..models import LTAE
from .base import BaseTask


class TemporalRegressionTask(BaseTask):
    """Trainer for sequence-to-value temporal regression.

    Supports regression over temporal inputs where a fixed-length window of
    past observations is used to predict one or more future values.

    * **Model** — any model that accepts ``(B, T, C)`` and returns
      ``(B, num_outputs)`` can be registered in :meth:`configure_models` with a
      new ``case`` branch.  Model-specific constructor arguments are forwarded
      via ``**model_kwargs``.
    * **Dataset ** — the batch keys consumed by the training loop are
      defined as class-level constants (:attr:`input_key` / :attr:`target_key`)
      and can be overridden in a subclass.  Optional normalisation stats
      (``'mean'`` / ``'std'``) are handled through the :meth:`_unnormalise`
      hook, which is a no-op when those keys are absent.

    **Concrete defaults**

    The default configuration wires up the
    :class:`~torchgeo.models.LTAE` encoder with the
    :class:`~torchgeo.datasets.AirQuality` datamodule, but the design
    explicitly supports future extension.

    **Input batch keys**

    * ``input_key`` (default ``'x_input'``) — ``(B, T, C)`` float
      tensor of past observations fed to the model.
    * ``target_key`` (default ``'y_target'``) — ``(B, H, C)`` float
      tensor of ground-truth future values.
    * ``'mean'`` *(optional)* — per-feature mean used to unnormalise
      predictions before metric computation.
    * ``'std'`` *(optional)* — per-feature standard deviation used to
      unnormalise predictions before metric computation.

    **Logged metrics**

    * ``{split}_loss`` — optimisation loss (MSE or MAE).
    * ``{split}_RMSE``, ``{split}_MSE``, ``{split}_MAE`` — regression
      metrics computed in the *original* (unnormalised) space when
      ``'mean'`` / ``'std'`` are present in the batch, otherwise in the
      normalised space.

    .. versionadded:: 0.10
    """

    #: Batch key for the model input tensor ``(B, T, C)``.
    #: Override in a subclass to match your dataset's field name.
    input_key: str = 'x_input'

    #: Batch key for the regression target tensor ``(B, H, C)``.
    #: Override in a subclass to match your dataset's field name.
    target_key: str = 'y_target'

    def __init__(
        self,
        model: str = 'ltae',
        in_channels: int = 13,
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
            lr: Learning rate for the Adam optimiser.
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

        To add a new architecture, add a ``case`` branch here, and no other
        method needs to change.

        Raises:
            ValueError: If :attr:`hparams` ``['model']`` is not a supported
                architecture name.
        """
        model: str = self.hparams['model']

        match model:
            case 'ltae':
                # Resolve L-TAE hyper-parameters, falling back to defaults
                # so callers do not have to specify every knob explicitly.
                n_head: int = self.hparams.get('n_head', 16)
                d_k: int = self.hparams.get('d_k', 8)
                d_model: int = self.hparams.get('d_model', 256)
                n_neurons: tuple[int, ...] = self.hparams.get('n_neurons', (256, 128))
                dropout: float = self.hparams.get('dropout', 0.2)
                len_max_seq: int = self.hparams.get('len_max_seq', 24)
                T: int = self.hparams.get('T', 1000)

                self.encoder = LTAE(
                    in_channels=self.hparams['in_channels'],
                    n_head=n_head,
                    d_k=d_k,
                    d_model=d_model,
                    n_neurons=n_neurons,
                    dropout=dropout,
                    len_max_seq=len_max_seq,
                    T=T,
                )
                # Thin linear head on top of the L-TAE embedding to produce
                # the requested number of output scalars.
                ltae_out_dim: int = n_neurons[-1]
                self.head: nn.Module = nn.Linear(
                    ltae_out_dim, self.hparams['num_outputs']
                )
                # Wrap into a single callable so BaseTask's self(x) works.
                self.model = nn.Sequential(self.encoder, self.head)

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
        metrics are evaluated in the original (unnormalised) space via
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

    def configure_optimizers(self) -> Any:
        """Initialise the optimiser and learning-rate scheduler.

        Returns:
            A dict with ``'optimizer'``, ``'lr_scheduler'``, and
            ``'monitor'`` keys compatible with Lightning's
            ``configure_optimizers`` contract.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.hparams['patience']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

    # Extensibility hooks

    def _prepare_targets(self, batch: Sample) -> tuple[Tensor, Tensor, Tensor]:
        """Extract and reshape input / target tensors from a batch.

        This method encapsulates the only assumptions the training loop makes
        about *batch structure* and *model output shape*:

        1. The model input is found at :attr:`input_key`.
        2. The regression target is found at :attr:`target_key`.
        3. The target is flattened to ``(B, H*C)`` so it aligns with the
           flat ``(B, num_outputs)`` vector produced by sequence-to-value
           models such as L-TAE.

        Override this method in a subclass when:

        * Your dataset uses different batch key names *and* overriding
          :attr:`input_key` / :attr:`target_key` is insufficient.
        * Your model outputs ``(B, H, C)`` directly (sequence-to-sequence)
          and flattening is not appropriate.

        Args:
            batch: Output of the DataLoader.

        Returns:
            A three-tuple ``(x, y_flat, y_raw)`` where

            * ``x``: input tensor ``(B, T, C)``
            * ``y_flat``: flattened target ``(B, H*C)`` used for loss
              computation and metric updates.
            * ``y_raw``: original target ``(B, H, C)`` retained for use
              in :meth:`: _unnormalise` (which needs the ``H`` dimension).
        """
        x: Tensor = batch[self.input_key]  # (B, T, C)
        y_raw: Tensor = batch[self.target_key]  # (B, H, C)
        y_flat = y_raw.reshape(x.shape[0], -1)  # (B, H*C)
        return x, y_flat, y_raw

    def _unnormalise(
        self, y_hat: Tensor, y: Tensor, batch: Sample, y_raw: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Optionally map predictions and targets back to the original scale.

        When the datamodule supplies per-feature ``'mean'`` and ``'std'``
        tensors in the batch (as :class:`~torchgeo.datamodules.AirQualityDataModule`
        does), both ``y_hat`` and ``y`` are de-normalised before metric
        computation so that reported values are interpretable in physical
        units.

        If neither key is present this method is a **no-op**, returning
        ``y_hat`` and ``y`` unchanged.  This makes the task safe to use with
        any dataset regardless of whether it normalises its targets.

        Override this method to implement a different normalisation
        convention (e.g. min-max scaling, per-sample stats).

        Args:
            y_hat: Model predictions ``(B, num_outputs)`` in normalised space.
            y: Flattened ground-truth targets ``(B, H*C)`` in normalised space.
            batch: The full batch dict; may contain ``'mean'`` and ``'std'``.
            y_raw: Original (un-flattened) target ``(B, H, C)`` used to
                recover the number of future steps ``H``.

        Returns:
            ``(y_hat_orig, y_orig)`` — tensors in the original scale, or the
            inputs unchanged when no normalisation stats are available.
        """
        mean: Tensor | None = batch.get('mean')  # type: ignore[assignment]
        std: Tensor | None = batch.get('std')  # type: ignore[assignment]

        if mean is None or std is None:
            return y_hat, y

        # Replicate per-feature stats across the H future steps so that the
        # resulting vectors align with the flat (B, H*C) tensors.
        H: int = y_raw.shape[1]
        mean_rep = mean.repeat(H)  # (H*C,)
        std_rep = std.repeat(H)  # (H*C,)

        y_hat_orig = y_hat * std_rep + mean_rep
        y_orig = y * std_rep + mean_rep
        return y_hat_orig, y_orig

    def _visualise_step(self, batch: Sample, batch_idx: int) -> None:
        """Optionally log a sample plot to the TensorBoard logger.

        Called during :meth:`validation_step` for the first few batches.

        The method is a no-op when:

        * ``batch_idx`` is 10 or higher (avoids flooding the logger).
        * No TensorBoard-compatible logger is attached.
        * The datamodule does not expose a ``plot`` method.

        Args:
            batch: The current validation batch.
            batch_idx: Index of this batch within the epoch.
        """
        if batch_idx >= 10:
            return
        if not (
            hasattr(self.trainer, 'datamodule')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            return

        datamodule = self.trainer.datamodule
        if not hasattr(datamodule, 'plot'):
            return

        # Move tensors to CPU before plotting.
        cpu_batch: Sample = {
            k: v.cpu() if isinstance(v, Tensor) else v for k, v in batch.items()
        }
        sample = cpu_batch

        fig: Figure | None = None
        try:
            fig = datamodule.plot(sample)
        except Exception:
            pass

        if fig is not None:
            self.logger.experiment.add_figure(
                f'temporal/{batch_idx}', fig, global_step=self.global_step
            )
            plt.close(fig)

    # Shared forward + loss + metric logic

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
        y_hat_m, y_m = self._unnormalise(y_hat, y, batch, y_raw)

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

        Also calls :meth:`_visualise_step` to optionally log a sample plot.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._common_step(batch, batch_idx, 'val')
        self._visualise_step(batch, batch_idx)

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
        x, _, y_raw = self._prepare_targets(batch)
        y_hat: Tensor = self.model(x)

        # Reuse _unnormalise; we pass a zero tensor as the dummy y because
        # we only care about y_hat_orig here.
        y_dummy = torch.zeros_like(y_hat)
        y_hat, _ = self._unnormalise(y_hat, y_dummy, batch, y_raw)

        return y_hat
