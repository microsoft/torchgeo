# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for regression."""

import os
from collections.abc import Sequence
from typing import Literal

import kornia.augmentation as K
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import timm
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.models._api import WeightsEnum

from ..datamodules import BaseDataModule
from ..datasets import RGBBandsMissingError, unbind_samples
from ..datasets.utils import Sample
from ..losses import PinballLoss
from ..models import FCN, get_weight
from . import utils
from .base import BaseTask
from .mixins import RegressionMixin


class Regression(RegressionMixin, BaseTask):
    """Regression."""

    target_key = 'label'

    def __init__(
        self,
        model: str = 'resnet50',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        num_outputs: int = 1,
        labels: list[str] | None = None,
        num_filters: int = 3,
        loss: Literal['mae', 'mse', 'pinball'] = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> None:
        """Initialize a new Regression instance.

        Args:
            model: Name of the
                `timm <https://huggingface.co/docs/timm/reference/models>`__ or
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the
                `timm <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or
                `smp <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone
                to use. Only applicable to PixelwiseRegression.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_outputs: Number of prediction outputs.
            labels: List of feature names.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: One of 'mse', 'mae', or 'pinball'. Quantile regression with
                'pinball' requires num_outputs=1.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegression.
            quantiles: Quantile levels to predict when loss='pinball'. Must include
                0.5, which is used for metrics and plotting. Predictions contain
                one channel per quantile, in this order. These estimates are not
                calibrated prediction intervals.

        Raises:
            ValueError: If pinball loss is used with multiple targets, invalid
                quantile levels, or without the median quantile.

        .. versionchanged:: 0.4
           Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.

        .. versionadded:: 0.10
           The *labels* parameter.

        .. versionadded:: 0.11
           The *quantiles* parameter and 'pinball' loss.
        """
        self.median_index = None
        if loss == 'pinball':
            if num_outputs != 1 or 0.5 not in quantiles:
                raise ValueError(
                    'Pinball loss requires num_outputs=1 and quantile 0.5.'
                )
            self.median_index = quantiles.index(0.5)
        self.weights = weights
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        # Create model
        weights = self.weights
        self.model = timm.create_model(
            self.hparams['model'],
            num_classes=(
                len(self.hparams['quantiles'])
                if self.median_index is not None
                else self.hparams['num_outputs']
            ),
            in_chans=self.hparams['in_channels'],
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(
                    progress=True, check_hash=True, weights_only=True
                )
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(
                    progress=True, check_hash=True, weights_only=True
                )
            utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams['freeze_backbone']:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():  # ty: ignore[call-non-callable]
                param.requires_grad = True

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        if self.hparams['loss'] == 'pinball':
            self.criterion = PinballLoss(self.hparams['quantiles'])
        else:
            super().configure_losses()

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch['image']
        batch_size = x.shape[0]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss: Tensor = self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size)
        if self.median_index is not None:
            y_hat = y_hat[:, self.median_index : self.median_index + 1].contiguous()
        self.train_metrics(y_hat, y)

        return loss

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        batch_size = x.shape[0]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size)
        if self.median_index is not None:
            y_hat = y_hat[:, self.median_index : self.median_index + 1].contiguous()
        self.val_metrics(y_hat, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and isinstance(self.trainer.datamodule, BaseDataModule)
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule
            aug = K.AugmentationSequential(
                K.Denormalize(datamodule.mean, datamodule.std),
                data_keys=None,
                keepdim=True,
            )
            batch = aug(batch)
            if self.target_key == 'mask':
                y = y.squeeze(dim=1)
                y_hat = y_hat.squeeze(dim=1)
            batch['prediction'] = y_hat
            for key in ['image', self.target_key, 'prediction']:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )  # ty: ignore[call-non-callable]
                plt.close()

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        batch_size = x.shape[0]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, batch_size=batch_size)
        if self.median_index is not None:
            y_hat = y_hat[:, self.median_index : self.median_index + 1].contiguous()
        self.test_metrics(y_hat, y)

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted regression values.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Regression predictions. With pinball loss, the shape is (B, Q)
            or (B, Q, H, W), with channels in the order of *quantiles*.
        """
        x = batch['image']
        y_hat: Tensor = self(x)
        return y_hat


class PixelwiseRegression(Regression):
    """LightningModule for pixelwise regression of images.

    .. versionadded:: 0.5
    """

    target_key = 'mask'

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights

        model = self.hparams['model']
        backbone = self.hparams['backbone']
        in_channels = self.hparams['in_channels']
        num_outputs = (
            len(self.hparams['quantiles']) if self.median_index is not None else 1
        )

        match model:
            case 'unet':
                self.model = smp.Unet(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_outputs,
                )
            case 'deeplabv3+':
                self.model = smp.DeepLabV3Plus(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_outputs,
                )
            case 'fcn':
                self.model = FCN(
                    in_channels=in_channels,
                    classes=num_outputs,
                    num_filters=self.hparams['num_filters'],
                )
            case 'upernet':
                self.model = smp.UPerNet(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_outputs,
                )
            case 'segformer':
                self.model = smp.Segformer(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_outputs,
                )
            case 'dpt':
                self.model = smp.DPT(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_outputs,
                )

        if model != 'fcn' and weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(
                    progress=True, check_hash=True, weights_only=True
                )
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(
                    progress=True, check_hash=True, weights_only=True
                )
            self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams.get('freeze_backbone', False) and model != 'fcn':
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams.get('freeze_decoder', False) and model != 'fcn':
            for param in self.model.decoder.parameters():
                param.requires_grad = False
