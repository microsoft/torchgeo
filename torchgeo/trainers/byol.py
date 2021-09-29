# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake datasets."""
import random
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from kornia import augmentation as K
from kornia import filters
from kornia.geometry import transform as KorniaTransform
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.autograd import Variable
from torch.nn.modules import BatchNorm1d, Conv2d, Linear, Module, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.resnet import resnet50

DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"
Sequential.__module__ = "torch.nn"
Linear.__module__ = "torch.nn"
ReLU.__module__ = "torch.nn"
BatchNorm1d.__module__ = "torch.nn"
Conv2d.__module__ = "torch.nn"


def simCLR_default_augmentation(image_size: Tuple[int, int] = (256, 256)) -> Module:
    """Applies default augmentation from simCLR.

    Args:
        image_size: Tuple of integers defining the image size
    """
    return Sequential(
        KorniaTransform.Resize(size=image_size),  # type: ignore[attr-defined]
        # Not suitable for multispectral adapt
        # RandomApply(K.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        # K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        K.RandomResizedCrop(size=image_size),
    )


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    """Computes the normalized mean square error between x and y.

    Args:
        x: Tensor x
        y: Tensor y
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    mse = torch.mean(2 - 2 * (x * y).sum(dim=-1))  # type: ignore[attr-defined]
    return cast(Tensor, mse)


def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> Module:
    """MLP used in the projection head.

    Args:
        dim: size of layer to project
        projection_size: First layer MLP projection size
        hidden_size: Size of MLP hidden layer
    """
    return Sequential(
        Linear(dim, hidden_size),
        BatchNorm1d(hidden_size),  # type: ignore[no-untyped-call]
        ReLU(inplace=True),
        Linear(hidden_size, projection_size),
    )


class RandomApply(Module):
    """Applies augmentation function (augm) with probability p."""

    def __init__(
        self, augm: Callable[[Dict[str, Tensor]], Dict[str, Tensor]], p: float
    ) -> None:
        """Initialize RandomApply.

        Keyword Args:
            augm: Augmentation function to apply
            p: Probability with which the augmentation function is applied
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.augm = augm
        self.p = p

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomapply forward method."""
        return x if random.random() > self.p else self.augm(x)


class EncoderWrapper(Module):
    """Encoder wrapper joining model and projection head."""

    def __init__(
        self,
        model: Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ) -> None:
        """Initializes EncoderWrapper.

        Keyword Args:
            model: Model to encode
            projection_size: Size of fist layer of projector MLP
            hidden_size: Size of hidden layer of projector
            layer: Layer from model to project
        """
        super().__init__()  # type: ignore[no-untyped-call]

        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector: Optional[Module] = None
        self._projector_dim = 256
        self._encoded = torch.empty(0)  # type: ignore[attr-defined]
        self._register_hook()

    @property
    def projector(self) -> Module:
        """Wrapper of projector head."""
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    # See: https://towardsdatascience.com/how-to-use-pytorch-hooks-5041d777f904

    def _hook(self, _: Any, __: Any, output: Tensor) -> None:
        """Projection hook."""
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            # If we haven't already, measure the output size
            self._projector_dim = output.shape[-1]

        # Project the output to get encodings
        self._encoded = self.projector(output)

    def _register_hook(self) -> None:
        """Register hook for forward."""
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        """Pass through the model, and collect 'encodings' from our forward hook!"""
        _ = self.model(x)
        return cast(Tensor, self._encoded)


class BYOL(LightningModule):
    """BYOL implementation.

    BYOL contains two identical Encoder networks. The first is trained
    as usual, and its weights are updated with each training batch. The
    second (referred to as the “target” network) is updated using a running
    average of the first Encoder’s weights.
    Citation: Grill JB, Strub F, Altché F, Tallec C, Richemond PH, Buchatskaya E,
    Doersch C, Pires BA, Guo ZD, Azar MG, Piot B. Bootstrap your own latent: A
    new approach to self-supervised learning. arXiv preprint arXiv:2006.07733.
    2020 Jun 13.
    """

    def __init__(
        self,
        model: Module,
        image_size: Tuple[int, int] = (256, 256),
        hidden_layer: Union[str, int] = -2,
        input_channels: int = 4,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Optional[Module] = None,
        beta: float = 0.99,
        **kwargs: Any,
    ) -> None:
        """Initialize BYOL with the pre-training model and projection heads parameters.

        Keyword Args:
            model: Model to pretrain using BYOL
            image_size: Tuple defining the saize of the training images
            hidden_layer: Defines the layer projected
            input_channels: Number of input channels to the model
            projection_size: Size of first layer of projection MLP
            hidden_size: Size of the hidden layer of the projection MLP
            augment_fn: param for augmentation
            beta: Beta parameter on BYOL, dictates speed at which the target encoder
                    is updated using the main encoder

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.augment: Module = (
            simCLR_default_augmentation(image_size)
            if augment_fn is None
            else augment_fn
        )
        self.beta = beta
        self.input_channels = input_channels
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = Linear(projection_size, hidden_size)
        self.save_hyperparameters()  # creates `self.hparams` from kwargs
        self._target: Optional[Module] = None

        # Perform a single forward pass, it initializes the 'projector' of the wrapper
        self.encoder(
            torch.zeros(  # type: ignore[attr-defined]
                2, self.input_channels, *image_size
            )
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        x = self.predictor(self.encoder(x))
        return x

    @property
    def target(self) -> Module:
        """Build target model by copying the encoder."""
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self) -> None:
        """Function to update the target model."""
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data


class BYOLTask(LightningModule):
    """Class for pre-training any PyTorch model using BYOL."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        in_channels = self.hparams["in_channels"]
        encoder = None
        if self.hparams["encoder"] == "resnet18":
            if self.hparams["imagenet_pretraining"]:
                encoder = resnet18(pretrained=True)
            else:
                encoder = resnet18()
        elif self.hparams["encoder"] == "resnet50":
            if self.hparams["imagenet_pretraining"]:
                encoder = resnet50(pretrained=True)
            else:
                encoder = resnet50()
        else:
            raise ValueError(f"Encoder type '{self.hparams['encoder']}' is not valid.")

        layer = encoder.conv1
        # Creating new Conv2d layer
        new_layer = Conv2d(
            in_channels=in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        ).requires_grad_()
        # initialize the weights from new channel with the red channel weights
        copy_weights = 0
        # Copying the weights from the old to the new layer
        new_layer.weight[:, : layer.in_channels, :, :].data[
            ...  # type: ignore[index]
        ] = Variable(layer.weight.clone(), requires_grad=True)
        # Copying the weights of the old layer to the extra channels
        for i in range(in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel : channel + 1, :, :].data[
                ...  # type: ignore[index]
            ] = Variable(
                layer.weight[:, copy_weights : copy_weights + 1, ::].clone(),
                requires_grad=True,
            )

        encoder.conv1 = new_layer
        if self.hparams["model"] == "byol":
            self.model = BYOL(encoder, image_size=(256, 256))
        else:
            raise ValueError(f"Model type '{self.hparams['model']}' is not valid.")

    def __init__(
        self,
        input_channels: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            in_channels: Number of channels on the input imagery

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.input_channels = input_channels
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Define optimizers for lightning."""
        optimizer_class = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        optimizer = optimizer_class(self.parameters(), lr=lr, weight_decay=weight_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports BYOL loss.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        with torch.no_grad():
            x1, x2 = self.model.augment(x), self.model.augment(x)

        pred1, pred2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(  # type: ignore[attr-defined]
            normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)
        )

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.model.update_target()

        return cast(Tensor, loss)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Logs iteration level validation loss."""
        x = batch["image"]
        x1, x2 = self.model.augment(x), self.model.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(  # type: ignore[attr-defined]
            normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)
