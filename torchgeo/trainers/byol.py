# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BYOL tasks."""

import os
from typing import Any, Dict, Optional, Tuple, cast

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation as K
from torch import Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models._api import WeightsEnum

from ..models import get_weight
from . import utils


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    """Computes the normalized mean squared error between x and y.

    Args:
        x: tensor x
        y: tensor y

    Returns:
        the normalized MSE between x and y
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    mse = torch.mean(2 - 2 * (x * y).sum(dim=-1))
    return mse


# TODO: This isn't _really_ applying the augmentations from SimCLR as we have
# multispectral imagery and thus can't naively apply color jittering or grayscale
# conversions. We should think more about what makes sense here.
class SimCLRAugmentation(nn.Module):
    """A module for applying SimCLR augmentations.

    SimCLR was one of the first papers to show the effectiveness of random data
    augmentation in self-supervised-learning setups. See
    https://arxiv.org/pdf/2002.05709.pdf for more details.
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 256)) -> None:
        """Initialize a module for applying SimCLR augmentations.

        Args:
            image_size: Tuple of integers defining the image size
        """
        super().__init__()
        self.size = image_size

        self.augmentation = nn.Sequential(
            K.Resize(size=image_size, align_corners=False),
            # Not suitable for multispectral adapt
            # K.ColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
            # K.RandomGrayscale(p=0.2),
            K.RandomHorizontalFlip(),
            K.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.1),
            K.RandomResizedCrop(size=image_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        return cast(Tensor, self.augmentation(x))


class MLP(nn.Module):
    """MLP used in the BYOL projection head."""

    def __init__(
        self, dim: int, projection_size: int = 256, hidden_size: int = 4096
    ) -> None:
        """Initializes the MLP projection head.

        Args:
            dim: size of layer to project
            projection_size: size of the output layer
            hidden_size: size of the hidden layer
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP model.

        Args:
            x: batch of imagery

        Returns:
            embedded version of the input
        """
        return cast(Tensor, self.mlp(x))


class BackboneWrapper(nn.Module):
    """Backbone wrapper for joining a model and a projection head.

    When we call .forward() on this module the following steps happen:

    * The input is passed through the base model
    * When the encoding layer is reached a hook is called
    * The output of the encoding layer is passed through the projection head
    * The forward call returns the output of the projection head

    .. versionchanged 0.4: Name changed from *EncoderWrapper* to
        *BackboneWrapper*.
    """

    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: int = -2,
    ) -> None:
        """Initializes BackboneWrapper.

        Args:
            model: model to encode
            projection_size: size of the ouput layer of the projector MLP
            hidden_size: size of hidden layer of the projector MLP
            layer: layer from model to project
        """
        super().__init__()

        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector: Optional[nn.Module] = None
        self._projector_dim: Optional[int] = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self) -> nn.Module:
        """Wrapper module for the projector head."""
        assert self._projector_dim is not None
        if self._projector is None:
            self._projector = MLP(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    def _hook(self, module: Any, input: Any, output: Tensor) -> None:
        """Hook to record the activations at the projection layer.

        See the following docs page for more details on hooks:
        https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html

        Args:
            module: the calling module
            input: input to the module this hook was registered to
            output: output from the module this hook was registered to
        """
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            # If we haven't already, measure the output size
            self._projector_dim = output.shape[-1]

        # Project the output to get encodings, the projector model is created the first
        # time this is called
        self._encoded = self.projector(output)

        # Store the image embeddings
        self._embedding = output

    def _register_hook(self) -> None:
        """Register a hook for layer that we will extract features from."""
        layer = list(self.model.children())[self.layer]
        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        """Pass through the model, and collect the representation from our forward hook.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        _ = self.model(x)
        return self._encoded


class BYOL(nn.Module):
    """BYOL implementation.

    BYOL contains two identical backbone networks. The first is trained as usual, and
    its weights are updated with each training batch. The second, "target" network,
    is updated using a running average of the first backbone's weights.

    See https://arxiv.org/abs/2006.07733 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (256, 256),
        hidden_layer: int = -2,
        in_channels: int = 4,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Optional[nn.Module] = None,
        beta: float = 0.99,
        **kwargs: Any,
    ) -> None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            hidden_layer: the hidden layer in ``model`` to attach the projection
                head to, can be the name of the layer or index of the layer
            in_channels: number of input channels to the model
            projection_size: size of first layer of the projection MLP
            hidden_size: size of the hidden layer of the projection MLP
            augment_fn: an instance of a module that performs data augmentation
            beta: the speed at which the target backbone is updated using the main
                backbone
        """
        super().__init__()

        self.augment: nn.Module
        if augment_fn is None:
            self.augment = SimCLRAugmentation(image_size)
        else:
            self.augment = augment_fn

        self.beta = beta
        self.in_channels = in_channels
        self.backbone = BackboneWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = MLP(projection_size, projection_size, hidden_size)
        self.target = BackboneWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )

        # Perform a single forward pass to initialize the wrapper correctly
        self.backbone(torch.zeros(2, self.in_channels, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the backbone model through the MLP and prediction head.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.predictor(self.backbone(x)))

    def update_target(self) -> None:
        """Method to update the "target" model weights."""
        for p, pt in zip(self.backbone.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data


class BYOLTask(pl.LightningModule):
    """Class for pre-training any PyTorch model using BYOL.

    Supports any available `Timm model
    <https://rwightman.github.io/pytorch-image-models/>`_
    as an architecture choice. To see a list of available pretrained
    models, you can do:

    .. code-block:: python

        import timm
        print(timm.list_models())
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        # Create model
        in_channels = self.hyperparams["in_channels"]
        weights = self.hyperparams["weights"]
        backbone = timm.create_model(
            self.hyperparams["backbone"],
            in_chans=in_channels,
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            backbone = utils.load_state_dict(backbone, state_dict)

        self.model = BYOL(backbone, in_channels=in_channels, image_size=(256, 256))

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with BYOL.

        Keyword Args:
            in_channels: Number of input channels to model
            backbone: Name of the timm model to use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        Raises:
            ValueError: if kwargs arguments are invalid

        .. versionchanged:: 0.4
           The *backbone_name* parameter was renamed to *backbone*. Change backbone
           support from torchvision.models to timm.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "Adam"))
        lr = self.hyperparams.get("learning_rate", 1e-4)
        weight_decay = self.hyperparams.get("weight_decay", 1e-6)
        optimizer = optimizer_class(self.parameters(), lr=lr, weight_decay=weight_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        with torch.no_grad():
            x1, x2 = self.model.augment(x), self.model.augment(x)

        pred1, pred2 = self(x1), self(x2)
        with torch.no_grad():
            targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.model.update_target()

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        x1, x2 = self.model.augment(x), self.model.augment(x)
        pred1, pred2 = self(x1), self(x2)
        targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the output embeddings of the image backbone.

        Args:
            batch: the output of your DataLoader

        Returns:
            image embeddings
        """
        batch = args[0]
        x = batch["image"]
        self(x)
        return self.model.backbone._embedding
