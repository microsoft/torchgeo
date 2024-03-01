# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BYOL trainer for self-supervised learning (SSL)."""

import os
from typing import Any, Optional, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation as K
from torch import Tensor
from torchvision.models._api import WeightsEnum

from ..models import get_weight
from . import utils
from .base import BaseTask


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

    def __init__(self, image_size: tuple[int, int] = (256, 256)) -> None:
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
        z: Tensor = self.augmentation(x)
        return z


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
        z: Tensor = self.mlp(x)
        return z


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
        image_size: tuple[int, int] = (256, 256),
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
        z: Tensor = self.predictor(self.backbone(x))
        return z

    def update_target(self) -> None:
        """Method to update the "target" model weights."""
        for p, pt in zip(self.backbone.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data


class BYOLTask(BaseTask):
    """BYOL: Bootstrap Your Own Latent.

    Reference implementation:

    * https://github.com/deepmind/deepmind-research/tree/master/byol

    If you use this trainer in your research, please cite the following paper:

    * https://arxiv.org/abs/2006.07733
    """

    monitor = "train_loss"

    def __init__(
        self,
        model: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> None:
        """Initialize a new BYOLTask instance.

        Args:
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.

        .. versionchanged:: 0.4
           *backbone_name* was renamed to *backbone*. Changed backbone support from
           torchvision.models to timm.

        .. versionchanged:: 0.5
           *backbone*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *model*, *lr*, and *patience*.
        """
        self.weights = weights
        super().__init__(ignore="weights")

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]

        # Create backbone
        backbone = timm.create_model(
            self.hparams["model"], in_chans=in_channels, pretrained=weights is True
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            utils.load_state_dict(backbone, state_dict)

        self.model = BYOL(backbone, in_channels=in_channels, image_size=(224, 224))

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.

        Raises:
            AssertionError: If channel dimensions are incorrect.
        """
        x = batch["image"]

        in_channels = self.hparams["in_channels"]
        assert x.size(1) == in_channels or x.size(1) == 2 * in_channels

        if x.size(1) == in_channels:
            x1 = x
            x2 = x
        else:
            x1 = x[:, :in_channels]
            x2 = x[:, in_channels:]

        with torch.no_grad():
            x1 = self.model.augment(x1)
            x2 = self.model.augment(x2)

        pred1 = self(x1)
        pred2 = self(x2)
        with torch.no_grad():
            targ1 = self.model.target(x1)
            targ2 = self.model.target(x2)

        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("train_loss", loss)
        self.model.update_target()

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""
