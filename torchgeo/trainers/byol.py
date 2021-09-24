# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake datasets."""
from typing import Any, Callable, Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from kornia import augmentation as K
from kornia.geometry import transform as KorniaTransform
from kornia import filters
import random
from torch import optim
from copy import deepcopy
from torchvision.models import resnet18
from torch.autograd import Variable


DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"

def simCLR_default_augmentation(image_size: Tuple[int, int]= (256, 256)) -> nn.Module:
    return nn.Sequential(
        KorniaTransform.Resize(size=image_size),
        #RandomApply(K.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8), Not suitable for multispectral adapt
      #  K.RandomGrayscale(p=0.2), Not suitable for multispectral
        K.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        K.RandomResizedCrop(size=image_size),
    )


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.mean(2 - 2 * (x * y).sum(dim=-1))


def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    # ---------- Methods for registering the forward hook ----------
    # For more info on PyTorch hook, see:
    # https://towardsdatascience.com/how-to-use-pytorch-hooks-5041d777f904
    
    def _hook(self, _, __, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            # If we haven't already, measure the output size
            self._projector_dim = output.shape[-1]

        # Project the output to get encodings
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)
        
    # ------------------- End hooks methods ----------------------

    def forward(self, x: Tensor) -> Tensor:
        # Pass through the model, and collect 'encodings' from our forward hook!
        _ = self.model(x)
        return self._encoded


class BYOL(LightningModule):
    """LightningModule for training models on the Chesapeake CVPR Land Cover dataset.


    """
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (256, 256),
        hidden_layer: Union[str, int] = -2,
        n_input_channel: int = 4,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.99,
        **kwargs: Any,
    ) :
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.augment = simCLR_default_augmentation(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.n_input_channel = n_input_channel
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self.save_hyperparameters()  # creates `self.hparams` from kwargs
        self._target = None

        # Perform a single forward pass, which initializes the 'projector' in our 
        # 'EncoderWrapper' layer.
        self.encoder(torch.zeros(2, self.n_input_channel, *image_size))
        


    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data



class BYOLTask(LightningModule):
    """LightningModule for training models on the Chesapeake CVPR Land Cover dataset.
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["encoder"] == "resnet18":
            encoder = resnet18(pretrained=True)
            layer = encoder.conv1
            n_input_channel = 4
        
            # Creating new Conv2d layer
            new_layer = nn.Conv2d(in_channels=n_input_channel, 
                            out_channels=layer.out_channels, 
                            kernel_size=layer.kernel_size, 
                            stride=layer.stride, 
                            padding=layer.padding,
                            bias=layer.bias).requires_grad_()

            copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights

            # Copying the weights from the old to the new layer
            new_layer.weight[:, :layer.in_channels, :, :].data[...] = Variable(layer.weight.clone(), requires_grad=True)

            #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            for i in range(n_input_channel - layer.in_channels):
                channel = layer.in_channels + i
                new_layer.weight[:, channel:channel+1, :, :].data[...]= Variable(layer.weight[:, copy_weights:copy_weights+1, : :].clone(), requires_grad=True)

            encoder.conv1 = new_layer 
        else:
            raise ValueError(
                f"Model type '{self.hparams['encoder']}' is not valid."
            )

        if self.hparams["model"] == "byol":
            self.model = BYOL(encoder, image_size=(256, 256))
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid."
            )


    def __init__(
        self,
        n_input_channel = 4,
        **kwargs: Any,
    ) :
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.n_input_channel = n_input_channel
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

        
    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    

    
    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], *_
    ) -> Dict[str, Union[Tensor, Dict]]:
        """Training step - reports average accuracy and average IoU.

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
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)) 

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.model.update_target()

        return {"loss": loss}

    
    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x= batch["image"]
        x1, x2 = self.model.augment(x), self.model.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        return {"loss": loss}


    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())
    
