# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Classification tasks."""

import os

import torch
import torch.nn as nn
import torchvision.models
from torch.nn.modules import Conv2d, Linear

from . import utils
from .classification import ClassificationTask


# TODO: move this functionality into ClassificationTask and remove this class
class So2SatClassificationTask(ClassificationTask):
    """LightningModule for training models on the So2Sat Dataset.

    .. deprecated:: 0.1
       Use :class:`ClassificationTask` instead.
    """

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters passed to the constructor."""
        in_channels = self.hparams["in_channels"]

        pretrained = False
        if self.hparams["weights"] and not os.path.exists(self.hparams["weights"]):
            if self.hparams["weights"] == "imagenet":
                pretrained = True
            elif self.hparams["weights"] == "random":
                pretrained = False
            else:
                raise ValueError(
                    f"Weight type '{self.hparams['weights']}' is not valid."
                )

        # Create the model
        if "resnet" in self.hparams["classification_model"]:
            self.model = getattr(
                torchvision.models.resnet, self.hparams["classification_model"]
            )(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = Linear(
                in_features, out_features=self.hparams["num_classes"]
            )

            # Update first layer
            if in_channels != 3:
                w_old = None
                if pretrained:
                    w_old = torch.clone(  # type: ignore[attr-defined]
                        self.model.conv1.weight
                    ).detach()
                # Create the new layer
                self.model.conv1 = Conv2d(
                    in_channels, 64, kernel_size=7, stride=1, padding=2, bias=False
                )
                nn.init.kaiming_normal_(  # type: ignore[no-untyped-call]
                    self.model.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

                # We copy over the pretrained RGB weights
                if pretrained:
                    w_new = torch.clone(  # type: ignore[attr-defined]
                        self.model.conv1.weight
                    ).detach()
                    w_new[:, :3, :, :] = w_old
                    self.model.conv1.weight = nn.Parameter(  # type: ignore[attr-defined] # noqa: E501
                        w_new
                    )
        else:
            raise ValueError(
                f"Model type '{self.hparams['classification_model']}' is not valid."
            )

        # Load pretrained weights checkpoint weights
        if "resnet" in self.hparams["classification_model"]:
            if os.path.exists(self.hparams["weights"]):
                name, state_dict = utils.extract_encoder(self.hparams["weights"])

                if self.hparams["classification_model"] != name:
                    raise ValueError(
                        f"Trying to load {name} weights into a "
                        f"{self.hparams['classification_model']}"
                    )

                self.model = utils.load_state_dict(self.model, state_dict)
