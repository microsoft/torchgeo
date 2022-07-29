# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Fully convolutional change detection (FCCD) implementations."""

from typing import Any, Callable, Optional, Sequence, Union

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base.model import SegmentationModel
from torch import Tensor

Unet.__module__ = "segmentation_models_pytorch"


class FCSiamConc(SegmentationModel):  # type: ignore[misc]
    """Fully-convolutional Siamese Concatenation (FC-Siam-conc).

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1109/ICIP.2018.8451652
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
    ):
        """Initialize a new FCSiamConc model.

        Args:
            encoder_name: Name of the classification model that will
                be used as an encoder (a.k.a backbone) to extract features
                of different spatial resolution
            encoder_depth: A number of stages used in encoder in range [3, 5].
                two times smaller in spatial dimensions than previous one
                (e.g. for depth 0 we will have features. Each stage generate
                features with shapes [(N, C, H, W),], for depth
                1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
            encoder_weights: One of **None** (random initialization), **"imagenet"**
                (pre-training on ImageNet) and other pretrained weights (see table
                with available weights for each encoder_name)
            decoder_channels: List of integers which specify **in_channels**
                parameter for convolutions used in decoder. Length of the list
                should be the same as **encoder_depth**
            decoder_use_batchnorm: If **True**, BatchNorm2d layer between
                Conv2D and Activation layers is used. If **"inplace"** InplaceABN
                will be used, allows to decrease memory consumption. Available
                options are **True, False, "inplace"**
            decoder_attention_type: Attention module used in decoder of the model.
                Available options are **None** and **scse**. SCSE paper
                https://arxiv.org/abs/1808.08127
            in_channels: A number of input channels for the model,
                default is 3 (RGB images)
            classes: A number of classes for output mask (or you can think as a number
                of channels of output mask)
            activation: An activation function to apply after the final convolution
                n layer. Available options are **"sigmoid"**, **"softmax"**,
                **"logsoftmax"**, **"tanh"**, **"identity"**, **callable**
                and **None**. Default is **None**
        """
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_out_channels = [c * 2 for c in self.encoder.out_channels[1:]]
        encoder_out_channels.insert(0, self.encoder.out_channels[0])
        try:
            # smp 0.3+
            UnetDecoder = smp.decoders.unet.decoder.UnetDecoder
        except AttributeError:
            # smp 0.2
            UnetDecoder = smp.unet.decoder.UnetDecoder
        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.classification_head = None
        self.name = f"u-{encoder_name}"
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input images of shape (b, t, c, h, w)

        Returns:
            predicted change masks of size (b, classes, h, w)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [
            torch.cat([features2[i], features1[i]], dim=1)
            for i in range(1, len(features1))
        ]
        features.insert(0, features2[0])
        decoder_output = self.decoder(*features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks


class FCSiamDiff(Unet):  # type: ignore[misc]
    """Fully-convolutional Siamese Difference (FC-Siam-diff).

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1109/ICIP.2018.8451652
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a new FCSiamConc model.

        Args:
            encoder_name: Name of the classification model that will
                be used as an encoder (a.k.a backbone) to extract features
                of different spatial resolution
            encoder_depth: A number of stages used in encoder in range [3, 5].
                two times smaller in spatial dimensions than previous one
                (e.g. for depth 0 we will have features. Each stage generate
                features with shapes [(N, C, H, W),], for depth
                1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
            encoder_weights: One of **None** (random initialization), **"imagenet"**
                (pre-training on ImageNet) and other pretrained weights (see table
                with available weights for each encoder_name)
            decoder_channels: List of integers which specify **in_channels**
                parameter for convolutions used in decoder. Length of the list
                should be the same as **encoder_depth**
            decoder_use_batchnorm: If **True**, BatchNorm2d layer between
                Conv2D and Activation layers is used. If **"inplace"** InplaceABN
                will be used, allows to decrease memory consumption. Available
                options are **True, False, "inplace"**
            decoder_attention_type: Attention module used in decoder of the model.
                Available options are **None** and **scse**. SCSE paper
                https://arxiv.org/abs/1808.08127
            in_channels: A number of input channels for the model,
                default is 3 (RGB images)
            classes: A number of classes for output mask (or you can think as a number
                of channels of output mask)
            activation: An activation function to apply after the final convolution
                n layer. Available options are **"sigmoid"**, **"softmax"**,
                **"logsoftmax"**, **"tanh"**, **"identity"**, **callable**
                and **None**. Default is **None**
        """
        kwargs["aux_params"] = None
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input images of shape (b, t, c, h, w)

        Returns:
            predicted change masks of size (b, classes, h, w)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [features2[i] - features1[i] for i in range(1, len(features1))]
        features.insert(0, features2[0])
        decoder_output = self.decoder(*features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks
