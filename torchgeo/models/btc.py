# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
#
# Based on the original code: https://github.com/blaz-r/BTC-change-detection

"""Be The Change (BTC) change detection model implementation."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models.feature_extraction import create_feature_extractor

from torchgeo.models.swin import (
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    swin_b,
    swin_s,
    swin_t,
)


class BTC(Module):
    """Be The Change (BTC) change detection model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2507.03367

    .. versionadded:: 0.8
    """

    def __init__(self, backbone: str, classes: int = 1) -> None:
        """Initialise BTC model.

        Args:
            backbone: backbone type (either swin_tiny, swin_small or swin_base).
            classes: number of classes (default is 1).
        """
        super().__init__()
        self.encoder = SwinBackbone(backbone)
        self.difference = subtraction_fusion
        # pad at the beginning since smp impl. cuts first two elements off
        self.decoder = smp.decoders.upernet.decoder.UPerNetDecoder(
            encoder_channels=[0, 0, *self.encoder.channels],
            encoder_depth=4,
            decoder_channels=512,
        )
        # we already have layernorms as part of backbone
        self.decoder.feature_norms = nn.ModuleList(
            [nn.Identity() for _ in self.encoder.channels]
        )
        self.final_layer = smp.base.SegmentationHead(
            in_channels=512,
            out_channels=classes,
            activation=None,
            kernel_size=1,
            upsampling=0,  # avoid here in case of uneven factor
        )
        smp.base.model.init.initialize_decoder(self.decoder)
        smp.base.model.init.initialize_head(self.final_layer)
        # padding tensors used for compatibility with UPerNet implementation
        self.upernet_padding = [torch.tensor(0), torch.tensor(0)]

    def forward(self, x: Tensor) -> Tensor:
        """BTC forward call.

        Extract multi-resolution features, fuse by subtraction, decode with UperNet.

        Args:
            x: input image tensor (b, t*c, h, w)

        Returns:
            binary change map prediction [b, n_cls, h, w].
        """
        h, w = x.shape[-2:]
        # change trainer stacks in channel, we want stacked in batch dim for backbone
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', c=3)
        # extract multi-resolution features
        features = self.encoder(x)
        # feature difference by subtraction
        fused = self.difference(features)
        # UperNet impl. skips first 2 feats, we don't want that so we pad with 0
        fused = self.upernet_padding + fused
        # decode to change map
        x = self.decoder(fused)
        x = self.final_layer(x)
        # scale to match input image
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
        return x


class SwinBackbone(Module):
    """Swin backbone for multi-resolution feature extraction."""

    def __init__(self, model_size: str = 'swin_base') -> None:
        """Initialise swin backbone for multi-resolution feature extraction.

        Args:
            model_size: Swin size, one of 'swin_tiny', 'swin_small', or 'swin_base'.
        """
        super().__init__()
        match model_size:
            case 'swin_tiny':
                weights = Swin_T_Weights.CITYSCAPES_SEMSEG
                model = swin_t(weights)
            case 'swin_small':
                weights = Swin_S_Weights.CITYSCAPES_SEMSEG
                model = swin_s(weights)
            case 'swin_base':
                weights = Swin_B_Weights.CITYSCAPES_SEMSEG
                model = swin_b(weights)
            case _:
                raise ValueError(
                    f'Invalid swin size: {model_size}. Possible options: swin_[tiny | small | base]'
                )

        # we select layers before reduction!
        return_layers = ['features.1', 'features.3', 'features.5', 'features.7']
        self.feature_extractor = create_feature_extractor(
            model, return_nodes=return_layers
        )
        self.channels = self._get_feature_channels()
        self.image_normalization = weights.transforms

        norms = []
        for ch in self.channels:
            norms.append(nn.LayerNorm(ch))
        self.norms = nn.ModuleList(norms)

        # load pretrained feature norm weights
        state_dict = weights.get_state_dict(include_norms=True, progress=True)
        self.norms.load_state_dict(state_dict['feat_norms_state_dict'])

    def forward(self, x: Tensor) -> list[Tensor]:
        """Get multi-resolution features and apply layernorm to each level.

        Args:
            x: input image tensor (b*t, c, h, w).

        Returns:
            list of multi-resolution feature tensors list[(b*t, c, h', w')].
        """
        x = self.image_normalization(x)
        features = self.feature_extractor(x)
        output = []
        for feat, norm in zip(features.values(), self.norms):
            n, h, w, _c = feat.shape
            x = norm(feat)
            x = rearrange(x, 'n h w c -> n c h w', n=n, h=h, w=w)
            output.append(x)
        return output

    def _get_feature_channels(self) -> list[int]:
        """Get the number of channels in features.

        Returns:
            list of channels for each feature map in hierarchy.
        """
        is_training = self.feature_extractor.training
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, 256, 256))
        # revert feature extractor training state
        self.feature_extractor.train(is_training)
        # torchvision swin is channel last
        return [feature.shape[-1] for feature in features.values()]


def subtraction_fusion(x: list[Tensor]) -> list[Tensor]:
    """Bi-temporal feature fusion by elementwise subtraction.

    Args:
        x: list of multi-resolution feature tensors list[(b*t c h w)].

    Returns:
        fused feature tensors list[(b c h w)].
    """
    out_features = []
    for feat in x:
        f1, f2 = rearrange(feat, '(b t) c h w -> t b c h w', t=2)
        out_features.append(f1 - f2)

    return out_features
