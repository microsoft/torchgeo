# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Based on the original code: https://github.com/blaz-r/BTC-change-detection

"""Be The Change (BTC) change detection model implementation."""

import kornia.augmentation as K
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models import Weights, WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.swin_transformer import SwinTransformer, swin_s, swin_t


class BTC(Module):
    """Be The Change (BTC) change detection model.

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1109/TGRS.2025.3585342
    * https://arxiv.org/abs/2507.03367
    """

    def __init__(self, backbone: str, classes: int = 1) -> None:
        """Initialise BTC model.

        Args:
            backbone: backbone type (either swin_tiny, swin_small or swin_base).
            classes: number of classes (default is 1).
        """
        super().__init__()
        self.backbone = SwinBackbone(backbone)
        self.difference = subtraction_fusion
        self.decoder = smp.decoders.upernet.decoder.UPerNetDecoder(
            encoder_channels=[
                0,
                0,
                *self.backbone.channels,
            ],  # pad at the beginning since impl. cuts first two off
            encoder_depth=4,
            decoder_channels=512,
        )
        # we already have layernorms as part of backbone
        self.decoder.feature_norms = nn.ModuleList(
            [nn.Identity() for _ in self.backbone.channels]
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

    def forward(self, x: Tensor) -> Tensor:
        """BTC forward call.

        Extract multi-resolution features, fuse by subtraction, decode with UperNet.

        Args:
            x: input image tensor (b, t*c, h, w)
        """
        h, w = x.shape[-2:]
        # change trainer stacks in channel, we want stacked in batch dim for backbone
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', c=3)
        # extract multi-resolution features
        features = self.backbone(x)
        # feature difference by subtraction
        fused = self.difference(features)
        # UperNet impl. skips first 2 feats, we don't want that so we pad with 0
        fused = [torch.tensor(0), torch.tensor(0), *fused]
        # decode to change map
        x = self.decoder(fused)
        x = self.final_layer(x)
        # scale to match input image
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
        return x


class SwinBackbone_Weights(WeightsEnum):  # type: ignore[misc]
    """SwinBackbone weights."""

    CITYSCAPES_SEMSEG_TINY = Weights(
        url='https://huggingface.co/blaz-r/swin_tiny_cityscapes_semantic_torchvision/resolve/main/swin_tiny_cityscapes_semantic.pth',
        transforms=lambda x: x,
        meta={
            'dataset': 'Cityscapes - semantic segmentation',
            'in_chans': 3,
            'model': 'SwinTransformer Tiny',
            'publication': 'https://arxiv.org/abs/2112.01527',
            'repo': 'https://github.com/facebookresearch/Mask2Former/',
            'license': 'mit',
        },
    )
    CITYSCAPES_SEMSEG_SMALL = Weights(
        url='https://huggingface.co/blaz-r/swin_small_cityscapes_semantic_torchvision/resolve/main/swin_small_cityscapes_semantic.pth',
        transforms=lambda x: x,
        meta={
            'dataset': 'Cityscapes - semantic segmentation',
            'in_chans': 3,
            'model': 'SwinTransformer Tiny',
            'publication': 'https://arxiv.org/abs/2112.01527',
            'repo': 'https://github.com/facebookresearch/Mask2Former/',
            'license': 'mit',
        },
    )
    CITYSCAPES_SEMSEG_BASE = Weights(
        url='https://huggingface.co/blaz-r/swin_base_cityscapes_semantic_torchvision/resolve/main/swin_base_cityscapes_semantic.pth',
        transforms=lambda x: x,
        meta={
            'dataset': 'Cityscapes - semantic segmentation',
            'in_chans': 3,
            'model': 'SwinTransformer Tiny',
            'publication': 'https://arxiv.org/abs/2112.01527',
            'repo': 'https://github.com/facebookresearch/Mask2Former/',
            'license': 'mit',
        },
    )


class SwinBackbone(Module):
    """Swin backbone for multi-resolution feature extraction."""

    def __init__(self, model_size: str = 'swin_base') -> None:
        """Initialise swin backbone for multi-resolution feature extraction.

        Args:
            model_size: "tiny" or "base" swin size.
        """
        super().__init__()
        match model_size:
            case 'swin_tiny':
                model = swin_t()
                weights = SwinBackbone_Weights.CITYSCAPES_SEMSEG_TINY
            case 'swin_small':
                model = swin_s()
                weights = SwinBackbone_Weights.CITYSCAPES_SEMSEG_SMALL
            case 'swin_base':
                model = SwinTransformer(
                    patch_size=[4, 4],
                    embed_dim=128,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    window_size=[12, 12],
                    stochastic_depth_prob=0.3,
                )
                weights = SwinBackbone_Weights.CITYSCAPES_SEMSEG_BASE
            case _:
                raise ValueError(
                    f'Invalid swin size: {model_size}. Possible options: swin_[tiny | small | base]'
                )

        # load weights before passing to feature extractor
        weights = weights.get_state_dict(progress=True)
        missing, unexpected = model.load_state_dict(weights['state_dict'], strict=False)
        if len(unexpected) > 0:
            msg = f'Failed to load pretrained weights for backbone: unexpected keys: {unexpected}'
            raise RuntimeError(msg)
        if any(
            key not in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
            for key in missing
        ):
            msg = f'Missing keys in pretrained weights: {missing}'
            raise RuntimeError(msg)

        # we select layers before reduction!
        return_layers = ['features.1', 'features.3', 'features.5', 'features.7']
        self.feature_extractor = create_feature_extractor(
            model, return_nodes=return_layers
        )
        self.channels = self._get_feature_channels()
        self.image_normalization = K.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        norms = []
        for ch in self.channels:
            norms.append(nn.LayerNorm(ch))
        self.norms = nn.ModuleList(norms)
        # load pretrained feature norm weights
        self.norms.load_state_dict(weights['feat_norms_state_dict'])

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
            n, h, w, c = feat.shape
            x = norm(feat)
            x = rearrange(x, 'n h w c -> n c h w', n=n, h=h, w=w)
            output.append(x)
        return output

    def _get_feature_channels(self) -> list[int]:
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, 256, 256))
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
