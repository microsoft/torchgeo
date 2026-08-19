# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pretrained Distillation for EO (DEO) model implementation."""

from functools import partial
from typing import Any, Literal

import torch
from torch import nn
from torchvision import models as torchvision_models
from torchvision import transforms
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models.swin_transformer import ShiftedWindowAttention
from torchvision.ops.misc import Permute


class DEO(nn.Module):
    """Pretrained Distillation for EO (DEO) model implementation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2602.19863

    .. versionadded:: 0.10
    """

    def __init__(self, model: Literal['swin_b'] = 'swin_b') -> None:
        """Initialise DEO model.

        Args:
            model: backbone type (for now swin_b).
        """
        super().__init__()

        # initialize the backbone
        self.feat_extr = torchvision_models.__dict__[model]()
        del self.feat_extr.features[0]
        del self.feat_extr.head

        # Swin window size used during pretraining (torchvision defaults to 7x7)
        window_size = [12, 12]
        for module in self.feat_extr.modules():
            if isinstance(module, ShiftedWindowAttention):
                shifted = sum(module.shift_size) > 0
                module.window_size = list(window_size)
                module.shift_size = [w // 2 if shifted else 0 for w in window_size]
                module.define_relative_position_bias_table()
                module.define_relative_position_index()

        # Conv layers for Swin
        norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
        norm_layer_rgb = partial(nn.LayerNorm, eps=1e-5)
        self.feat_extr.conv_ms = nn.Sequential(
            nn.Conv2d(
                10,
                self.feat_extr.features[0][0].norm1.normalized_shape[0],
                kernel_size=(4, 4),
                stride=(4, 4),
            ),
            Permute([0, 2, 3, 1]),
            norm_layer_ms(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
        )
        self.feat_extr.conv_rgb = nn.Sequential(
            nn.Conv2d(
                3,
                self.feat_extr.features[0][0].norm1.normalized_shape[0],
                kernel_size=(4, 4),
                stride=(4, 4),
            ),
            Permute([0, 2, 3, 1]),
            norm_layer_rgb(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
        )

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get multi-stage swin features.

        Args:
            x: input image tensor (b, c, h, w).

        Returns:
            list of swin feature tensors list[(b, c, h', w')].
        """
        features = []
        # apply the appropriate conv layer based on the number of input channels
        if x.shape[1] == 10:
            x = self.feat_extr.conv_ms(x)
        else:
            x = self.feat_extr.conv_rgb(x)

        # extract intermediate swin layers
        for i, layer in enumerate(self.feat_extr.features):
            x = layer(x)
            if i in [0, 2, 4, 6]:
                features.append(x)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get last layer swin features.

        Args:
            x: input image tensor (b, c, h, w).

        Returns:
            swin feature tensor (b, c, h', w').
        """
        return self.forward_features(x)[-1]


# Transforms used during pretraining
_deo_transforms = transforms.Normalize(
    (
        # rgb norms for high res optical bands
        0.4182007312774658,
        0.4214799106121063,
        0.3991275727748871,
        # ms norms for 7 band S2 multispectral (60m left out)
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        1732.16362238,
        1247.91870117,
    ),
    (
        # rgb norms for high res optical bands
        0.28774282336235046,
        0.27541765570640564,
        0.2764017581939697,
        # ms norms for 7 band S2 multispectral (60m left out)
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        1310.36996126,
        1087.6020813,
    ),
)


class DEO_Weights(WeightsEnum):
    """DEO base model weights.

    .. versionadded:: 0.10
    """

    DEO_SWIN = Weights(
        url='https://huggingface.co/SolaireTheSun/DEO/resolve/f973d29f6324fb12fca734778cf9d2ae539524bb/DEO_swin_b.pth',
        transforms=_deo_transforms,
        meta={
            'dataset': 'fMoW, fMoW-Sentinel',
            'model': 'Swin_b',
            'publication': 'https://arxiv.org/abs/2602.19863',
            'repo': 'https://github.com/wolfilip/DEO-FM',
            'license': 'MIT',
            'ssl_method': 'DEO',
            'bands': ['B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
            'in_chans': 10,
            'img_size': 224,
        },
    )


def deo_base(weights: DEO_Weights | None = None, *args: Any, **kwargs: Any) -> DEO:
    """DEO Swin model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2602.19863

    Args:
        weights: Pretrained weights to load.
        *args: Additional arguments to pass to :class:`DEO`.
        **kwargs: Additional keyword arguments to pass to :class:`DEO`.

    Returns:
        DEO Swin model.
    """
    model = DEO(*args, **kwargs)

    if weights:
        state_dict = weights.get_state_dict(
            progress=True, check_hash=True, weights_only=True
        )
        model.load_state_dict(state_dict, strict=False)

        assert set(model.state_dict()) & set(state_dict)
        assert tuple(
            state_dict['feat_extr.features.0.0.attn.relative_position_bias_table'].shape
        ) == tuple(
            model.state_dict()[
                'feat_extr.features.0.0.attn.relative_position_bias_table'
            ].shape
        )

    return model
