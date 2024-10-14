# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dynamic One-For-All (DOFA) models."""

from functools import partial
from typing import Any

import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.vision_transformer import Block
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum


def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute the 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: A list of positions to be encoded, of size (M,).

    Returns:
        Position embeddings of size (M, D).

    Raises:
        AssertionError: If *embed_dim* is not even.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    """Dynamic weight generator for DOFA."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize a new TransformerWeightGenerator instance.

        Args:
            input_dim: Input dimensions.
            output_dim: Output dimensions.
            embed_dim: Embedding dimensions.
            num_heads: Number of heads.
            num_layers: Number of layers.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation='gelu',
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch of size (seq_len, batch, input_dim).

        Returns:
            Weight and bias.
        """
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        # Using the last output to generate bias
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    """Fully-connected residual layer."""

    def __init__(self, linear_size: int = 128) -> None:
        """Initialize a new FCResLayer instance.

        Args:
            linear_size: Size of linear layer.
        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.

        Returns:
            Output of the model.
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out: Tensor = x + y
        return out


class DOFAEmbedding(nn.Module):
    """Dynamic One-For-All (DOFA) embedding."""

    def __init__(
        self, dynamic_embed_dim: int, kernel_size: int = 3, embed_dim: int = 1024
    ) -> None:
        """Initialize a new DOFAEmbedding instance.

        Args:
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            kernel_size: Kernel size of the depth-wise convolution.
            embed_dim: Embedding dimensions.
        """
        super().__init__()
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            dynamic_embed_dim, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(dynamic_embed_dim)

        self._init_weights()

    def _init_weight(self, m: object) -> None:
        """Initialize weights of a single layer.

        Args:
            m: A single layer.
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize weights of all layers."""
        self.weight_generator.apply(self._init_weight)
        self.fclayer.apply(self._init_weight)

    def forward(self, x: Tensor, wavelengths: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Return:
            Output mini-batch and wavelengths.
        """
        inplanes = wavelengths.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)  # 3x3x3

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            x, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class DOFA(nn.Module):
    """Dynamic One-For-All (DOFA) model.

    Reference implementation:

    * https://github.com/zhu-xlab/DOFA

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        drop_rate: float = 0.0,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        dynamic_embed_dim: int = 128,
        num_classes: int = 45,
        global_pool: bool = True,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # type: ignore[assignment]
    ) -> None:
        """Initialize a new DOFA instance.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            drop_rate: Head dropout rate.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            num_classes: Number of classes for classification head.
            global_pool: Whether or not to perform global pooling.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            norm_layer: Normalization layer.
        """
        super().__init__()

        self.dynamic_embed_dim = dynamic_embed_dim
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = DOFAEmbedding(
            dynamic_embed_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio

    def forward_features(self, x: Tensor, wavelengths: list[float]) -> Tensor:
        """Forward pass of the feature embedding layer.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Output mini-batch.
        """
        # embed patches
        wavelist = torch.tensor(wavelengths, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome: Tensor = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the attention head.

        Args:
            x: Input mini-batch.
            pre_logits: Whether or not to return the layer before logits are computed.

        Returns:
            Output mini-batch.
        """
        x = self.head_drop(x)
        x = x if pre_logits else self.head(x)
        return x

    def forward(self, x: Tensor, wavelengths: list[float]) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Output mini-batch.
        """
        x = self.forward_features(x, wavelengths)
        x = self.forward_head(x)
        return x


# https://github.com/zhu-xlab/DOFA/blob/master/normalize_dataset.py
# Normalization is sensor-dependent and is therefore left out
_dofa_transforms = K.AugmentationSequential(K.CenterCrop((224, 224)), data_keys=None)

# https://github.com/pytorch/vision/pull/6883
# https://github.com/pytorch/vision/pull/7107
# Can be removed once torchvision>=0.15 is required
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]


class DOFABase16_Weights(WeightsEnum):  # type: ignore[misc]
    """Dynamic One-For-All (DOFA) base patch size 16 weights.

    .. versionadded:: 0.6
    """

    DOFA_MAE = Weights(
        url='https://hf.co/torchgeo/dofa/resolve/b8db318b64a90b9e085ec04ba8851233c5893666/dofa_base_patch16_224-a0275954.pth',
        transforms=_dofa_transforms,
        meta={
            'dataset': 'SatlasPretrain, Five-Billion-Pixels, HySpecNet-11k',
            'model': 'dofa_base_patch16_224',
            'publication': 'https://arxiv.org/abs/2403.15356',
            'repo': 'https://github.com/zhu-xlab/DOFA',
            'ssl_method': 'mae',
        },
    )


class DOFALarge16_Weights(WeightsEnum):  # type: ignore[misc]
    """Dynamic One-For-All (DOFA) large patch size 16 weights.

    .. versionadded:: 0.6
    """

    DOFA_MAE = Weights(
        url='https://hf.co/torchgeo/dofa/resolve/b8db318b64a90b9e085ec04ba8851233c5893666/dofa_large_patch16_224-0ff904d3.pth',
        transforms=_dofa_transforms,
        meta={
            'dataset': 'SatlasPretrain, Five-Billion-Pixels, HySpecNet-11k',
            'model': 'dofa_large_patch16_224',
            'publication': 'https://arxiv.org/abs/2403.15356',
            'repo': 'https://github.com/zhu-xlab/DOFA',
            'ssl_method': 'mae',
        },
    )


def dofa_small_patch16_224(*args: Any, **kwargs: Any) -> DOFA:
    """Dynamic One-For-All (DOFA) small patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6

    Args:
        *args: Additional arguments to pass to :class:`DOFA`.
        **kwargs: Additional keywork arguments to pass to :class:`DOFA`.

    Returns:
        A DOFA small 16 model.
    """
    kwargs |= {'patch_size': 16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6}
    model = DOFA(*args, **kwargs)
    return model


def dofa_base_patch16_224(
    weights: DOFABase16_Weights | None = None, *args: Any, **kwargs: Any
) -> DOFA:
    """Dynamic One-For-All (DOFA) base patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`DOFA`.
        **kwargs: Additional keywork arguments to pass to :class:`DOFA`.

    Returns:
        A DOFA base 16 model.
    """
    kwargs |= {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    model = DOFA(*args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # Both fc_norm and head are generated dynamically
        assert set(missing_keys) <= {
            'fc_norm.weight',
            'fc_norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model


def dofa_large_patch16_224(
    weights: DOFALarge16_Weights | None = None, *args: Any, **kwargs: Any
) -> DOFA:
    """Dynamic One-For-All (DOFA) large patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`DOFA`.
        **kwargs: Additional keywork arguments to pass to :class:`DOFA`.

    Returns:
        A DOFA large 16 model.
    """
    kwargs |= {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16}
    model = DOFA(*args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # Both fc_norm and head are generated dynamically
        assert set(missing_keys) <= {
            'fc_norm.weight',
            'fc_norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model


def dofa_huge_patch16_224(*args: Any, **kwargs: Any) -> DOFA:
    """Dynamic One-For-All (DOFA) huge patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6

    Args:
        *args: Additional arguments to pass to :class:`DOFA`.
        **kwargs: Additional keywork arguments to pass to :class:`DOFA`.

    Returns:
        A DOFA huge 16 model.
    """
    kwargs |= {'patch_size': 14, 'embed_dim': 1280, 'depth': 32, 'num_heads': 16}
    model = DOFA(*args, **kwargs)
    return model
