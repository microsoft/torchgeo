# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Code based on https://github.com/antofuller/CROMA under MIT License

"""CROMA model."""

import itertools
import math
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, einsum, nn
from torchvision.models._api import Weights, WeightsEnum


class CROMA(nn.Module):
    """Pretrained CROMA model.

    Corresponds to the pretrained CROMA mdel found in the CROMA repository:

    * https://github.com/antofuller/CROMA/blob/main/pretrain_croma.py

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.00566
    """

    valid_modalities = ('sar', 'optical')

    def __init__(
        self,
        modalities: list[str] = ['sar', 'optical'],
        encoder_dim: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 16,
        patch_size: int = 8,
        image_size: int = 120,
    ) -> None:
        """Initialize the CROMA model.

        Args:
            modalities: List of modalities used during forward pass, list can contain
                'sar', 'optical', or both.
            encoder_dim: Dimension of the encoder.
            encoder_depth: Depth of the encoder.
            num_heads: Number of heads for the multi-head attention, should be power of 2.
            patch_size: Size of the patches.
            image_size: Size of the input images, CROMA was trained on 120x120 images,
                must be a multiple of 8.

        Raises:
            AssertionError: If the modality is not valid.
        """
        super().__init__()
        for modality in modalities:
            assert (
                modality in self.valid_modalities
            ), f'{modality} is not a valid modality'

        assert image_size % 8 == 0, 'image_size must be a multiple of 8'
        assert num_heads % 2 == 0, 'num_heads must be a power of 2'

        self.modalities = modalities
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.image_size = image_size

        self.num_patches = int((image_size / 8) ** 2)
        self.s1_channels = 2  # fixed at 2 SAR backscatter channels
        self.s2_channels = 12  # fixed at 12 multispectral optical channels

        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        def initialize_encoder(
            encoder_dim: int, encoder_depth: int, in_channels: int
        ) -> tuple[nn.Module, nn.Module]:
            """Initialize the encoder and GAP-FFN for a given modality.

            Args:
                encoder_dim: Dimension of the encoder.
                encoder_depth: Depth of the encoder.
                in_channels: Number of input channels.

            Returns:
                Tuple containing the encoder and GAP-FFN.
            """
            encoder = ViT(dim=encoder_dim, depth=encoder_depth, in_channels=in_channels)
            gap_ffn = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, int(4 * encoder_dim)),
                nn.GELU(),
                nn.Linear(int(4 * encoder_dim), encoder_dim),
            )
            return encoder, gap_ffn

        if 'sar' in modalities:
            self.s1_encoder, self.s1_GAP_FFN = initialize_encoder(
                encoder_dim, int(encoder_depth / 2), self.s1_channels
            )
        if 'optical' in modalities:
            self.s2_encoder, self.s2_GAP_FFN = initialize_encoder(
                encoder_dim, encoder_depth, self.s2_channels
            )
        if set(self.modalities) == {'sar', 'optical'}:
            self.joint_encoder = BaseTransformerCrossAttn(
                dim=encoder_dim, depth=int(encoder_depth / 2), num_heads=num_heads
            )

    def forward(
        self, x_sar: Tensor | None = None, x_optical: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Forward pass of the CROMA model.

        Args:
            x_sar: Input mini-batch of SAR images [B, 2, H, W].
            x_optical: Input mini-batch of optical images [B, 12, H, W].
        """
        return_dict: dict[str, Tensor] = {}

        if 'sar' in self.modalities and x_sar is not None:
            sar_encodings = self.s1_encoder(imgs=x_sar, attn_bias=self.attn_bias)
            sar_GAP = self.s1_GAP_FFN(sar_encodings.mean(dim=1))
            return_dict['sar_encodings'] = sar_encodings
            return_dict['sar_GAP'] = sar_GAP

        if 'optical' in self.modalities and x_optical is not None:
            optical_encodings = self.s2_encoder(
                imgs=x_optical, attn_bias=self.attn_bias
            )
            optical_GAP = self.s2_GAP_FFN(optical_encodings.mean(dim=1))
            return_dict['optical_encodings'] = optical_encodings
            return_dict['optical_GAP'] = optical_GAP

        if set(self.modalities) == {'sar', 'optical'}:
            joint_encodings = self.joint_encoder(
                x=sar_encodings,
                context=optical_encodings,
                relative_position_bias=self.attn_bias,
            )
            joint_GAP = joint_encodings.mean(dim=1)
            return_dict['joint_encodings'] = joint_encodings
            return_dict['joint_GAP'] = joint_GAP

        return return_dict


def get_2dalibi(num_heads: int, num_patches: int) -> Tensor:
    """Get 2D relative position bias for the attention layer.

    Args:
        num_heads: Number of heads for the multi-head attention.
        num_patches: Number of patches.

    Returns:
        2D relative position bias tensor.
    """
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(
        itertools.product(
            range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))
        )
    )

    def get_slopes(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    """Feed-forward network for the transformer."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        """Initialize the feed-forward network.

        Args:
            dim: Dimension of the input.
            mult: Multiplier for the inner dimension of the feed-forward network.
            dropout: Dropout probability
        """
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the feed-forward network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.input_norm(x)
        x = self.net(x)
        return x


class Attention(nn.Module):
    """Multi-head attention layer for the transformer."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        """Initialize the multi-head attention layer.

        Args:
            dim: Dimension of the input.
            num_heads: Number of heads for the multi-head attention.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, relative_position_bias: Tensor) -> Tensor:
        """Forward pass of the multi-head attention layer.

        Args:
            x: Input tensor.
            relative_position_bias: Relative position bias tensor.

        Returns:
            Output tensor.
        """
        x = self.input_norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v)
        )

        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention_scores = attention_scores + relative_position_bias

        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    """Cross-attention layer for the transformer."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        """Initialize the cross-attention layer.

        Args:
            dim: Dimension of the input.
            num_heads: Number of heads for the multi-head attention.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, context: Tensor, relative_position_bias: Tensor
    ) -> Tensor:
        """Forward pass of the cross-attention layer.

        Args:
            x: Input tensor.
            context: Context tensor.
            relative_position_bias: Relative position bias tensor.

        Returns:
            Output tensor.
        """
        x = self.input_norm(x)
        context = self.input_norm(context)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v)
        )

        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention_scores = attention_scores + relative_position_bias

        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class BaseTransformer(nn.Module):
    """Base transformer model."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        final_norm: bool = True,
    ) -> None:
        """Initialize the base transformer model.

        Args:
            dim: Dimension of the input.
            depth: Depth of the transformer.
            num_heads: Number of heads for the multi-head attention.
            attn_dropout: Dropout probability for the attention layer.
            ff_dropout: Dropout probability for the feed-forward network.
            ff_mult: Multiplier for the inner dimension of the feed-forward network.
            final_norm: Whether to apply a final layer normalization.
        """
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: Tensor, relative_position_bias: Tensor) -> Tensor:
        """Forward pass of the base transformer model.

        Args:
            x: Input tensor.
            relative_position_bias: whether to use relative position bias.
        """
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x
            x = ffn(x) + x

        x = self.norm_out(x) if self.final_norm else x
        return x


class BaseTransformerCrossAttn(nn.Module):
    """Base transformer model with cross-attention."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        """Initialize the base transformer model with cross-attention.

        Args:
            dim: Dimension of the input.
            depth: Depth of the transformer.
            num_heads: Number of heads for the multi-head attention.
            attn_dropout: Dropout probability for the attention layer.
            ff_dropout: Dropout probability for the feed-forward network.
            ff_mult: Multiplier for the inner dimension of the feed-forward network.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        CrossAttention(
                            dim=dim, num_heads=num_heads, dropout=attn_dropout
                        ),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self, x: Tensor, context: Tensor, relative_position_bias: Tensor
    ) -> Tensor:
        """Forward pass of the base transformer model with cross-attention.

        Args:
            x: Input tensor.
            context: Context tensor.
            relative_position_bias: Relative position bias tensor.

        Returns:
            Output tensor.
        """
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x
            x = cross_attn(x, context, relative_position_bias) + x
            x = ffn(x) + x

        x = self.norm_out(x)
        return x


class ViT(nn.Module):
    """Vision Transformer model."""

    def __init__(self, dim: int, depth: int, in_channels: int) -> None:
        """Initialize the vision transformer model.

        Args:
            dim: Dimension of the input.
            depth: Depth of the transformer.
            in_channels: Number of input channels.
        """
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim, depth=self.depth, num_heads=self.num_heads
        )

    def forward(self, imgs: Tensor, attn_bias: Tensor) -> Tensor:
        """Forward pass of the vision transformer model.

        Args:
            imgs: Input tensor.
            attn_bias: Relative position bias tensor.

        Returns:
            Output tensor.
        """
        x = rearrange(
            imgs,
            'b c (h i) (w j) -> b (h w) (c i j)',
            i=self.patch_size,
            j=self.patch_size,
        )
        # x is shape -> (bsz, num_patches, self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)
        x = self.transformer(x, relative_position_bias=attn_bias)
        return x


class CROMABase_Weights(WeightsEnum):  # type: ignore[misc]
    """CROMA base model weights.

    .. versionadded:: 0.7
    """

    CROMA_VIT = Weights(
        url='https://hf.co/antofuller/CROMA/resolve/0dd28e3d633bd6715856ae9890e8c49360040598/CROMA_base.pt',
        transforms=None,
        meta={
            'dataset': 'SSL4EO',
            'model': 'vit',
            'publication': 'https://arxiv.org/abs/2311.00566',
            'repo': 'https://github.com/antofuller/CROMA',
            'ssl_method': 'croma',
        },
    )


class CROMALarge_Weights(WeightsEnum):  # type: ignore[misc]
    """CROMA large model weights.

    .. versionadded:: 0.7
    """

    CROMA_VIT = Weights(
        url='https://huggingface.co/antofuller/CROMA/resolve/0dd28e3d633bd6715856ae9890e8c49360040598/CROMA_large.pt',
        transforms=None,
        meta={
            'dataset': 'SSL4EO',
            'model': 'vit',
            'publication': 'https://arxiv.org/abs/2311.00566',
            'repo': 'https://github.com/antofuller/CROMA',
            'ssl_method': 'croma',
        },
    )


def load_weights(model: CROMA, weights: WeightsEnum) -> None:
    """Load weights from a WeightsEnum object.

    Args:
        model: Model to load the weights into.
        weights: Weights to load.

    Raises:
        AssertionError: If there are missing or unexpected keys.
    """
    state_dict = weights.get_state_dict(progress=True)
    missing_keys, unexpected_keys = [], []

    if 'sar' in model.modalities:
        miss_key, unexp_key = model.s1_encoder.load_state_dict(
            state_dict['s1_encoder'], strict=False
        )
        missing_keys.extend(miss_key)
        unexpected_keys.extend(unexp_key)
        miss_key, unexp_key = model.s1_GAP_FFN.load_state_dict(
            state_dict['s1_GAP_FFN'], strict=False
        )
        missing_keys.extend(miss_key)
        unexpected_keys.extend(unexp_key)

    if 'optical' in model.modalities:
        miss_key, unexp_key = model.s2_encoder.load_state_dict(
            state_dict['s2_encoder'], strict=False
        )
        missing_keys.extend(miss_key)
        unexpected_keys.extend(unexp_key)
        miss_key, unexp_key = model.s2_GAP_FFN.load_state_dict(
            state_dict['s2_GAP_FFN'], strict=False
        )
        missing_keys.extend(miss_key)
        unexpected_keys.extend(unexp_key)

    if set(model.modalities) == {'sar', 'optical'}:
        miss_key, unexp_key = model.joint_encoder.load_state_dict(
            state_dict['joint_encoder'], strict=False
        )
        missing_keys.extend(miss_key)
        unexpected_keys.extend(unexp_key)

    assert not missing_keys, f'Missing keys: {missing_keys}'
    assert not unexpected_keys, f'Unexpected keys: {unexpected_keys}'


def croma_base(
    weights: CROMABase_Weights | None = None, *args: Any, **kwargs: Any
) -> CROMA:
    """CROMA base model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.00566

    .. versionadded:: 0.7

    Args:
        weights: Pretrained weights to load.
        *args: Additional arguments to pass to :class:CROMA.`
        **kwargs: Additional keyword arguments to pass to :class:CROMA.`

    Returns:
        CROMA base model.
    """
    kwargs |= {
        'encoder_dim': 768,
        'encoder_depth': 12,
        'num_heads': 16,
        'patch_size': 8,
    }
    model = CROMA(*args, **kwargs)
    if weights:
        load_weights(model, weights)
    return model


def croma_large(
    weights: CROMALarge_Weights | None = None, *args: Any, **kwargs: Any
) -> CROMA:
    """CROMA large model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.00566

    .. versionadded:: 0.7

    Args:
        weights: Pretrained weights to load.
        *args: Additional arguments to pass to :class:CROMA.`
        **kwargs: Additional keyword arguments to pass to :class:CROMA.`

    Returns:
        CROMA large model.
    """
    kwargs |= {
        'encoder_dim': 1024,
        'encoder_depth': 24,
        'num_heads': 16,
        'patch_size': 8,
    }
    model = CROMA(*args, **kwargs)
    if weights:
        load_weights(model, weights)
    return model
