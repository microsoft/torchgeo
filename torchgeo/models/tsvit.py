# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Time-Series Vision Transformer (TSViT) model."""

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Block
from torch import Tensor, nn


class TSViT(nn.Module):
    """Time-Series Vision Transformer (TSViT).

    This model processes Satellite Image Time Series (SITS) for dense
    semantic-segmentation prediction.

    The implementation follows the published TSViT architecture and is
    designed to be compatible with the released PASTIS24 TSViT checkpoints.

    If you use this model in your research, please cite:

        Tarasiou, M., Chavez, E., & Zafeiriou, S.
        "ViTs for SITS: Vision Transformers for Satellite Image Time Series."
        CVPR 2023.
        https://arxiv.org/abs/2301.04944

    .. versionadded:: 0.11

    Args:
        img_res: Spatial resolution of the input image.
        patch_size: Spatial size of each patch.
        num_channels: Number of input channels, including the final acquisition-date channel.
        num_classes: Number of output segmentation classes.
        max_seq_len: Maximum number of temporal observations.
        dim: Transformer embedding dimension.
        temporal_depth: Number of temporal Transformer blocks.
        spatial_depth: Number of spatial Transformer blocks.
        heads: Number of attention heads.
        dim_head: Dimension of each attention head.
        dropout: Dropout applied inside Transformer blocks.
        emb_dropout: Dropout applied before the spatial Transformer.
        scale_dim: MLP expansion factor inside Transformer blocks.
    """

    def __init__(
        self,
        img_res: int = 24,
        patch_size: int = 2,
        num_channels: int = 11,
        num_classes: int = 19,
        max_seq_len: int = 60,
        dim: int = 128,
        temporal_depth: int = 4,
        spatial_depth: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        scale_dim: int = 4,
    ) -> None:
        """Initialize the TSViT model."""
        super().__init__()

        if img_res % patch_size != 0:
            raise ValueError('Image dimensions must be divisible by patch size.')
        if dim != heads * dim_head:
            raise ValueError(
                'dim must equal heads * dim_head for the timm attention block.'
            )

        self.image_size = img_res
        self.patch_size = patch_size
        self.num_patches_1d = img_res // patch_size
        self.num_patches = self.num_patches_1d**2
        self.num_classes = num_classes
        self.num_frames = max_seq_len
        self.num_channels = num_channels
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.temporal_depth = temporal_depth
        self.spatial_depth = spatial_depth
        self.scale_dim = scale_dim

        # The original TSViT treats the last channel as the acquisition date.
        patch_dim = (num_channels - 1) * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)',
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )

        # Day-of-year lookup represented as a learned linear embedding.
        self.to_temporal_embedding_input = nn.Linear(366, dim)

        # One learned class token per output class.
        self.temporal_token = nn.Parameter(torch.randn(1, num_classes, dim))

        # timm Block matches the pre-norm Transformer structure:
        # norm -> attention -> residual, then norm -> MLP -> residual.
        #
        # norm_layer=nn.LayerNorm is explicit so the normalization behavior
        # matches the original PyTorch LayerNorm used by TSViT.

        self.temporal_transformer = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=float(scale_dim),
                    qkv_bias=False,
                    qk_norm=False,
                    proj_bias=True,
                    proj_drop=dropout,
                    attn_drop=0.0,
                    init_values=None,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(temporal_depth)
            ],
            nn.LayerNorm(dim),
        )

        self.space_transformer = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=float(scale_dim),
                    qkv_bias=False,
                    qk_norm=False,
                    proj_bias=True,
                    proj_drop=dropout,
                    attn_drop=0.0,
                    init_values=None,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(spatial_depth)
            ],
            nn.LayerNorm(dim),
        )

        self.space_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_size**2))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor with shape ``(B, T, H, W, C)``.

        Returns:
            Segmentation logits with shape ``(B, num_classes, H, W)``.
        """
        if x.ndim != 5:
            raise ValueError('Expected input shape (B, T, H, W, C).')

        B, T, H, W, C = x.shape

        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f'Expected spatial size {self.image_size}x{self.image_size}, '
                f'got {H}x{W}.'
            )

        if C != self.num_channels:
            raise ValueError(f'Expected {self.num_channels} input channels, got {C}.')

        if T > self.num_frames:
            raise ValueError(
                f'Expected at most {self.num_frames} temporal frames, got {T}.'
            )

        # Move channels before height/width for patch extraction.
        x = x.permute(0, 1, 4, 2, 3)

        # The final channel stores the normalized acquisition date.
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]

        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)

        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(
            B, T, self.dim
        )

        # Convert every spatial patch into a Transformer embedding.
        x = self.to_patch_embedding(x)
        x = x.reshape(B, self.num_patches, T, self.dim)

        # Add the acquisition-time embedding to every spatial patch.
        x = x + temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(B * self.num_patches, T, self.dim)

        # Add one learned class token per output class.
        cls_temporal_tokens = repeat(
            self.temporal_token, '() N d -> b N d', b=B * self.num_patches
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # Temporal reasoning: each class token attends to the time series.
        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]

        # Rearrange so each class independently sees all spatial patches.
        x = x.reshape(B, self.num_patches, self.num_classes, self.dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * self.num_classes, self.num_patches, self.dim)

        # Spatial reasoning across the patch grid.
        x = x + self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)

        # Predict patch pixels, then reconstruct the full image.
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches, self.patch_size**2)

        x = rearrange(
            x,
            'b c (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)',
            h1=self.num_patches_1d,
            w1=self.num_patches_1d,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        return x


def convert_tsvit_checkpoint(
    state_dict: Mapping[str, Tensor], model: TSViT
) -> dict[str, Tensor]:
    """Convert the released TSViT checkpoint keys to TorchGeo/timm keys.

    The released checkpoint was trained with the original TSViT implementation.
    The model implementation here is independently written, so its parameter
    names follow the timm Transformer block naming convention.

    Args:
        state_dict: Original TSViT state dictionary.
        model: Instantiated TorchGeo TSViT model.

    Returns:
        State dictionary compatible with ``model.load_state_dict``.

    Raises:
        KeyError: If a required checkpoint parameter is missing.
        ValueError: If a converted tensor shape does not match the model.
    """
    converted: dict[str, Tensor] = {}

    direct_keys = {
        'temporal_token',
        'space_pos_embedding',
        'to_patch_embedding.1.weight',
        'to_patch_embedding.1.bias',
        'to_temporal_embedding_input.weight',
        'to_temporal_embedding_input.bias',
        'mlp_head.0.weight',
        'mlp_head.0.bias',
        'mlp_head.1.weight',
        'mlp_head.1.bias',
    }

    for key in direct_keys:
        if key not in state_dict:
            raise KeyError(f'Missing checkpoint key: {key}')
        converted[key] = state_dict[key]

    def map_transformer(old_prefix: str, new_prefix: str, depth: int) -> None:
        for i in range(depth):
            old = f'{old_prefix}.layers.{i}'
            new = f'{new_prefix}.{i}'

            mappings = {
                f'{old}.0.norm.weight': f'{new}.norm1.weight',
                f'{old}.0.norm.bias': f'{new}.norm1.bias',
                f'{old}.0.fn.to_qkv.weight': f'{new}.attn.qkv.weight',
                f'{old}.0.fn.to_out.0.weight': f'{new}.attn.proj.weight',
                f'{old}.0.fn.to_out.0.bias': f'{new}.attn.proj.bias',
                f'{old}.1.norm.weight': f'{new}.norm2.weight',
                f'{old}.1.norm.bias': f'{new}.norm2.bias',
                f'{old}.1.fn.net.0.weight': f'{new}.mlp.fc1.weight',
                f'{old}.1.fn.net.0.bias': f'{new}.mlp.fc1.bias',
                f'{old}.1.fn.net.3.weight': f'{new}.mlp.fc2.weight',
                f'{old}.1.fn.net.3.bias': f'{new}.mlp.fc2.bias',
            }

            for old_key, new_key in mappings.items():
                if old_key not in state_dict:
                    raise KeyError(f'Missing checkpoint key: {old_key}')
                converted[new_key] = state_dict[old_key]

        old_final = f'{old_prefix}.norm'
        new_final = f'{new_prefix}.{depth}'
        for suffix in ('weight', 'bias'):
            old_key = f'{old_final}.{suffix}'
            new_key = f'{new_final}.{suffix}'
            if old_key not in state_dict:
                raise KeyError(f'Missing checkpoint key: {old_key}')
            converted[new_key] = state_dict[old_key]

    map_transformer(
        'temporal_transformer', 'temporal_transformer', model.temporal_depth
    )
    map_transformer('space_transformer', 'space_transformer', model.spatial_depth)

    model_state = model.state_dict()
    for key, value in converted.items():
        if key not in model_state:
            raise ValueError(f'Converted checkpoint key not in model: {key}')
        if model_state[key].shape != value.shape:
            raise ValueError(
                f'Shape mismatch for {key}: '
                f'checkpoint={tuple(value.shape)}, '
                f'model={tuple(model_state[key].shape)}'
            )

    return converted


def load_tsvit_checkpoint(model: TSViT, checkpoint_path: str | Path) -> None:
    """Load a released TSViT checkpoint into a TorchGeo model.

    Args:
        model: Instantiated TorchGeo TSViT model.
        checkpoint_path: Path to a released ``best.pth`` checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    converted = convert_tsvit_checkpoint(checkpoint, model)
    model.load_state_dict(converted, strict=True)
