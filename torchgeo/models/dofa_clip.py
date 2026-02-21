# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""DOFA-CLIP model."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

from .dofa import DOFA


class TextTransformer(nn.Module):
    """Transformer text encoder used in DOFA-CLIP."""

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        layers: int = 6,
        heads: int = 8,
        output_dim: int = 512,
    ) -> None:
        """Initialize a new text encoder.

        Args:
            context_length: Maximum token sequence length.
            vocab_size: Token vocabulary size.
            width: Transformer hidden width.
            layers: Number of transformer layers.
            heads: Number of attention heads.
            output_dim: Output embedding dimension.
        """
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=0.02)

    def forward(self, text: Tensor) -> Tensor:
        """Encode tokenized text.

        Args:
            text: Input token IDs of shape ``(batch_size, context_length)``.

        Returns:
            Encoded text embeddings of shape ``(batch_size, output_dim)``.
        """
        x = self.token_embedding(text)
        x = x + self.positional_embedding.unsqueeze(0).to(x.dtype)
        x = self.transformer(x)
        x = self.ln_final(x)
        pooled = x[:, 0, :]
        return torch.matmul(pooled, self.text_projection)


class DOFA_CLIP(nn.Module):
    """DOFA-CLIP model.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        embed_dim: int = 512,
        image_embed_dim: int = 768,
        image_depth: int = 12,
        image_num_heads: int = 12,
        context_length: int = 77,
        vocab_size: int = 49408,
        text_width: int = 512,
        text_layers: int = 6,
        text_heads: int = 8,
    ) -> None:
        """Initialize a DOFA-CLIP model.

        Args:
            embed_dim: Shared embedding dimension.
            image_embed_dim: Image encoder embedding dimension.
            image_depth: Number of image transformer layers.
            image_num_heads: Number of image transformer attention heads.
            context_length: Maximum text token length.
            vocab_size: Text vocabulary size.
            text_width: Text encoder width.
            text_layers: Number of text transformer layers.
            text_heads: Number of text transformer attention heads.
        """
        super().__init__()
        self.image_encoder = DOFA(
            embed_dim=image_embed_dim,
            depth=image_depth,
            num_heads=image_num_heads,
            num_classes=0,
            global_pool=True,
        )
        self.image_projection = nn.Linear(image_embed_dim, embed_dim, bias=False)
        self.text_encoder = TextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            output_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_image(self, image: Tensor, wavelengths: list[float]) -> Tensor:
        """Encode images.

        Args:
            image: Input image tensor with shape ``(batch_size, channels, height, width)``.
            wavelengths: Spectral wavelengths in micrometers for each image channel.

        Returns:
            L2-normalized image embeddings.
        """
        features = self.image_encoder.forward_features(image, wavelengths)
        features = self.image_projection(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, text: Tensor) -> Tensor:
        """Encode text tokens.

        Args:
            text: Input token IDs with shape ``(batch_size, context_length)``.

        Returns:
            L2-normalized text embeddings.
        """
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1)

    def forward(
        self, image: Tensor, text: Tensor, wavelengths: list[float]
    ) -> tuple[Tensor, Tensor]:
        """Compute CLIP logits.

        Args:
            image: Input image tensor.
            text: Input text token IDs.
            wavelengths: Image band wavelengths.

        Returns:
            Tuple of image-to-text and text-to-image logits.
        """
        image_features = self.encode_image(image, wavelengths)
        text_features = self.encode_text(text)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


_dofa_clip_transforms = nn.Sequential(T.CenterCrop((224, 224)))


class DOFA_CLIP_Base16_Weights(WeightsEnum):  # type: ignore[misc]
    """DOFA-CLIP base patch size 16 weights.

    .. versionadded:: 0.9
    """

    DOFA_CLIP = Weights(
        url='https://huggingface.co/xiong-zhitong/DOFA-CLIP/resolve/main/DOFA_CLIP_BASE.pth',
        transforms=_dofa_clip_transforms,
        meta={
            'dataset': 'RS5M',
            'model': 'dofa_clip_base_patch16_224',
            'publication': 'https://arxiv.org/abs/2503.23447',
            'repo': 'https://github.com/xiong-zhitong/DOFA-CLIP',
            'ssl_method': 'clip',
        },
    )


def _convert_state_dict(
    state_dict: OrderedDict[str, Tensor],
) -> OrderedDict[str, Tensor]:
    """Normalize checkpoint key names to match TorchGeo module names.

    Args:
        state_dict: Checkpoint state dict.

    Returns:
        Converted state dict.
    """
    converted_state_dict: OrderedDict[str, Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('state_dict.'):
            new_key = new_key.removeprefix('state_dict.')
        if new_key.startswith('module.'):
            new_key = new_key.removeprefix('module.')
        if new_key.startswith('model.'):
            new_key = new_key.removeprefix('model.')
        converted_state_dict[new_key] = value
    return converted_state_dict


def dofa_clip_base_patch16_224(
    weights: DOFA_CLIP_Base16_Weights | None = None,
    embed_dim: int = 512,
    image_embed_dim: int = 768,
    image_depth: int = 12,
    image_num_heads: int = 12,
    context_length: int = 77,
    vocab_size: int = 49408,
    text_width: int = 512,
    text_layers: int = 6,
    text_heads: int = 8,
) -> DOFA_CLIP:
    """DOFA-CLIP base patch size 16 model.

    Args:
        weights: Pre-trained model weights to use.
        embed_dim: Shared embedding dimension.
        image_embed_dim: Image encoder embedding dimension.
        image_depth: Number of image transformer layers.
        image_num_heads: Number of image transformer attention heads.
        context_length: Maximum text token length.
        vocab_size: Text vocabulary size.
        text_width: Text encoder width.
        text_layers: Number of text transformer layers.
        text_heads: Number of text transformer attention heads.

    Returns:
        A DOFA-CLIP model.
    """
    model = DOFA_CLIP(
        embed_dim=embed_dim,
        image_embed_dim=image_embed_dim,
        image_depth=image_depth,
        image_num_heads=image_num_heads,
        context_length=context_length,
        vocab_size=vocab_size,
        text_width=text_width,
        text_layers=text_layers,
        text_heads=text_heads,
    )

    if weights:
        checkpoint = weights.get_state_dict(progress=True)
        if 'state_dict' in checkpoint and isinstance(
            checkpoint['state_dict'], OrderedDict
        ):
            checkpoint = checkpoint['state_dict']
        state_dict = _convert_state_dict(checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert not missing_keys
        assert not unexpected_keys

    return model
