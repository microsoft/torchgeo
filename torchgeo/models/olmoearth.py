# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained OlmoEarth v1 models."""

from typing import Any

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

_olmoearth_transforms = nn.Identity()

_olmoearth_meta = {
    'dataset': 'Major TOM',
    'model': 'OlmoEarthPretrain_v1',
    'architecture': 'Vision Transformer',
    'publication': 'https://arxiv.org/abs/2506.10890',
    'repo': 'https://github.com/allenai/olmoearth_pretrain',
    'license': 'OlmoEarth Artifact License',
    'model_size': None,
    'hf_repo': None,
}


class OlmoEarthV1_Weights(WeightsEnum):
    """OlmoEarth v1 pre-trained weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    .. versionadded:: 0.10
    """

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/c48459cd6264704b9d1761a2904c46eb98755fda/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'nano', 'hf_repo': 'allenai/OlmoEarth-v1-Nano'},
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/edd9418badc5a9f769ba1aa622cb6d0af4586f8b/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'tiny', 'hf_repo': 'allenai/OlmoEarth-v1-Tiny'},
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/93589e2dee5b5c95a660d1e9365bc017ea7f35d6/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'base', 'hf_repo': 'allenai/OlmoEarth-v1-Base'},
    )
    LARGE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/8cf072c70d4a1c403531ca9a9653bb1f8f60eb83/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'large', 'hf_repo': 'allenai/OlmoEarth-v1-Large'},
    )


def olmoearth_v1(
    weights: OlmoEarthV1_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth v1 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.10

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``model_size``, ``max_patch_size``).

    Returns:
        An OlmoEarth v1 model.
    """
    olmoearth = lazy_import('olmoearth_pretrain_minimal')

    model_size = kwargs.pop('model_size', 'nano')
    if weights is not None:
        model_size = weights.meta.get('model_size', model_size)
    model: nn.Module = olmoearth.OlmoEarthPretrain_v1(model_size=model_size, **kwargs)
    if weights is not None:
        state_dict = weights.get_state_dict(progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def olmoearth_v1_unet_decoder(
    in_dim: int = 768,
    num_classes: int = 1,
    patch_size: int = 16,
    conv_layers_per_resolution: int = 1,
    **kwargs: Any,
) -> nn.Module:
    """UNet-style decoder head for OlmoEarth v1 features.

    A progressive upsampling decoder that turns OlmoEarth ViT patch tokens of
    shape ``(B, H_p, W_p, in_dim)`` into per-pixel logits of shape
    ``(B, num_classes, H, W)`` where ``H = H_p * patch_size``, for segmentation
    or regression on top of a frozen or fine-tuned backbone.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to build the decoder.

    .. versionadded:: 0.10

    Args:
        in_dim: Number of input feature channels, i.e. the embedding dimension
            of the OlmoEarth backbone that produces the patch tokens.
        num_classes: Number of output channels (segmentation classes or
            regression targets).
        patch_size: Backbone patch size. The decoder performs
            ``log2(patch_size)`` upsampling stages, so this must be a power of
            two (4, 8, 16, ...).
        conv_layers_per_resolution: Number of 3x3 conv + ReLU blocks applied at
            each upsampling resolution.
        **kwargs: Additional keyword arguments passed to
            ``olmoearth_pretrain_minimal.UNetDecoder``.

    Returns:
        A UNet-style decoder head.

    Raises:
        ValueError: If *patch_size* is not a power of two.
    """
    olmoearth = lazy_import('olmoearth_pretrain_minimal')
    decoder: nn.Module = olmoearth.UNetDecoder(
        in_dim=in_dim,
        num_classes=num_classes,
        patch_size=patch_size,
        conv_layers_per_resolution=conv_layers_per_resolution,
        **kwargs,
    )
    return decoder
