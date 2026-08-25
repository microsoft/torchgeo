# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained OlmoEarth v1 models."""

from typing import Any

from torch import nn
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
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/529248a4dc3c54014c56b7504641cec98de31d1c/weights-795c68419a658fd22ccf8f2e020607675f963e9ef3b93d8e368bb17646765347.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'nano', 'hf_repo': 'allenai/OlmoEarth-v1-Nano'},
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/885784437d4e2d632b7bf51b4233426c6f4479dc/weights-66b9827af383bc444d7909a406a5b62c072bb08d6804ff47a247c2dce8fad9a4.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'tiny', 'hf_repo': 'allenai/OlmoEarth-v1-Tiny'},
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/4bd1392a4539404d2c74276c39f3cb4cfff466cc/weights-551c1cc53337c6faaddead88071d7ebd2bd53ec271600fa6f0ee0a518c8b6e11.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'base', 'hf_repo': 'allenai/OlmoEarth-v1-Base'},
    )
    LARGE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/b2c9f41de3d8454cb37f0cd9cc3e79ec7c4af435/weights-1adb5026bd520c54bc415a1282386954927623bab81d01be2f5b6379cc039035.pth',
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
    model: nn.Module = olmoearth.OlmoEarthPretrain_v1(
        model_size=model_size, model_version='v1', **kwargs
    )
    if weights is not None:
        state_dict = weights.get_state_dict(
            progress=True, check_hash=True, weights_only=True
        )
        # The checkpoints are keyed encoder.*/decoder.*, but OlmoEarthPretrain_v1 holds the
        # network in self.model, so its parameters are model.encoder.* etc. Without re-keying
        # the two name sets are disjoint and strict=False silently drops every tensor, leaving
        # the returned model randomly initialized.
        state_dict = {f'model.{key}': value for key, value in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        assert not missing_keys
        assert not unexpected_keys
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

    .. versionadded:: 0.11

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
