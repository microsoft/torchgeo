# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained OlmoEarth v1 models."""

import os
from typing import Any

import torch
from torch import nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

_olmoearth_transforms = nn.Identity()

# Artifacts every OlmoEarth repo publishes. The config records the architecture, so it is
# pinned and verified alongside the weights.
_CONFIG_FILENAME = 'config.json'
_WEIGHTS_FILENAME = 'weights.pth'

_olmoearth_meta = {
    'dataset': 'Major TOM',
    'model': 'OlmoEarthPretrain_v1',
    'architecture': 'Vision Transformer',
    'publication': 'https://arxiv.org/abs/2506.10890',
    'repo': 'https://github.com/allenai/olmoearth_pretrain',
    'license': 'OlmoEarth Artifact License',
    'model_size': None,
    'hf_repo': None,
    'revision': None,
    'config_sha256': None,
    'weights_sha256': None,
}


class OlmoEarthV1_Weights(WeightsEnum):
    """OlmoEarth v1 pre-trained weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    .. versionadded:: 0.10
    """

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/529248a4dc3c54014c56b7504641cec98de31d1c/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'nano',
            'hf_repo': 'allenai/OlmoEarth-v1-Nano',
            'revision': '529248a4dc3c54014c56b7504641cec98de31d1c',
            'config_sha256': '088f5314909cfcfd75c2ba4b07c9f88f5ca919d88235b702467046f5fab3a35a',
            'weights_sha256': '795c68419a658fd22ccf8f2e020607675f963e9ef3b93d8e368bb17646765347',
        },
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/885784437d4e2d632b7bf51b4233426c6f4479dc/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'tiny',
            'hf_repo': 'allenai/OlmoEarth-v1-Tiny',
            'revision': '885784437d4e2d632b7bf51b4233426c6f4479dc',
            'config_sha256': '975dc2d755688c3010f9fecb662b62784d335731d701d8dda3f9ea0ccec88781',
            'weights_sha256': '66b9827af383bc444d7909a406a5b62c072bb08d6804ff47a247c2dce8fad9a4',
        },
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/4bd1392a4539404d2c74276c39f3cb4cfff466cc/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'base',
            'hf_repo': 'allenai/OlmoEarth-v1-Base',
            'revision': '4bd1392a4539404d2c74276c39f3cb4cfff466cc',
            'config_sha256': 'bd7759b9185f3d51607ca8d554c9be6ca87d932bc46fbc0f75524b0c3d5512bf',
            'weights_sha256': '551c1cc53337c6faaddead88071d7ebd2bd53ec271600fa6f0ee0a518c8b6e11',
        },
    )
    LARGE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/b2c9f41de3d8454cb37f0cd9cc3e79ec7c4af435/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'large',
            'hf_repo': 'allenai/OlmoEarth-v1-Large',
            'revision': 'b2c9f41de3d8454cb37f0cd9cc3e79ec7c4af435',
            'config_sha256': 'bd5f0fe3f571cf8beed64072d6c0029d6072223e6dce0b84b689c34d6638bbf1',
            'weights_sha256': '1adb5026bd520c54bc415a1282386954927623bab81d01be2f5b6379cc039035',
        },
    )


class OlmoEarthV1_1_Weights(WeightsEnum):
    """OlmoEarth v1.1 pre-trained weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/html/2605.20804v1

    .. versionadded:: 0.11
    """

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_1-Nano/resolve/6c16c7da0d05a1c4f32c2a7f9233e07c9ebfa61a/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'nano',
            'hf_repo': 'allenai/OlmoEarth-v1_1-Nano',
            'revision': '6c16c7da0d05a1c4f32c2a7f9233e07c9ebfa61a',
            'config_sha256': '7a7cbd2b1b06e500869e9bc00293ec0714a50da92676dc8ac9b1cf026da470bd',
            'weights_sha256': '883561154dd4eb874a3e28bb0559c15aa06bd8a2e4ffb54a0d6bc4bc15777eb3',
        },
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_1-Tiny/resolve/74fab5714f763d6b94f8b1536bdd3300d77f45e8/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'tiny',
            'hf_repo': 'allenai/OlmoEarth-v1_1-Tiny',
            'revision': '74fab5714f763d6b94f8b1536bdd3300d77f45e8',
            'config_sha256': '01dcb438144d8f70647ab2d11aef656a1632f3b5af1fdf9263c111127ad7bbc3',
            'weights_sha256': '2a3fe8132adf9ff2ca96d00c9e376b8925bfe430fda6140749b3b92764c67ae1',
        },
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_1-Base/resolve/4ef31d45f80c1d4fcce18f9cde40c1b5e4d96cf4/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'base',
            'hf_repo': 'allenai/OlmoEarth-v1_1-Base',
            'revision': '4ef31d45f80c1d4fcce18f9cde40c1b5e4d96cf4',
            'config_sha256': 'c74db2ea8c80b7568da926826e4f6cbca7e8cd1a2d0af45568f94af82722d336',
            'weights_sha256': '37fc73d542618f28357583fd17307002a4bdfb5321142c08e928102341105989',
        },
    )


class OlmoEarthV1_2_Weights(WeightsEnum):
    """OlmoEarth v1.2 pre-trained weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2605.20804

    .. versionadded:: 0.11
    """

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/resolve/e1f693ae2a7d5b57871a978e9d09e22d05206747/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'nano',
            'hf_repo': 'allenai/OlmoEarth-v1_2-Nano',
            'revision': 'e1f693ae2a7d5b57871a978e9d09e22d05206747',
            'config_sha256': '4cd2888e79dc543f262cc3d86fcd30d667068fd53a728ca5bd306d5ddb509d1d',
            'weights_sha256': '2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378',
        },
    )
    SMALL = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Small/resolve/a207c9a789483f95de1e9fb06acadb3da3775863/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'small',
            'hf_repo': 'allenai/OlmoEarth-v1_2-Small',
            'revision': 'a207c9a789483f95de1e9fb06acadb3da3775863',
            'config_sha256': '254703d9b5da4a6679003ff21f2da964a25d903fea70dc0b2cce5d0cd388f70b',
            'weights_sha256': '459796ed5680bc85926f9a0e023476d14cb637bc19f826575c43836c909a5fa6',
        },
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Tiny/resolve/12a9fdbfeff905d7e147e7497f9f7a95c518eefc/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'tiny',
            'hf_repo': 'allenai/OlmoEarth-v1_2-Tiny',
            'revision': '12a9fdbfeff905d7e147e7497f9f7a95c518eefc',
            'config_sha256': 'bb11f91f5afbd6138f75feee3f66fc0e272da089d05a6e515713c799057155ac',
            'weights_sha256': '835c0b21ab010c4c4515faafa44dc1a41c9bc512d3a30af184803c4f4257697d',
        },
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Base/resolve/581aa9baaa7aed4348c0903617eb92ee9f89e2ec/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {
            'model_size': 'base',
            'hf_repo': 'allenai/OlmoEarth-v1_2-Base',
            'revision': '581aa9baaa7aed4348c0903617eb92ee9f89e2ec',
            'config_sha256': '0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a',
            'weights_sha256': '57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec',
        },
    )


def _download_pinned(weights: WeightsEnum) -> str:
    """Download a checkpoint's config and weights into one directory.

    Both files are pinned to a commit and checked against the sha256 recorded in *weights*,
    so neither the architecture nor the weights can change under a pinned release.

    Args:
        weights: Pre-trained weights to download.

    Returns:
        Directory holding the downloaded ``config.json`` and ``weights.pth``.
    """
    repo = weights.meta['hf_repo']
    revision = weights.meta['revision']
    directory = os.path.join(torch.hub.get_dir(), 'checkpoints', repo, revision)
    os.makedirs(directory, exist_ok=True)

    for filename, key in (
        (_CONFIG_FILENAME, 'config_sha256'),
        (_WEIGHTS_FILENAME, 'weights_sha256'),
    ):
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
            torch.hub.download_url_to_file(url, path, hash_prefix=weights.meta[key])
    return directory


def _olmoearth(
    weights: WeightsEnum | None, model_version: str, **kwargs: Any
) -> nn.Module:
    """Build an OlmoEarth model, optionally loading pre-trained weights.

    Pre-trained models are built from the checkpoint's own config rather than from
    *model_version*, since that is what records the architecture and which modalities the
    encoder was trained to ingest.

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        model_version: Architecture version to build when randomly initializing.
        **kwargs: Passed to ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``.

    Returns:
        An OlmoEarth model.
    """
    olmoearth = lazy_import('olmoearth_pretrain_minimal')

    if weights is None:
        model_size = kwargs.pop('model_size', 'nano')
        random: nn.Module = olmoearth.OlmoEarthPretrain_v1(
            model_size=model_size, model_version=model_version, **kwargs
        )
        return random

    pretrained: nn.Module = olmoearth.load_model_from_path(_download_pinned(weights))
    return pretrained


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
    return _olmoearth(weights, 'v1', **kwargs)


def olmoearth_v1_1(
    weights: OlmoEarthV1_1_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth v1.1 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/html/2605.20804v1

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.11

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``model_size``, ``max_patch_size``). Ignored when *weights* is given,
            since the architecture then comes from the checkpoint's own config.

    Returns:
        An OlmoEarth v1.1 model.
    """
    return _olmoearth(weights, 'v1.1', **kwargs)


def olmoearth_v1_2(
    weights: OlmoEarthV1_2_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth v1.2 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2605.20804

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.11

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``model_size``, ``max_patch_size``). Ignored when *weights* is given,
            since the architecture then comes from the checkpoint's own config.

    Returns:
        An OlmoEarth v1.2 model.
    """
    return _olmoearth(weights, 'v1.2', **kwargs)


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
