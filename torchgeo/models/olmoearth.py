# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained OlmoEarth v1 models."""

import hashlib
import os
from typing import Any

from torch import nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

_olmoearth_transforms = nn.Identity()

# Artifacts every OlmoEarth repo publishes. The config records the architecture, so it is
# pinned and verified alongside the weights.
_CONFIG_FILENAME = 'config.json'
_WEIGHTS_FILENAME = 'weights.pth'
_SHA256_LENGTH = 64

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
        },
    )


class OlmoEarthV1_1_Weights(WeightsEnum):
    """OlmoEarth v1.1 pre-trained weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

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
        },
    )


def _verified_download(repo: str, filename: str, revision: str) -> str:
    """Download one artifact at a pinned revision and check it against the Hub's digest.

    ``hf_hub_download`` only checks the downloaded file's size, so a truncated or corrupted
    file of the right length is accepted. The Hub's ETag carries a digest of the contents:
    the sha256 for a git-LFS file, the git blob sha1 for a regular git file.

    Args:
        repo: Hugging Face repo ID.
        filename: File to download from the repo.
        revision: Git revision to pin to, e.g. a commit hash.

    Returns:
        Local path to the downloaded file.

    Raises:
        RuntimeError: If the Hub reports no digest, or the contents do not match it.
    """
    hub = lazy_import('huggingface_hub')
    path = hub.hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    url = hub.hf_hub_url(repo_id=repo, filename=filename, revision=revision)
    etag = hub.get_hf_file_metadata(url).etag
    if etag is None:
        raise RuntimeError(
            f'{repo}/{filename} has no ETag to verify its contents against'
        )

    with open(path, 'rb') as f:
        if len(etag) == _SHA256_LENGTH:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        else:
            # git names a blob by hashing 'blob <size>\0' followed by the contents
            header = b'blob %d\0' % os.path.getsize(path)
            digest = hashlib.file_digest(f, lambda: hashlib.sha1(header)).hexdigest()

    if digest != etag:
        raise RuntimeError(
            f'{repo}/{filename} failed its integrity check: expected {etag}, got {digest}'
        )
    return path


def _olmoearth(
    weights: WeightsEnum | None, model_version: str, **kwargs: Any
) -> nn.Module:
    """Build an OlmoEarth model, optionally loading pre-trained weights.

    Pre-trained models are built from the checkpoint's own config rather than from
    *model_version*, since that is what records the architecture and which modalities the
    encoder was trained to ingest. Both artifacts are pinned to a commit and verified.

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

    repo = weights.meta['hf_repo']
    revision = weights.meta['revision']
    for filename in (_CONFIG_FILENAME, _WEIGHTS_FILENAME):
        path = _verified_download(repo, filename, revision)
    pretrained: nn.Module = olmoearth.load_model_from_path(os.path.dirname(path))
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

    * https://arxiv.org/abs/2511.13655

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
