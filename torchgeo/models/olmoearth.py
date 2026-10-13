# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained OlmoEarth models."""

from typing import Any

from torch import nn
from torchvision.models._api import Weights, WeightsEnum
from typing_extensions import deprecated

from ..datasets.utils import lazy_import

_olmoearth_transforms = nn.Identity()

_olmoearth_meta = {
    'dataset': 'OlmoEarthPretrain',
    'model': 'OlmoEarthPretrain_v1',
    'architecture': 'Vision Transformer',
    'publication': 'https://arxiv.org/abs/2511.13655',
    'repo': 'https://github.com/allenai/olmoearth_pretrain',
    'license': 'OlmoEarth Artifact License',
    'model_version': None,
    'hf_repo': None,
    'model_size': None,
    'embed_dim': None,
    'depth': None,
    'num_heads': None,
    'patch_size': (1, 8),
    'max_sequence_length': 12,
    'modalities': ['sentinel2_l2a', 'sentinel1', 'landsat'],
    'bands': {
        'sentinel2_l2a': [
            'B02',
            'B03',
            'B04',
            'B08',
            'B05',
            'B06',
            'B07',
            'B8A',
            'B11',
            'B12',
            'B01',
            'B09',
        ],
        'sentinel1': ['vv', 'vh'],
        'landsat': ['B8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9', 'B10', 'B11'],
    },
}

# Architecture of each model size (identical across versions).
_olmoearth_sizes = {
    'nano': {'model_size': 'nano', 'embed_dim': 128, 'depth': 4, 'num_heads': 8},
    'tiny': {'model_size': 'tiny', 'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'model_size': 'small', 'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base': {'model_size': 'base', 'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'model_size': 'large', 'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
}


class OlmoEarthNano_Weights(WeightsEnum):
    """OlmoEarth Nano weights.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2511.13655
    * https://arxiv.org/abs/2605.20804v1
    * https://arxiv.org/abs/2605.20804

    .. versionadded:: 0.11
    """

    V1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/529248a4dc3c54014c56b7504641cec98de31d1c/weights-795c68419a658fd22ccf8f2e020607675f963e9ef3b93d8e368bb17646765347.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['nano']
        | {'model_version': 'v1', 'hf_repo': 'allenai/OlmoEarth-v1-Nano'},
    )
    V1_1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_1-Nano/resolve/6c16c7da0d05a1c4f32c2a7f9233e07c9ebfa61a/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['nano']
        | {'model_version': 'v1.1', 'hf_repo': 'allenai/OlmoEarth-v1_1-Nano'},
    )
    V1_2 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/resolve/e1f693ae2a7d5b57871a978e9d09e22d05206747/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['nano']
        | {'model_version': 'v1.2', 'hf_repo': 'allenai/OlmoEarth-v1_2-Nano'},
    )


class OlmoEarthTiny_Weights(WeightsEnum):
    """OlmoEarth Tiny weights.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2511.13655
    * https://arxiv.org/abs/2605.20804v1
    * https://arxiv.org/abs/2605.20804

    .. versionadded:: 0.11
    """

    V1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/885784437d4e2d632b7bf51b4233426c6f4479dc/weights-66b9827af383bc444d7909a406a5b62c072bb08d6804ff47a247c2dce8fad9a4.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['tiny']
        | {'model_version': 'v1', 'hf_repo': 'allenai/OlmoEarth-v1-Tiny'},
    )
    V1_1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_1-Tiny/resolve/74fab5714f763d6b94f8b1536bdd3300d77f45e8/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['tiny']
        | {'model_version': 'v1.1', 'hf_repo': 'allenai/OlmoEarth-v1_1-Tiny'},
    )
    V1_2 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Tiny/resolve/12a9fdbfeff905d7e147e7497f9f7a95c518eefc/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['tiny']
        | {'model_version': 'v1.2', 'hf_repo': 'allenai/OlmoEarth-v1_2-Tiny'},
    )


class OlmoEarthSmall_Weights(WeightsEnum):
    """OlmoEarth Small weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2605.20804

    .. versionadded:: 0.11
    """

    V1_2 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Small/resolve/a207c9a789483f95de1e9fb06acadb3da3775863/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['small']
        | {'model_version': 'v1.2', 'hf_repo': 'allenai/OlmoEarth-v1_2-Small'},
    )


class OlmoEarthBase_Weights(WeightsEnum):
    """OlmoEarth Base weights.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2511.13655
    * https://arxiv.org/abs/2605.20804v1
    * https://arxiv.org/abs/2605.20804

    .. versionadded:: 0.11
    """

    V1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/4bd1392a4539404d2c74276c39f3cb4cfff466cc/weights-551c1cc53337c6faaddead88071d7ebd2bd53ec271600fa6f0ee0a518c8b6e11.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['base']
        | {'model_version': 'v1', 'hf_repo': 'allenai/OlmoEarth-v1-Base'},
    )
    V1_1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_1-Base/resolve/4ef31d45f80c1d4fcce18f9cde40c1b5e4d96cf4/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['base']
        | {'model_version': 'v1.1', 'hf_repo': 'allenai/OlmoEarth-v1_1-Base'},
    )
    V1_2 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1_2-Base/resolve/581aa9baaa7aed4348c0903617eb92ee9f89e2ec/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['base']
        | {'model_version': 'v1.2', 'hf_repo': 'allenai/OlmoEarth-v1_2-Base'},
    )


class OlmoEarthLarge_Weights(WeightsEnum):
    """OlmoEarth Large weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    .. versionadded:: 0.11
    """

    V1 = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/b2c9f41de3d8454cb37f0cd9cc3e79ec7c4af435/weights-1adb5026bd520c54bc415a1282386954927623bab81d01be2f5b6379cc039035.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | _olmoearth_sizes['large']
        | {'model_version': 'v1', 'hf_repo': 'allenai/OlmoEarth-v1-Large'},
    )


def _olmoearth(
    weights: WeightsEnum | None, model_size: str, default_version: str, **kwargs: Any
) -> nn.Module:
    """Build an OlmoEarth model, optionally loading pre-trained weights.

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        model_size: Model size to build.
        default_version: Model version to build when ``weights`` is ``None`` and no
            ``model_version`` keyword argument is given.
        **kwargs: Passed to ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``.

    Returns:
        An OlmoEarth model.
    """
    olmoearth = lazy_import('olmoearth_pretrain_minimal')

    model_version = kwargs.pop('model_version', default_version)
    if weights:
        model_version = weights.meta['model_version']

    model: nn.Module = olmoearth.OlmoEarthPretrain_v1(
        model_size=model_size, model_version=model_version, **kwargs
    )
    if weights:
        model.model.load_state_dict(
            weights.get_state_dict(progress=True, check_hash=True, weights_only=True),
            strict=True,
        )

    return model


def olmoearth_nano(
    weights: OlmoEarthNano_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth Nano model.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2511.13655
    * https://arxiv.org/abs/2605.20804v1
    * https://arxiv.org/abs/2605.20804

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.11

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``max_patch_size``). Without ``weights``, ``model_version``
            (default ``'v1.2'``) selects the architecture; with ``weights``,
            the version comes from the weights metadata.

    Returns:
        An OlmoEarth Nano model.
    """
    return _olmoearth(weights, 'nano', 'v1.2', **kwargs)


def olmoearth_tiny(
    weights: OlmoEarthTiny_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth Tiny model.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2511.13655
    * https://arxiv.org/abs/2605.20804v1
    * https://arxiv.org/abs/2605.20804

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.11

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``max_patch_size``). Without ``weights``, ``model_version``
            (default ``'v1.2'``) selects the architecture; with ``weights``,
            the version comes from the weights metadata.

    Returns:
        An OlmoEarth Tiny model.
    """
    return _olmoearth(weights, 'tiny', 'v1.2', **kwargs)


def olmoearth_small(
    weights: OlmoEarthSmall_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth Small model.

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
            (e.g. ``max_patch_size``). Without ``weights``, ``model_version``
            (default ``'v1.2'``) selects the architecture; with ``weights``,
            the version comes from the weights metadata.

    Returns:
        An OlmoEarth Small model.
    """
    return _olmoearth(weights, 'small', 'v1.2', **kwargs)


def olmoearth_base(
    weights: OlmoEarthBase_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth Base model.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2511.13655
    * https://arxiv.org/abs/2605.20804v1
    * https://arxiv.org/abs/2605.20804

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.11

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``max_patch_size``). Without ``weights``, ``model_version``
            (default ``'v1.2'``) selects the architecture; with ``weights``,
            the version comes from the weights metadata.

    Returns:
        An OlmoEarth Base model.
    """
    return _olmoearth(weights, 'base', 'v1.2', **kwargs)


def olmoearth_large(
    weights: OlmoEarthLarge_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth Large model.

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
            (e.g. ``max_patch_size``). Without ``weights``, ``model_version``
            (default ``'v1'``) selects the architecture; with ``weights``,
            the version comes from the weights metadata.

    Returns:
        An OlmoEarth Large model.
    """
    return _olmoearth(weights, 'large', 'v1', **kwargs)


_olmoearth_v1_meta = {
    'dataset': 'OlmoEarthPretrain',
    'model': 'OlmoEarthPretrain_v1',
    'architecture': 'Vision Transformer',
    'publication': 'https://arxiv.org/abs/2511.13655',
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

    .. deprecated:: 0.11
       Will be removed in 1.0. Use :class:`OlmoEarthNano_Weights`,
       :class:`OlmoEarthTiny_Weights`, :class:`OlmoEarthBase_Weights` or
       :class:`OlmoEarthLarge_Weights` instead.
    """

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/529248a4dc3c54014c56b7504641cec98de31d1c/weights-795c68419a658fd22ccf8f2e020607675f963e9ef3b93d8e368bb17646765347.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_v1_meta
        | {'model_size': 'nano', 'hf_repo': 'allenai/OlmoEarth-v1-Nano'},
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/885784437d4e2d632b7bf51b4233426c6f4479dc/weights-66b9827af383bc444d7909a406a5b62c072bb08d6804ff47a247c2dce8fad9a4.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_v1_meta
        | {'model_size': 'tiny', 'hf_repo': 'allenai/OlmoEarth-v1-Tiny'},
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/4bd1392a4539404d2c74276c39f3cb4cfff466cc/weights-551c1cc53337c6faaddead88071d7ebd2bd53ec271600fa6f0ee0a518c8b6e11.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_v1_meta
        | {'model_size': 'base', 'hf_repo': 'allenai/OlmoEarth-v1-Base'},
    )
    LARGE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/b2c9f41de3d8454cb37f0cd9cc3e79ec7c4af435/weights-1adb5026bd520c54bc415a1282386954927623bab81d01be2f5b6379cc039035.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_v1_meta
        | {'model_size': 'large', 'hf_repo': 'allenai/OlmoEarth-v1-Large'},
    )


@deprecated(
    'Use torchgeo.models.olmoearth_nano, olmoearth_tiny, olmoearth_base or '
    'olmoearth_large instead'
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

    .. deprecated:: 0.11
       Will be removed in 1.0. Use :func:`olmoearth_nano`, :func:`olmoearth_tiny`,
       :func:`olmoearth_base` or :func:`olmoearth_large` instead.

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
