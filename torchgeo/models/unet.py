# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained U-Net models."""

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision.transforms.v2 as T
from torchvision.models._api import Weights, WeightsEnum

# Specified in https://github.com/fieldsoftheworld/ftw-baselines
# First 4 S2 bands are for image t1 and last 4 bands are for image t2
_ftw_sentinel2_bands = ['B4', 'B3', 'B2', 'B8', 'B4', 'B3', 'B2', 'B8']

# https://github.com/fieldsoftheworld/ftw-baselines/blob/main/src/ftw/datamodules.py
# Normalization by 3k (for S2 uint16 input)
_ftw_transforms = nn.Sequential(T.Normalize(mean=[0.0], std=[3000.0], inplace=True))

# No normalization used see: https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/data/datamodule.py#L145
_tcd_bands = ['R', 'G', 'B']
_tcd_transforms = nn.Sequential(T.Resize(size=(1024, 1024)))


class Unet_Weights(WeightsEnum):
    """U-Net weights.

    For `smp <https://github.com/qubvel-org/segmentation_models.pytorch>`_
    *Unet* implementation.

    .. versionadded:: 0.8
    """

    SENTINEL2_2CLASS_FTW = Weights(
        url='https://huggingface.co/torchgeo/ftw/resolve/d2fdab6ea9d9cd38b491292cc9a5c8642533cef5/commercial/2-class/sentinel2_unet_effb3-9c04b7c6.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 2,
            'model': 'U-Net',
            'encoder': 'efficientnet-b3',
            'publication': 'https://arxiv.org/abs/2409.16252',
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'CC-BY-4.0',
        },
    )
    SENTINEL2_3CLASS_FTW = Weights(
        url='https://huggingface.co/torchgeo/ftw/resolve/d2fdab6ea9d9cd38b491292cc9a5c8642533cef5/commercial/3-class/sentinel2_unet_effb3-5d591cbb.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b3',
            'publication': 'https://arxiv.org/abs/2409.16252',
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'CC-BY-4.0',
        },
    )
    SENTINEL2_2CLASS_NC_FTW = Weights(
        url='https://huggingface.co/torchgeo/ftw/resolve/d2fdab6ea9d9cd38b491292cc9a5c8642533cef5/noncommercial/2-class/sentinel2_unet_effb3-bf010a31.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 2,
            'model': 'U-Net',
            'encoder': 'efficientnet-b3',
            'publication': 'https://arxiv.org/abs/2409.16252',
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'non-commercial',
        },
    )
    SENTINEL2_3CLASS_NC_FTW = Weights(
        url='https://huggingface.co/torchgeo/ftw/resolve/d2fdab6ea9d9cd38b491292cc9a5c8642533cef5/noncommercial/3-class/sentinel2_unet_effb3-ed36f465.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b3',
            'publication': 'https://arxiv.org/abs/2409.16252',
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'non-commercial',
        },
    )
    SENTINEL2_FTW_PRUE_EFNETB3 = Weights(
        url='https://hf.co/isaaccorley/ftw-prue/resolve/c2d73d8478415db89b51e7635c1d2722e1056c29/prue_efnet3.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b3',
            'publication': None,
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'non-commercial',
        },
    )
    SENTINEL2_FTW_PRUE_EFNETB5 = Weights(
        url='https://hf.co/isaaccorley/ftw-prue/resolve/c2d73d8478415db89b51e7635c1d2722e1056c29/prue_efnet5.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b5',
            'publication': None,
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'non-commercial',
        },
    )
    SENTINEL2_FTW_PRUE_EFNETB7 = Weights(
        url='https://hf.co/isaaccorley/ftw-prue/resolve/c2d73d8478415db89b51e7635c1d2722e1056c29/prue_efnet7.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b7',
            'publication': None,
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'non-commercial',
        },
    )
    SENTINEL2_FTW_PRUE_CCBY_EFNETB3 = Weights(
        url='https://hf.co/isaaccorley/ftw-prue-ccby/resolve/ce7ffffbceb1b55b3b6db77ecbc82313b7afa163/prue_efnetb3_ccby-aa82bfe9.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b3',
            'publication': None,
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'CC-BY-4.0',
        },
    )
    SENTINEL2_FTW_PRUE_CCBY_EFNETB5 = Weights(
        url='https://hf.co/isaaccorley/ftw-prue-ccby/resolve/ce7ffffbceb1b55b3b6db77ecbc82313b7afa163/prue_efnetb5_ccby-a3aaa8b6.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b5',
            'publication': None,
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'CC-BY-4.0',
        },
    )
    SENTINEL2_FTW_PRUE_CCBY_EFNETB7 = Weights(
        url='https://hf.co/isaaccorley/ftw-prue-ccby/resolve/ce7ffffbceb1b55b3b6db77ecbc82313b7afa163/prue_efnetb7_ccby-da5ad55e.pth',
        transforms=_ftw_transforms,
        meta={
            'dataset': 'FTW',
            'in_chans': 8,
            'num_classes': 3,
            'model': 'U-Net',
            'encoder': 'efficientnet-b7',
            'publication': None,
            'repo': 'https://github.com/fieldsoftheworld/ftw-baselines',
            'bands': _ftw_sentinel2_bands,
            'license': 'CC-BY-4.0',
        },
    )
    OAM_RGB_RESNET50_TCD = Weights(
        url='https://hf.co/isaaccorley/unet_resnet50_oam_rgb_tcd/resolve/5df2fe5a0e80fd6e12939686b7370c53f73bf389/unet_resnet50_oam_rgb_tcd-72b9b753.pth',
        transforms=_tcd_transforms,
        meta={
            'dataset': 'OAM-TCD',
            'in_chans': 3,
            'num_classes': 2,
            'model': 'U-Net',
            'encoder': 'resnet50',
            'publication': 'https://arxiv.org/abs/2407.11743',
            'repo': 'https://github.com/restor-foundation/tcd',
            'bands': _tcd_bands,
            'classes': ('background', 'tree-canopy'),
            'input_shape': (3, 1024, 1024),
            'resolution': 0.1,
            'license': 'CC-BY-NC-4.0',
        },
    )
    OAM_RGB_RESNET34_TCD = Weights(
        url='https://hf.co/isaaccorley/unet_resnet34_oam_rgb_tcd/resolve/40c914bbcbe43a6a87c81adb0a22ff2d4a53204d/unet_resnet34_oam_rgb_tcd-9472042e.pth',
        transforms=_tcd_transforms,
        meta={
            'dataset': 'OAM-TCD',
            'in_chans': 3,
            'num_classes': 2,
            'model': 'U-Net',
            'encoder': 'resnet34',
            'publication': 'https://arxiv.org/abs/2407.11743',
            'repo': 'https://github.com/restor-foundation/tcd',
            'bands': _tcd_bands,
            'classes': ('background', 'tree-canopy'),
            'input_shape': (3, 1024, 1024),
            'resolution': 0.1,
            'license': 'CC-BY-NC-4.0',
        },
    )
    NAIP_RGBN_RESNET18_CHESAPEAKERSC = Weights(
        url='https://hf.co/isaaccorley/chesapeakersc/resolve/fe3dc77a9edfe95fde49b0318fb047c1fc6dd195/unet-resnet18-6c2e3984.pth',
        transforms=T.Normalize(mean=[0.0], std=[255.0], inplace=True),
        meta={
            'dataset': 'ChesapeakeRSC',
            'in_chans': 4,
            'num_classes': 2,
            'model': 'U-Net',
            'encoder': 'resnet18',
            'publication': 'https://arxiv.org/abs/2401.06762',
            'repo': 'https://github.com/isaaccorley/ChesapeakeRSC',
            'bands': ('R', 'G', 'B', 'N'),
            'classes': ('background', 'road'),
            'input_shape': (4, 512, 512),
            'resolution': 1.0,
            'license': 'MIT',
        },
    )


def unet(
    weights: Unet_Weights | None = None,
    classes: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    """U-Net model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/1505.04597

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        classes: Number of output classes. If not specified, the number of
            classes will be inferred from the weights.
        *args: Additional arguments to pass to ``segmentation_models_pytorch.create_model``
        **kwargs: Additional keyword arguments to pass to ``segmentation_models_pytorch.create_model``

    Returns:
        A U-Net model.
    """
    kwargs['arch'] = 'Unet'
    kwargs['encoder_weights'] = None

    if weights:
        kwargs['in_channels'] = weights.meta['in_chans']
        kwargs['encoder_name'] = weights.meta['encoder']
        kwargs['classes'] = weights.meta['num_classes'] if classes is None else classes
    else:
        kwargs['classes'] = 1 if classes is None else classes

    model: nn.Module = smp.create_model(*args, **kwargs)

    if weights:
        state_dict = weights.get_state_dict(progress=True)

        # Load full pretrained model
        if kwargs['classes'] == weights.meta['num_classes']:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=True
            )
        # Random initialize segmentation head for new task
        else:
            del state_dict['segmentation_head.0.weight']  # type: ignore[not-subscriptable]
            del state_dict['segmentation_head.0.bias']  # type: ignore[not-subscriptable]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
        assert set(missing_keys) <= {
            'segmentation_head.0.weight',
            'segmentation_head.0.bias',
        }
        assert not unexpected_keys

    return model
