# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained U-Net models."""

from typing import Any

import segmentation_models_pytorch as smp
import torchvision.transforms.v2 as T
from torch import nn
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
        url='https://hf.co/torchgeo/ftw/resolve/2ff807f35ec4e04ab329cd66de6117d8b4c26578/commercial/2-class/sentinel2_unet_effb3-3ce575ed.pth',
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
        url='https://hf.co/torchgeo/ftw/resolve/d85ee8487a1b513bceaa747d11ad48aa63519f61/commercial/3-class/sentinel2_unet_effb3-74f7eb81.pth',
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
        url='https://hf.co/torchgeo/ftw/resolve/3224ec89837c6b051b796f329a59dbd44efbf81f/noncommercial/2-class/sentinel2_unet_effb3-2bbd554b.pth',
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
        url='https://hf.co/torchgeo/ftw/resolve/71f0b677a3ac8fd00bc08f61106fe8d8f70ce158/noncommercial/3-class/sentinel2_unet_effb3-364c657c.pth',
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
        url='https://hf.co/isaaccorley/ftw-prue-ccby/resolve/ebd2a7948ac1aad45eadd1d47abc6f63260843bc/prue_efnetb3_ccby-b35d59a6.pth',
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
        url='https://hf.co/isaaccorley/ftw-prue-ccby/resolve/9d54aff1c6b903fb9f308dbcb5cba6b98c87a336/prue_efnetb5_ccby-4fd92b24.pth',
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
        url='https://hf.co/isaaccorley/ftw-prue-ccby/resolve/59f3136ceb4904dba8eda40e0168b206071cbe6b/prue_efnetb7_ccby-4bfbbbde.pth',
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
        url='https://hf.co/isaaccorley/unet_resnet50_oam_rgb_tcd/resolve/74a8e34652e9bcd08af24d98195c6610edeb80da/unet_resnet50_oam_rgb_tcd-c648cc7d.pth',
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
        url='https://hf.co/isaaccorley/unet_resnet34_oam_rgb_tcd/resolve/064afa3dae671d62200cd34566617a02a107d4c3/unet_resnet34_oam_rgb_tcd-3e2a4603.pth',
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
        url='https://hf.co/isaaccorley/chesapeakersc/resolve/f237c88706903ae522410ecd0688ef307e3d95a7/unet-resnet18-829a85c5.pth',
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
        state_dict = weights.get_state_dict(
            progress=True, check_hash=True, weights_only=True
        )

        # Load full pretrained model
        if kwargs['classes'] == weights.meta['num_classes']:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=True
            )
        # Random initialize segmentation head for new task
        else:
            del state_dict['segmentation_head.0.weight']
            del state_dict['segmentation_head.0.bias']
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
        assert set(missing_keys) <= {
            'segmentation_head.0.weight',
            'segmentation_head.0.bias',
        }
        assert not unexpected_keys

    return model
