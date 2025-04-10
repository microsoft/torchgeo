# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained U-Net models."""

from typing import Any

import kornia.augmentation as K
import torch
from segmentation_models_pytorch import Unet
from torchvision.models._api import Weights, WeightsEnum

_ftw_sentinel2_bands = ['B4', 'B3', 'B2', 'B8A', 'B4', 'B3', 'B2', 'B8A']

# https://github.com/fieldsoftheworld/ftw-baselines/blob/main/src/ftw/datamodules.py
# Normalization by 3k (for S2 uint16 input)
_ftw_transforms = K.AugmentationSequential(
    K.Normalize(mean=torch.tensor(0.0), std=torch.tensor(3000.0)), data_keys=None
)

# https://github.com/pytorch/vision/pull/6883
# https://github.com/pytorch/vision/pull/7107
# Can be removed once torchvision>=0.15 is required
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]


class Unet_Weights(WeightsEnum):  # type: ignore[misc]
    """U-Net weights.

    For `smp <https://github.com/qubvel-org/segmentation_models.pytorch>`_
    *unet* implementation.

    .. versionadded:: 0.7
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
            'license': 'CC BY',
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
            'license': 'CC BY',
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
            'license': 'CC BY-NC',
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
            'license': 'CC BY-NC',
        },
    )


def unet(
    weights: Unet_Weights | None = None,
    classes: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> Unet:
    """U-Net model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/1505.04597

    .. versionadded:: 0.7

    Args:
        weights: Pre-trained model weights to use.
        classes: Number of output classes. If not specified, the number of
            classes will be inferred from the weights.
        *args: Additional arguments to pass to :func:`smp.Unet`
        **kwargs: Additional keyword arguments to pass to :func:`smp.Unet`

    Returns:
        A U-Net model.
    """
    if weights:
        kwargs['encoder_weights'] = None
        kwargs['in_chans'] = weights.meta['in_chans']
        kwargs['encoder_name'] = weights.meta['encoder']
        kwargs['classes'] = weights.meta['num_classes'] if classes is None else classes
    else:
        kwargs['classes'] = 1 if classes is None else classes

    model: Unet = Unet(*args, **kwargs)

    if weights:
        state_dict = weights.get_state_dict(progress=True)

        # Load full pretrained model
        if kwargs['classes'] == weights.meta['num_classes']:
            missing_keys, _ = model.load_state_dict(state_dict, strict=True)
        # Random initialize segmentation head for new task
        else:
            del state_dict['segmentation_head.0.weight']
            del state_dict['segmentation_head.0.bias']
            missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        assert set(missing_keys) <= {
            'segmentation_head.0.weight',
            'segmentation_head.0.bias',
        }

    return model
