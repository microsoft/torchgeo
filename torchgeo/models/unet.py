# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained U-Net models."""

from typing import Any

import kornia.augmentation as K
import segmentation_models_pytorch as smp
import torch
from kornia.constants import Resample
from segmentation_models_pytorch import Unet
from torchvision.models._api import Weights, WeightsEnum

import torchgeo.transforms as T
from torchgeo.transforms.transforms import _Clamp

# Specified in https://github.com/fieldsoftheworld/ftw-baselines
# First 4 S2 bands are for image t1 and last 4 bands are for image t2
_ftw_sentinel2_bands = ('B4', 'B3', 'B2', 'B8A', 'B4', 'B3', 'B2', 'B8A')

# https://github.com/fieldsoftheworld/ftw-baselines/blob/main/src/ftw/datamodules.py
# Normalization by 3k (for S2 uint16 input)
_ftw_transforms = K.AugmentationSequential(
    K.Normalize(mean=torch.tensor(0.0), std=torch.tensor(3000.0)), data_keys=None
)

# Specified in https://github.com/fieldsoftheworld/ftw-baselines
# First 4 S2 bands are for image t1 and last 4 bands are for image t2
_ai4g_flood_sentinel1_bands = ('VV', 'VH')
_ai4g_flood_sentinel1_transform_bands = ('VV', 'VH', 'VV', 'VH')

# https://github.com/microsoft/ai4g-flood/blob/main/src/run_flood_detection_downloaded_images.py#L54
_ai4g_flood_transforms = K.AugmentationSequential(
    # Convert to decibel scale and shift to [0, 255] range
    T.PowerToDecibel(shift=135.0, scale=2.0),
    _Clamp(p=1, min=0, max=255),
    # Extract change mask from pre and post images
    T.ToThresholdedChangeMask(
        change_thresholds=[10.0, 10.0],
        thresholds=[100.0, 90.0],
        min_thresholds=[75.0, 70.0],
    ),
    K.Resize(size=(128, 128), resample=Resample.NEAREST),
    data_keys=None,
)

# No normalization used see: https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/data/datamodule.py#L145
_tcd_bands = ['R', 'G', 'B']
_tcd_transforms = K.AugmentationSequential(K.Resize(size=(1024, 1024)), data_keys=None)


class Unet_Weights(WeightsEnum):  # type: ignore[misc]
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
    SENTINEL1_AI4G_FLOOD = Weights(
        url='https://huggingface.co/torchgeo/ai4g_flood/resolve/672bfb53b61a91114941ac9e4338ebc96dff6ec7/unet_mobilenetv2_sentinel1_ai4g_flood-d95df7aa.pth',
        transforms=_ai4g_flood_transforms,
        meta={
            'dataset': 'AI4G Global Flood Extent',
            'in_chans': 2,
            'in_chans_transform': 4,
            'num_classes': 2,
            'model': 'U-Net',
            'encoder': 'mobilenet_v2',
            'publication': 'https://arxiv.org/abs/2411.01411',
            'repo': 'https://github.com/microsoft/ai4g-flood',
            'bands': _ai4g_flood_sentinel1_bands,
            'bands_transform': _ai4g_flood_sentinel1_transform_bands,
            'license': 'MIT',
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
        url='https://hf.co/isaaccorley/unet_resnet34_oam_rgb_tcd/resolve/40c914bbcbe43a6a87c81adb0a22ff2d4a53204d/unet_resnet34_oam_rgb_tcd-72b9b753.pth',
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


def unet(
    weights: Unet_Weights | None = None,
    classes: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> Unet:
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

    if weights:
        kwargs['encoder_weights'] = None
        kwargs['in_channels'] = weights.meta['in_chans']
        kwargs['encoder_name'] = weights.meta['encoder']
        kwargs['classes'] = weights.meta['num_classes'] if classes is None else classes
    else:
        kwargs['classes'] = 1 if classes is None else classes

    model: Unet = smp.create_model(*args, **kwargs)

    if weights:
        state_dict = weights.get_state_dict(progress=True)

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
