# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained U-Net models."""

from typing import Any

import kornia.augmentation as K
import segmentation_models_pytorch as smp
import torch
from kornia.constants import Resample
from kornia.contrib import Lambda
from segmentation_models_pytorch import Unet
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

import torchgeo.transforms.transforms as T

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


# https://github.com/microsoft/ai4g-flood/blob/main/src/run_flood_detection_downloaded_images.py#L54
class Sentinel1ChangeMap(torch.nn.Module):
    """Extracts change map from Sentinel-1 pre and post imagery.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        vv_threshold: int = 100,
        vh_threshold: int = 90,
        vv_min_threshold: int = 75,
        vh_min_threshold: int = 70,
        delta_amplitude: float = 10,
    ) -> None:
        """Initializes the Sentinel1ChangeMap transform.

        Args:
            vv_threshold: Threshold for VV band to detect change.
            vh_threshold: Threshold for VH band to detect change.
            vv_min_threshold: Minimum threshold for VV band to consider valid data.
            vh_min_threshold: Minimum threshold for VH band to consider valid data.
            delta_amplitude: Minimum change in amplitude to consider as a change.
        """
        super().__init__()
        self.vv_threshold = vv_threshold
        self.vh_threshold = vh_threshold
        self.vv_min_threshold = vv_min_threshold
        self.vh_min_threshold = vh_min_threshold
        self.delta_amplitude = delta_amplitude

    def forward(self, x: Tensor) -> Tensor:
        """Extracts change map from Sentinel-1 pre and post imagery.

        Args:
            x: Input tensor of shape (N, 4, H, W) where N is the batch size,
                4 is the number of channels (VV pre, VH pre, VV post, VH post),
                and H, W are the height and width of the images.

        Returns:
            A tensor of shape (N, 2, H, W) containing the change maps for VV and VH bands.
            The values are 1 for change detected and 0 for no change.
        """
        has_batch_dim = x.dim() == 4

        if not has_batch_dim:
            x = x.unsqueeze(0)

        vv_pre, vh_pre, vv_post, vh_post = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        vv_change = (
            (vv_post < self.vv_threshold)
            & (vv_pre > self.vv_threshold)
            & ((vv_pre - vv_post) > self.delta_amplitude)
        ).to(torch.int)
        vh_change = (
            (vh_post < self.vh_threshold)
            & (vh_pre > self.vh_threshold)
            & ((vh_pre - vh_post) > self.delta_amplitude)
        ).to(torch.int)

        zero_idx = (
            (vv_post < self.vv_min_threshold)
            | (vv_pre < self.vv_min_threshold)
            | (vh_post < self.vh_min_threshold)
            | (vh_pre < self.vh_min_threshold)
        )
        vv_change[zero_idx] = 0
        vh_change[zero_idx] = 0
        change = torch.stack([vv_change, vh_change], dim=1).to(torch.float)

        if not has_batch_dim:
            change = change.squeeze(dim=0)

        return change


_ai4g_flood_transforms = K.AugmentationSequential(
    # Convert to decibel scale and shift to [0, 255] range
    K.ImageSequential(Lambda(lambda x: 2 * 10 * torch.log10(x) + 135)),
    T._Clamp(p=1, min=0, max=255),
    # Extract change map from pre and post images
    K.ImageSequential(
        Sentinel1ChangeMap(
            vv_threshold=100,
            vh_threshold=90,
            vv_min_threshold=75,
            vh_min_threshold=70,
            delta_amplitude=10,
        )
    ),
    K.Resize(size=(128, 128), resample=Resample.NEAREST),
    data_keys=None,
)

# https://github.com/pytorch/vision/pull/6883
# https://github.com/pytorch/vision/pull/7107
# Can be removed once torchvision>=0.15 is required
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]


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
            'license': 'MIT',
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
