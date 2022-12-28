# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utility functions for models."""

import os
from typing import Any, Dict

import torch
from torch import Tensor

from torchgeo.datasets.utils import download_url


def adjust_dino_weights_zhu_lab(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Loading Dino weights from https://github.com/zhu-xlab/SSL4EO-S12.

    # https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/
           # src/benchmark/transfer_classification/models/dino/utils.py#L92-L103
    """
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict


def load_state_dict_from_url(
    root: str, filename: str, url: str, map_location: torch.device
) -> Dict[str, Any]:
    """Download and load a checkpoint.

    Args:
        root: root directory where checkpoint file should be saved
        filename: filename the downloaded checkpoint file should have
        url: url to checkpoint file to download
        map_location: torch device to load checkpoint

    Returns:
        loaded checkpoint
    """
    if not os.path.exists(os.path.join(root, filename)):
        download_url(url, root=root, filename=filename)
    return torch.load(os.path.join(root, filename), map_location=map_location)
