# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo pretrained weight definitions."""

import enum
from dataclasses import dataclass, field
from typing import Any

import torch
from timm.models import PretrainedCfg
from torch import nn


@dataclass
class Weights(PretrainedCfg):
    """Pretrained model configuration with TorchGeo metadata."""

    transforms: nn.Module = field(default_factory=nn.Identity)
    meta: dict[str, Any] = field(default_factory=dict)

    def get_state_dict(
        self, *args: Any, progress: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Download and return the pretrained state dict."""
        if not isinstance(self.url, str):
            raise ValueError('No URL is available for these weights')

        state_dict: dict[str, Any] = torch.hub.load_state_dict_from_url(
            self.url, *args, progress=progress, **kwargs
        )
        return state_dict


class WeightsEnum(enum.Enum):
    """Compatibility enum for TorchGeo pretrained configurations."""

    def __getattr__(self, name: str) -> Any:
        """Forward configuration attributes to the enum value."""
        try:
            value = object.__getattribute__(self, '_value_')
        except AttributeError as ex:
            raise AttributeError(name) from ex
        return getattr(value, name)

    def get_state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Download and return the pretrained state dict."""
        state_dict: dict[str, Any] = self.value.get_state_dict(*args, **kwargs)
        return state_dict
