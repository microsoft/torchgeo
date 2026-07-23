# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo model registry and pretrained weight definitions."""

import enum
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import timm
import torch
from timm.models import (
    PretrainedCfg,
    generate_default_cfgs,
    get_pretrained_cfg,
    register_model,
)
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


# timm looks for this mapping in the module that defines each entrypoint.
default_cfgs: dict[str, PretrainedCfg] = {}
_model_names: dict[str, str] = {}
_weights_by_name: dict[str, WeightsEnum] = {}


def _get_input_size(weight: WeightsEnum) -> tuple[int, int, int]:
    """Get a timm input size from TorchGeo weight metadata."""
    input_size = weight.input_size
    in_chans = weight.meta.get('in_chans', input_size[0])
    image_size = weight.meta.get('img_size', weight.meta.get('image_size'))
    if image_size is None:
        return (in_chans, input_size[1], input_size[2])
    return (in_chans, image_size, image_size)


def _resolve_weight(
    model_name: str, pretrained_cfg: str | PretrainedCfg | WeightsEnum | None
) -> WeightsEnum:
    """Resolve a timm configuration or tag to a TorchGeo weight."""
    if isinstance(pretrained_cfg, WeightsEnum):
        return pretrained_cfg

    if isinstance(pretrained_cfg, str):
        weight = _weights_by_name.get(pretrained_cfg)
        if weight is not None:
            return weight
        key = f'{model_name}.{pretrained_cfg}'
    elif pretrained_cfg is None:
        cfg = get_pretrained_cfg(model_name)
        if cfg is None:
            raise ValueError(f'{model_name} does not have pretrained weights')
        if cfg.tag is None:
            raise ValueError(f'{model_name} does not have a tagged pretrained weight')
        key = f'{model_name}.{cfg.tag}'
    else:
        if pretrained_cfg.tag is None:
            raise ValueError(f'{model_name} does not have a tagged pretrained weight')
        key = f'{model_name}.{pretrained_cfg.tag}'

    try:
        return _weights_by_name[key]
    except KeyError as ex:
        raise ValueError(f'{key} is not a valid TorchGeo pretrained weight') from ex


def _register_weights(
    model_name: str, weights: type[WeightsEnum]
) -> dict[str, PretrainedCfg]:
    """Convert TorchGeo weight members to timm tags and configurations."""
    configs: dict[str, PretrainedCfg] = {}

    for index, weight in enumerate(weights):
        tag = weight.name.lower()
        cfg_name = f'{model_name}.{tag}'
        if index == 0:
            cfg_name += '*'

        configs[cfg_name] = replace(weight.value, input_size=_get_input_size(weight))
        _weights_by_name[f'{model_name}.{tag}'] = weight
        _weights_by_name[str(weight)] = weight

    return configs


def _create_entrypoint(
    model_name: str, builder: Callable[..., nn.Module]
) -> Callable[..., nn.Module]:
    """Create a timm entrypoint around an existing TorchGeo builder."""

    def entrypoint(
        pretrained: bool = False,
        pretrained_cfg: str | PretrainedCfg | WeightsEnum | None = None,
        pretrained_cfg_overlay: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Adapt timm arguments to the TorchGeo builder API."""
        # These arguments are consumed by timm before it calls the entrypoint.
        del pretrained_cfg_overlay, cache_dir

        # Keep accepting the old TorchGeo ``weights=`` spelling.
        if 'weights' in kwargs:
            pretrained_cfg = kwargs.pop('weights')
            pretrained = True

        if not pretrained:
            return builder(**kwargs)

        weight = _resolve_weight(model_name, pretrained_cfg)
        return builder(weights=weight, **kwargs)

    entrypoint.__name__ = model_name
    entrypoint.__module__ = __name__
    return entrypoint


def register_models(
    models: dict[str, Callable[..., nn.Module]],
    model_weights: dict[str, type[WeightsEnum]],
) -> None:
    """Register TorchGeo builders, tags, and default configurations with timm."""
    configs: dict[str, PretrainedCfg | dict[str, Any]] = {}
    entrypoints: list[Callable[..., nn.Module]] = []

    # Build all configs before registering entrypoints so timm can find the
    # complete default_cfgs mapping during registration.
    for name, builder in models.items():
        model_name = f'torchgeo_{name}'
        _model_names[name] = model_name

        weights = model_weights.get(name)
        if weights is not None:
            configs.update(_register_weights(model_name, weights))

        entrypoints.append(_create_entrypoint(model_name, builder))

    default_cfgs.update(generate_default_cfgs(configs))
    for entrypoint in entrypoints:
        register_model(entrypoint)


def get_model_name(name: str) -> str:
    """Return the timm name for a TorchGeo model."""
    return _model_names[name]


def create_model(name: str, *args: Any, **kwargs: Any) -> nn.Module:
    """Create a model through timm's registry."""
    return timm.create_model(name, *args, **kwargs)


def get_weight(name: str) -> WeightsEnum:
    """Return a TorchGeo weight by its legacy or timm name."""
    try:
        return _weights_by_name[name]
    except KeyError as ex:
        raise ValueError(f'{name} is not a valid WeightsEnum') from ex
