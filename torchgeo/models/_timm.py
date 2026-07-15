# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""timm registry integration."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import timm
from timm.models import (
    PretrainedCfg,
    generate_default_cfgs,
    get_pretrained_cfg,
    register_model,
)
from torch import nn

from ._weights import WeightsEnum

default_cfgs: dict[str, PretrainedCfg] = {}
_model_names: dict[str, str] = {}
_weights_by_name: dict[str, WeightsEnum] = {}


def _cfg_input_size(weight: WeightsEnum) -> tuple[int, int, int]:
    """Infer an input size from TorchGeo metadata."""
    input_size = weight.input_size
    in_chans = weight.meta.get('in_chans', input_size[0])
    image_size = weight.meta.get('img_size', weight.meta.get('image_size'))
    if image_size is None:
        return (in_chans, input_size[1], input_size[2])
    return (in_chans, image_size, image_size)


def _get_weight(
    model_name: str,
    pretrained_cfg: str | PretrainedCfg | WeightsEnum | None,
) -> WeightsEnum:
    """Resolve a timm pretrained configuration to a TorchGeo weight."""
    if isinstance(pretrained_cfg, WeightsEnum):
        return pretrained_cfg

    if isinstance(pretrained_cfg, str) and pretrained_cfg in _weights_by_name:
        return _weights_by_name[pretrained_cfg]

    if pretrained_cfg is None:
        cfg = get_pretrained_cfg(model_name)
        if cfg is None:
            raise ValueError(f'{model_name} does not have pretrained weights')
        key = f'{model_name}.{cfg.tag}' if cfg.tag else model_name
    elif isinstance(pretrained_cfg, str):
        key = f'{model_name}.{pretrained_cfg}'
    elif isinstance(pretrained_cfg, PretrainedCfg):
        if pretrained_cfg.tag is None:
            raise ValueError(f'{model_name} does not have a tagged pretrained weight')
        key = f'{model_name}.{pretrained_cfg.tag}'
    else:
        raise TypeError(f'Unsupported pretrained configuration: {pretrained_cfg!r}')

    try:
        return _weights_by_name[key]
    except KeyError as ex:
        raise ValueError(f'{key} is not a valid TorchGeo pretrained weight') from ex


def register_models(
    models: dict[str, Callable[..., nn.Module]],
    model_weights: dict[str | Callable[..., nn.Module], type[WeightsEnum]],
) -> None:
    """Register TorchGeo models and weights with timm."""
    cfg_dict: dict[str, Any] = {}
    weights_by_model: dict[str, dict[str, WeightsEnum]] = {}

    for name in models:
        timm_name = f'torchgeo_{name}'
        _model_names[name] = timm_name
        weights = model_weights.get(name)
        if weights is None:
            continue

        model_weights_by_tag: dict[str, WeightsEnum] = {}
        for index, weight in enumerate(weights):
            tag = weight.name.lower()
            if index == 0:
                tag += '*'
            cfg = replace(weight.value, input_size=_cfg_input_size(weight))
            cfg_dict[f'{timm_name}.{tag}'] = cfg
            clean_tag = tag.rstrip('*')
            key = f'{timm_name}.{clean_tag}'
            model_weights_by_tag[clean_tag] = weight
            _weights_by_name[key] = weight
            _weights_by_name[str(weight)] = weight
        weights_by_model[timm_name] = model_weights_by_tag

    default_cfgs.update(generate_default_cfgs(cfg_dict))

    for name, builder in models.items():
        timm_name = _model_names[name]
        model_weights_by_tag = weights_by_model.get(timm_name, {})

        def entrypoint(
            pretrained: bool = False,
            pretrained_cfg: Any = None,
            pretrained_cfg_overlay: dict[str, Any] | None = None,
            cache_dir: str | None = None,
            _builder: Callable[..., nn.Module] = builder,
            _model_name: str = timm_name,
            _weights_by_tag: dict[str, WeightsEnum] = model_weights_by_tag,
            **kwargs: Any,
        ) -> nn.Module:
            del pretrained_cfg_overlay, cache_dir

            legacy_weights = kwargs.pop('weights', None)
            if legacy_weights is not None:
                pretrained_cfg = legacy_weights
                pretrained = True

            if pretrained:
                if isinstance(pretrained_cfg, str):
                    weight = _weights_by_tag.get(pretrained_cfg)
                    if weight is None:
                        weight = _get_weight(_model_name, pretrained_cfg)
                else:
                    weight = _get_weight(_model_name, pretrained_cfg)
                return _builder(weights=weight, **kwargs)

            return _builder(**kwargs)

        entrypoint.__name__ = timm_name
        entrypoint.__module__ = __name__
        register_model(entrypoint)


def get_model_name(name: str) -> str:
    """Return the timm name for a TorchGeo model."""
    return _model_names[name]


def create_model(name: str, *args: Any, **kwargs: Any) -> nn.Module:
    """Create a model through timm's registry."""
    return timm.create_model(name, *args, **kwargs)


def get_weight(name: str) -> WeightsEnum:
    """Return a TorchGeo weight by legacy or timm name."""
    try:
        return _weights_by_name[name]
    except KeyError as ex:
        raise ValueError(f'{name} is not a valid WeightsEnum') from ex
