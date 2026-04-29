# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench 2 + Aurora datamodule.

This module allows the Aurora foundation model to use a WeatherBench2 dataloader.

* :class:`AuroraWeatherBench2Sequence` — a :class:`~torch.utils.data.Dataset`
  wrapper around :class:`~torchgeo.datasets.WeatherBench2` that builds
  ``(context, target)`` time windows in a regional bounding box.
* :func:`aurora_collate_fn` — a collate function that turns a batch of those
  windows into an :class:`aurora.Batch` plus the matching ``target_*`` tensors.
* :class:`WeatherBench2AuroraDataModule` — a thin
  :class:`~torchgeo.datamodules.NonGeoDataModule` that wires the two together.
"""

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..datasets import WeatherBench2
from ..datasets.utils import GeoSlice, Sample, lazy_import
from .geo import NonGeoDataModule

#: Default mapping from WeatherBench2 variable names to Aurora's surface keys.
DEFAULT_SURF_VARS: dict[str, str] = {
    '2m_temperature': '2t',
    '10m_u_component_of_wind': '10u',
    '10m_v_component_of_wind': '10v',
    'mean_sea_level_pressure': 'msl',
}

#: Default mapping from WeatherBench2 variable names to Aurora's atmospheric keys.
DEFAULT_ATMOS_VARS: dict[str, str] = {
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'specific_humidity': 'q',
    'geopotential': 'z',
}

#: Default mapping from WeatherBench2 variable names to Aurora's static keys.
DEFAULT_STATIC_VARS: dict[str, str] = {
    'land_sea_mask': 'lsm',
    'soil_type': 'slt',
    'geopotential_at_surface': 'z',
}


def _to_hw(x: Tensor) -> Tensor:
    """Reduce a static field to ``(H, W)`` regardless of how it arrives.

    WeatherBench2 broadcasts static variables like ``land_sea_mask`` across the
    time dimension, and ``geopotential`` across both time and pressure level.
    Aurora expects each static tensor to be ``(H, W)``, so we squeeze the
    leading dims by taking the first slice (these fields are constant in time).
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x[0]
    if x.ndim == 4:
        return x[0, 0]
    raise ValueError(
        f'Cannot reduce static tensor with shape {tuple(x.shape)} to (H, W).'
    )


class AuroraWeatherBench2Sequence(Dataset[Sample]):
    """Build Aurora-shaped context/target windows from a regional WeatherBench2 slice.

    Each item produced is a dict with:

    * ``surf_vars`` — ``{aurora_key: tensor[T_ctx, H, W]}``
    * ``atmos_vars`` — ``{aurora_key: tensor[T_ctx, L, H, W]}``
    * ``static_vars`` — ``{aurora_key: tensor[H, W]}``
    * ``time`` — ``tuple[Timestamp, ...]`` for the ``T_ctx`` context steps
    * ``atmos_levels`` — pressure levels in hPa
    * ``lat`` / ``lon`` — 1D coordinate tensors
    * ``target_surf_vars`` — ``{aurora_key: tensor[T_tgt, H, W]}``
    * ``target_atmos_vars`` — ``{aurora_key: tensor[T_tgt, L, H, W]}``
    * ``target_time`` — timestamps of the target steps

    Window starts are stepped at the WeatherBench2 timestep and are clipped so
    that the full ``context_steps + target_steps`` window always fits inside the
    underlying store's time range.

    Args:
        dataset: a :class:`~torchgeo.datasets.WeatherBench2` instance.
        region: ``(xmin, ymin, xmax, ymax)`` longitude/latitude bounding box.
        start_time: ISO timestamp (or :class:`pandas.Timestamp`) for the first
            window start. Clipped to ``dataset.bounds`` if it falls outside.
        end_time: exclusive end timestamp. The last window start is at
            ``end_time - (context_steps + target_steps - 1) * timestep``.
        timestep: time between samples in the store. Strings accepted by
            :func:`pandas.to_timedelta` are also valid (e.g. ``'6h'``).
        context_steps: number of input timesteps fed to the model.
        target_steps: number of target timesteps used as supervision.
        surf_vars: WB2 → Aurora surface variable mapping.
        atmos_vars: WB2 → Aurora atmospheric variable mapping.
        static_vars: WB2 → Aurora static variable mapping.

    Raises:
        ValueError: If no valid window fits inside the dataset's time range.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        dataset: WeatherBench2,
        region: tuple[float, float, float, float],
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        timestep: str | pd.Timedelta = '6h',
        context_steps: int = 2,
        target_steps: int = 1,
        surf_vars: dict[str, str] | None = None,
        atmos_vars: dict[str, str] | None = None,
        static_vars: dict[str, str] | None = None,
    ) -> None:
        """Initialize a new AuroraWeatherBench2Sequence instance.

        See the class docstring for argument descriptions.
        """
        super().__init__()
        if context_steps < 1 or target_steps < 1:
            raise ValueError('context_steps and target_steps must be >= 1.')

        self.dataset = dataset
        self.region = region
        self.context_steps = context_steps
        self.target_steps = target_steps
        self.timestep = pd.to_timedelta(timestep)
        self.surf_vars = dict(surf_vars or DEFAULT_SURF_VARS)
        self.atmos_vars = dict(atmos_vars or DEFAULT_ATMOS_VARS)
        self.static_vars = dict(static_vars or DEFAULT_STATIC_VARS)

        # Cap window start times to the dataset's actual time range so we never
        # request timestamps past the Zarr store's last entry.
        _, _, t_bounds = dataset.bounds
        tmin = pd.Timestamp(t_bounds.start)
        tmax = pd.Timestamp(t_bounds.stop)
        window = (context_steps + target_steps - 1) * self.timestep

        first = max(pd.Timestamp(start_time), tmin)
        last = pd.Timestamp(end_time) - window
        last = min(last, tmax - window)
        if last < first:
            raise ValueError(
                f'No window of {context_steps + target_steps} steps fits between '
                f'{first} and {pd.Timestamp(end_time)} given dataset range '
                f'[{tmin}, {tmax}].'
            )

        self.starts: list[pd.Timestamp] = list(
            pd.date_range(first, last, freq=self.timestep)
        )

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.starts)

    def __getitem__(self, index: int) -> Sample:
        """Return the *index*-th context/target window.

        Args:
            index: Window index.

        Returns:
            A sample dict described in the class docstring.
        """
        t0 = self.starts[index]
        n_total = self.context_steps + self.target_steps
        t_end = t0 + (n_total - 1) * self.timestep

        xmin, ymin, xmax, ymax = self.region
        xres, yres = self.dataset.res
        geoslice: GeoSlice = (
            slice(xmin, xmax, abs(xres)),
            slice(ymin, ymax, abs(yres)),
            slice(t0, t_end, 1),
        )

        sample = self.dataset[geoslice]
        variables = cast(dict[str, Tensor], sample['variables'])

        surf: dict[str, Tensor] = {}
        for src, dst in self.surf_vars.items():
            if src in variables:
                surf[dst] = variables[src][: self.context_steps]

        atmos: dict[str, Tensor] = {}
        for src, dst in self.atmos_vars.items():
            if src in variables:
                atmos[dst] = variables[src][: self.context_steps]

        statics: dict[str, Tensor] = {}
        for src, dst in self.static_vars.items():
            if src in variables:
                statics[dst] = _to_hw(variables[src])

        target_surf: dict[str, Tensor] = {}
        for src, dst in self.surf_vars.items():
            if src in variables:
                target_surf[dst] = variables[src][self.context_steps :]

        target_atmos: dict[str, Tensor] = {}
        for src, dst in self.atmos_vars.items():
            if src in variables:
                target_atmos[dst] = variables[src][self.context_steps :]

        time_all = cast(tuple[pd.Timestamp, ...], sample['time'])
        return {
            'surf_vars': surf,
            'atmos_vars': atmos,
            'static_vars': statics,
            'lat': sample['lat'],
            'lon': sample['lon'],
            'time': tuple(time_all[: self.context_steps]),
            'atmos_levels': sample['atmos_levels'],
            'target_surf_vars': target_surf,
            'target_atmos_vars': target_atmos,
            'target_time': tuple(time_all[self.context_steps :]),
        }


def aurora_collate_fn(batch: Sequence[Sample]) -> dict[str, Any]:
    """Collate :class:`AuroraWeatherBench2Sequence` samples into an Aurora batch.

    Args:
        batch: sequence of samples produced by
            :class:`AuroraWeatherBench2Sequence`.

    Returns:
        A dict with:

        * ``batch`` — an :class:`aurora.Batch` ready to feed to the model
        * ``target_surf_vars`` — ``{key: tensor[B, T_tgt, H, W]}``
        * ``target_atmos_vars`` — ``{key: tensor[B, T_tgt, L, H, W]}``
        * ``target_time`` — tuple of per-sample target time tuples
    """
    aurora = lazy_import('aurora')

    surf_keys = list(batch[0]['surf_vars'].keys())
    atmos_keys = list(batch[0]['atmos_vars'].keys())
    static_keys = list(batch[0]['static_vars'].keys())

    surf_vars = {k: torch.stack([s['surf_vars'][k] for s in batch]) for k in surf_keys}
    atmos_vars = {
        k: torch.stack([s['atmos_vars'][k] for s in batch]) for k in atmos_keys
    }
    # Static fields are batch-invariant; take them from the first sample.
    static_vars = {k: batch[0]['static_vars'][k] for k in static_keys}

    sample0 = batch[0]
    metadata = aurora.Metadata(
        lat=sample0['lat'],
        lon=sample0['lon'],
        time=tuple(s['time'][-1] for s in batch),
        atmos_levels=tuple(int(level) for level in sample0['atmos_levels']),
    )

    target_surf_keys = list(batch[0]['target_surf_vars'].keys())
    target_atmos_keys = list(batch[0]['target_atmos_vars'].keys())
    target_surf_vars = {
        k: torch.stack([s['target_surf_vars'][k] for s in batch])
        for k in target_surf_keys
    }
    target_atmos_vars = {
        k: torch.stack([s['target_atmos_vars'][k] for s in batch])
        for k in target_atmos_keys
    }

    return {
        'batch': aurora.Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        ),
        'target_surf_vars': target_surf_vars,
        'target_atmos_vars': target_atmos_vars,
        'target_time': tuple(s['target_time'] for s in batch),
    }


class WeatherBench2AuroraDataModule(NonGeoDataModule):
    """LightningDataModule for fine-tuning Aurora on regional WeatherBench2 slices.

    All splits share the same regional bounding box and time settings; pass
    different ``start_time`` / ``end_time`` ranges through ``train_kwargs`` and
    ``val_kwargs`` if you want disjoint train/val/test splits.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        paths: str,
        region: tuple[float, float, float, float],
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        timestep: str | pd.Timedelta = '6h',
        context_steps: int = 2,
        target_steps: int = 1,
        surf_vars: dict[str, str] | None = None,
        atmos_vars: dict[str, str] | None = None,
        static_vars: dict[str, str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherBench2AuroraDataModule instance.

        Args:
            paths: path or URI passed to :class:`~torchgeo.datasets.WeatherBench2`.
            region: ``(xmin, ymin, xmax, ymax)`` longitude/latitude bbox.
            start_time: first window start (inclusive).
            end_time: exclusive end of the data range.
            timestep: time between consecutive samples in the store.
            context_steps: number of context (input) steps per window.
            target_steps: number of target (supervision) steps per window.
            surf_vars: WB2 → Aurora surface variable mapping.
            atmos_vars: WB2 → Aurora atmospheric variable mapping.
            static_vars: WB2 → Aurora static variable mapping.
            batch_size: per-GPU mini-batch size (Aurora typically uses 1).
            num_workers: dataloader workers.
            **kwargs: extra keyword arguments forwarded to
                :class:`~torchgeo.datasets.WeatherBench2`.
        """
        super().__init__(
            WeatherBench2,
            batch_size=batch_size,
            num_workers=num_workers,
            paths=paths,
            **kwargs,
        )
        self.region = region
        self.start_time = start_time
        self.end_time = end_time
        self.timestep = timestep
        self.context_steps = context_steps
        self.target_steps = target_steps
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.static_vars = static_vars
        self.collate_fn = aurora_collate_fn

    def setup(self, stage: str) -> None:
        """Build the underlying WB2 dataset and wrap it into Aurora windows.

        Args:
            stage: ``'fit'``, ``'validate'``, ``'test'`` or ``'predict'``.
        """
        self.dataset = WeatherBench2(**self.kwargs)
        sequence = AuroraWeatherBench2Sequence(
            dataset=self.dataset,
            region=self.region,
            start_time=self.start_time,
            end_time=self.end_time,
            timestep=self.timestep,
            context_steps=self.context_steps,
            target_steps=self.target_steps,
            surf_vars=self.surf_vars,
            atmos_vars=self.atmos_vars,
            static_vars=self.static_vars,
        )
        if stage in ('fit',):
            self.train_dataset = sequence
        if stage in ('fit', 'validate'):
            self.val_dataset = sequence
        if stage in ('test',):
            self.test_dataset = sequence
        if stage in ('predict',):
            self.predict_dataset = sequence

    def on_after_batch_transfer(
        self, batch: dict[str, Any], dataloader_idx: int
    ) -> dict[str, Any]:
        """Pass-through hook.

        Args:
            batch: A batch of data.
            dataloader_idx: Index of the dataloader (unused).

        Returns:
            The batch unchanged.
        """
        return batch
