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
            Defaults to the dataset's full spatial bounds (i.e. the entire
            store), which is also what Aurora expects for global rollouts.
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
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        region: tuple[float, float, float, float] | None = None,
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
        x_bounds, y_bounds, _ = dataset.bounds
        self.region = region or (
            x_bounds.start,
            y_bounds.start,
            x_bounds.stop,
            y_bounds.stop,
        )
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


def aurora_predictions_to_xarray(
    preds: Sequence[Any],
    init_time: str | pd.Timestamp,
    timestep: str | pd.Timedelta = '6h',
    surf_vars: dict[str, str] | None = None,
    atmos_vars: dict[str, str] | None = None,
) -> Any:
    """Pack Aurora rollout predictions into a WeatherBench2-shaped xarray Dataset.

    The output uses WeatherBench2 variable names and the standard
    ``(time, latitude, longitude[, level])`` coordinates, ready to be persisted
    via :meth:`xarray.Dataset.to_zarr` and re-opened with
    :class:`~torchgeo.datasets.WeatherBench2`.

    The *i*-th prediction is tagged with timestamp ``T0 + (i + 1) * timestep``
    (Aurora's :func:`~aurora.rollout` yields predictions strictly after the
    initialization time).

    Args:
        preds: a sequence of :class:`aurora.Batch` predictions, one per
            rollout step (e.g. as collected from :func:`aurora.rollout`).
        init_time: forecast initialization time T0.
        timestep: time between rollout steps. Strings accepted by
            :func:`pandas.to_timedelta` are also valid (e.g. ``'6h'``).
        surf_vars: WB2 → Aurora surface variable mapping (defaults to
            :data:`DEFAULT_SURF_VARS`). Pass an empty dict to skip surface
            variables entirely.
        atmos_vars: WB2 → Aurora atmospheric variable mapping (defaults to
            :data:`DEFAULT_ATMOS_VARS`). Pass an empty dict to skip
            atmospheric variables entirely.

    Returns:
        An :class:`xarray.Dataset` ready for ``ds.to_zarr(path)`` and reuse
        with :class:`~torchgeo.datasets.WeatherBench2`.

    Raises:
        ValueError: If *preds* is empty.

    .. versionadded:: 0.8
    """
    xr = lazy_import('xarray')
    if not preds:
        raise ValueError('preds must not be empty.')

    timestep_td = pd.to_timedelta(timestep)
    init = pd.Timestamp(init_time)
    times = [init + (i + 1) * timestep_td for i in range(len(preds))]

    surf = DEFAULT_SURF_VARS if surf_vars is None else surf_vars
    atmos = DEFAULT_ATMOS_VARS if atmos_vars is None else atmos_vars
    aurora_to_wb2_surf = {v: k for k, v in surf.items()}
    aurora_to_wb2_atmos = {v: k for k, v in atmos.items()}

    def _stack(field: str, key: str) -> Any:
        return (
            torch.stack([getattr(p, field)[key][0, 0] for p in preds])
            .to(torch.float32)
            .cpu()
            .numpy()
        )

    data_vars: dict[str, tuple[tuple[str, ...], Any]] = {}
    for k in preds[0].surf_vars:
        wb2_key = aurora_to_wb2_surf.get(k)
        if wb2_key is not None:
            data_vars[wb2_key] = (
                ('time', 'latitude', 'longitude'),
                _stack('surf_vars', k),
            )
    for k in preds[0].atmos_vars:
        wb2_key = aurora_to_wb2_atmos.get(k)
        if wb2_key is not None:
            data_vars[wb2_key] = (
                ('time', 'level', 'latitude', 'longitude'),
                _stack('atmos_vars', k),
            )

    coords: dict[str, Any] = {
        'time': times,
        'latitude': preds[0].metadata.lat.detach().cpu().numpy(),
        'longitude': preds[0].metadata.lon.detach().cpu().numpy(),
    }
    if any('level' in dims for dims, _ in data_vars.values()):
        coords['level'] = list(preds[0].metadata.atmos_levels)

    return xr.Dataset(data_vars, coords=coords)


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
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        region: tuple[float, float, float, float] | None = None,
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
            start_time: first window start (inclusive).
            end_time: exclusive end of the data range.
            region: ``(xmin, ymin, xmax, ymax)`` longitude/latitude bbox.
                Defaults to the dataset's full spatial bounds (i.e. the whole
                store), which is also what Aurora expects for global rollouts.
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
