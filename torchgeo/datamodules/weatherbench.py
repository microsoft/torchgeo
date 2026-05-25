# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench 2 + Aurora datamodule."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..datasets import WeatherBench2
from ..datasets.utils import lazy_import
from .geo import NonGeoDataModule

#: Default WeatherBench 2 -> Aurora surface variable mapping.
DEFAULT_SURF_VARS: dict[str, str] = {
    '2m_temperature': '2t',
    '10m_u_component_of_wind': '10u',
    '10m_v_component_of_wind': '10v',
    'mean_sea_level_pressure': 'msl',
}

#: Default WeatherBench 2 -> Aurora atmospheric variable mapping.
DEFAULT_ATMOS_VARS: dict[str, str] = {
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'specific_humidity': 'q',
    'geopotential': 'z',
}

#: Default WeatherBench 2 -> Aurora static variable mapping.
DEFAULT_STATIC_VARS: dict[str, str] = {
    'land_sea_mask': 'lsm',
    'soil_type': 'slt',
    'geopotential_at_surface': 'z',
}


def _tensor(arr: Any) -> Tensor:
    return torch.as_tensor(np.asarray(arr.values, dtype=np.float32))


def _slice(
    ds: Any,
    times: Sequence[Any],
    region: tuple[float, float, float, float] | None,
) -> Any:
    sel: dict[str, Any] = {'time': list(times)}
    if region is not None:
        xmin, ymin, xmax, ymax = region
        # WeatherBench 2 latitude is descending; slice ymax -> ymin.
        sel['longitude'] = slice(xmin, xmax)
        sel['latitude'] = slice(ymax, ymin)
    return ds.sel(**sel)


def aurora_batch_from_xarray(
    ds: Any,
    times: Sequence[str | pd.Timestamp],
    region: tuple[float, float, float, float] | None = None,
    surf_vars: Mapping[str, str] | None = None,
    atmos_vars: Mapping[str, str] | None = None,
    static_vars: Mapping[str, str] | None = None,
) -> Any:
    """Build an :class:`aurora.Batch` from a WeatherBench-shaped xarray Dataset.

    Args:
        ds: xarray Dataset with WeatherBench 2 variable names.
        times: ordered list of timestamps. Aurora uses the *last* one as ``T0``
            and earlier ones as history.
        region: optional ``(xmin, ymin, xmax, ymax)`` lon/lat bbox.
        surf_vars: WB2 -> Aurora surface variable mapping.
        atmos_vars: WB2 -> Aurora atmospheric variable mapping.
        static_vars: WB2 -> Aurora static variable mapping.

    Returns:
        An :class:`aurora.Batch` ready for the model.

    .. versionadded:: 0.8
    """
    aurora = lazy_import('aurora')
    surf = surf_vars if surf_vars is not None else DEFAULT_SURF_VARS
    atmos = atmos_vars if atmos_vars is not None else DEFAULT_ATMOS_VARS
    static = static_vars if static_vars is not None else DEFAULT_STATIC_VARS
    sliced = _slice(ds, [pd.Timestamp(t) for t in times], region)
    pick = lambda mp: {  # noqa: E731
        d: _tensor(sliced[s]).unsqueeze(0)
        for s, d in mp.items()
        if s in sliced.data_vars
    }
    return aurora.Batch(
        surf_vars=pick(surf),
        atmos_vars=pick(atmos),
        static_vars={
            d: _tensor(sliced[s]) for s, d in static.items() if s in sliced.data_vars
        },
        metadata=aurora.Metadata(
            lat=_tensor(sliced.latitude),
            lon=_tensor(sliced.longitude),
            time=(pd.Timestamp(sliced.time.values[-1]),),
            atmos_levels=tuple(int(v) for v in sliced.level.values)
            if 'level' in sliced.coords
            else (),
        ),
    )


class AuroraWeatherBench2Sequence(Dataset[dict[str, Any]]):
    """Aurora context/target windows over a WeatherBench 2 store.

    Each item is a ``{'context': aurora.Batch, 'target': aurora.Batch}`` pair.

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
        surf_vars: Mapping[str, str] | None = None,
        atmos_vars: Mapping[str, str] | None = None,
        static_vars: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize a new AuroraWeatherBench2Sequence instance.

        Args:
            dataset: a :class:`~torchgeo.datasets.WeatherBench2` instance.
            start_time: first window start (clipped to the store's range).
            end_time: exclusive end of the data range.
            region: ``(xmin, ymin, xmax, ymax)`` bbox (default: global).
            timestep: time between samples.
            context_steps: number of input timesteps.
            target_steps: number of supervision timesteps.
            surf_vars: WB2 -> Aurora surface variable mapping.
            atmos_vars: WB2 -> Aurora atmospheric variable mapping.
            static_vars: WB2 -> Aurora static variable mapping.

        Raises:
            ValueError: If steps are < 1 or no valid window fits.
        """
        super().__init__()
        if context_steps < 1 or target_steps < 1:
            raise ValueError('context_steps and target_steps must be >= 1.')
        self.dataset = dataset
        self.region = region
        self.context_steps = context_steps
        self.target_steps = target_steps
        self.timestep = pd.to_timedelta(timestep)
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.static_vars = static_vars

        tmin = pd.Timestamp(dataset.data.time.values.min())
        tmax = pd.Timestamp(dataset.data.time.values.max())
        window = (context_steps + target_steps - 1) * self.timestep
        first = max(pd.Timestamp(start_time), tmin)
        last = min(pd.Timestamp(end_time) - window, tmax - window)
        if last < first:
            raise ValueError(
                f'No window of {context_steps + target_steps} steps fits in '
                f'[{first}, {pd.Timestamp(end_time)}] given dataset range '
                f'[{tmin}, {tmax}].'
            )
        self.starts: list[pd.Timestamp] = list(
            pd.date_range(first, last, freq=self.timestep)
        )

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return ``{'context': Batch, 'target': Batch}`` for window *index*."""
        n = self.context_steps + self.target_steps
        times = pd.date_range(self.starts[index], periods=n, freq=self.timestep)
        return {
            'context': aurora_batch_from_xarray(
                self.dataset.data,
                times=times[: self.context_steps],
                region=self.region,
                surf_vars=self.surf_vars,
                atmos_vars=self.atmos_vars,
                static_vars=self.static_vars,
            ),
            'target': aurora_batch_from_xarray(
                self.dataset.data,
                times=times[self.context_steps :],
                region=self.region,
                surf_vars=self.surf_vars,
                atmos_vars=self.atmos_vars,
                static_vars={},
            ),
        }


def aurora_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of context/target Batch pairs along the batch dim.

    Args:
        batch: samples from :class:`AuroraWeatherBench2Sequence`.

    Returns:
        A dict with ``context`` and ``target`` :class:`aurora.Batch` items.

    .. versionadded:: 0.8
    """
    aurora = lazy_import('aurora')

    def merge(field: str) -> Any:
        first = batch[0][field]
        return aurora.Batch(
            surf_vars={
                k: torch.cat([b[field].surf_vars[k] for b in batch])
                for k in first.surf_vars
            },
            atmos_vars={
                k: torch.cat([b[field].atmos_vars[k] for b in batch])
                for k in first.atmos_vars
            },
            static_vars=dict(first.static_vars),
            metadata=aurora.Metadata(
                lat=first.metadata.lat,
                lon=first.metadata.lon,
                time=tuple(b[field].metadata.time[0] for b in batch),
                atmos_levels=first.metadata.atmos_levels,
            ),
        )

    return {'context': merge('context'), 'target': merge('target')}


def aurora_predictions_to_xarray(
    preds: Sequence[Any],
    init_time: str | pd.Timestamp,
    timestep: str | pd.Timedelta = '6h',
    surf_vars: Mapping[str, str] | None = None,
    atmos_vars: Mapping[str, str] | None = None,
) -> Any:
    """Pack Aurora rollout predictions into a WeatherBench-shaped xarray Dataset.

    The *i*-th prediction is tagged with ``init_time + (i + 1) * timestep``.

    Args:
        preds: sequence of :class:`aurora.Batch` predictions, one per rollout
            step (e.g. as collected from :func:`aurora.rollout`).
        init_time: forecast initialization time T0.
        timestep: time between rollout steps.
        surf_vars: WB2 -> Aurora surface variable mapping (``{}`` to skip).
        atmos_vars: WB2 -> Aurora atmospheric variable mapping (``{}`` to skip).

    Returns:
        An :class:`xarray.Dataset` ready for ``ds.to_zarr(path)``.

    Raises:
        ValueError: If *preds* is empty.

    .. versionadded:: 0.8
    """
    xr = lazy_import('xarray')
    if not preds:
        raise ValueError('preds must not be empty.')

    dt = pd.to_timedelta(timestep)
    init = pd.Timestamp(init_time)
    times = [init + (i + 1) * dt for i in range(len(preds))]
    surf = DEFAULT_SURF_VARS if surf_vars is None else surf_vars
    atmos = DEFAULT_ATMOS_VARS if atmos_vars is None else atmos_vars
    a2w_surf = {v: k for k, v in surf.items()}
    a2w_atmos = {v: k for k, v in atmos.items()}

    def stack(field: str, key: str) -> np.ndarray:
        return torch.stack([getattr(p, field)[key][0, 0] for p in preds]).cpu().numpy()

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for k in preds[0].surf_vars:
        if (wb2 := a2w_surf.get(k)) is not None:
            data_vars[wb2] = (('time', 'latitude', 'longitude'), stack('surf_vars', k))
    for k in preds[0].atmos_vars:
        if (wb2 := a2w_atmos.get(k)) is not None:
            data_vars[wb2] = (
                ('time', 'level', 'latitude', 'longitude'),
                stack('atmos_vars', k),
            )

    coords: dict[str, Any] = {
        'time': times,
        'latitude': preds[0].metadata.lat.detach().cpu().numpy(),
        'longitude': preds[0].metadata.lon.detach().cpu().numpy(),
    }
    if any('level' in dims for dims, _ in data_vars.values()):
        coords['level'] = list(preds[0].metadata.atmos_levels)
    return xr.Dataset(data_vars, coords=coords)


class WeatherBench2AuroraDataModule(NonGeoDataModule):
    """LightningDataModule for fine-tuning Aurora on WeatherBench 2 slices.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        region: tuple[float, float, float, float] | None = None,
        timestep: str | pd.Timedelta = '6h',
        context_steps: int = 2,
        target_steps: int = 1,
        surf_vars: Mapping[str, str] | None = None,
        atmos_vars: Mapping[str, str] | None = None,
        static_vars: Mapping[str, str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherBench2AuroraDataModule instance.

        Args:
            start_time: first window start (inclusive).
            end_time: exclusive end of the data range.
            region: ``(xmin, ymin, xmax, ymax)`` bbox (default: global).
            timestep: time between samples.
            context_steps: number of context (input) steps.
            target_steps: number of target (supervision) steps.
            surf_vars: WB2 -> Aurora surface variable mapping.
            atmos_vars: WB2 -> Aurora atmospheric variable mapping.
            static_vars: WB2 -> Aurora static variable mapping.
            batch_size: per-GPU mini-batch size.
            num_workers: dataloader workers.
            **kwargs: forwarded to :class:`~torchgeo.datasets.WeatherBench2`.
        """
        super().__init__(
            WeatherBench2,
            batch_size,
            num_workers,
            **kwargs,
        )
        self.start_time = start_time
        self.end_time = end_time
        self.region = region
        self.timestep = timestep
        self.context_steps = context_steps
        self.target_steps = target_steps
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.static_vars = static_vars
        self.collate_fn = aurora_collate_fn

    def setup(self, stage: str) -> None:
        """Build the dataset and reuse it for all splits.

        Args:
            stage: ``'fit'``, ``'validate'``, ``'test'`` or ``'predict'``.
        """
        wb2 = WeatherBench2(**self.kwargs)
        seq = AuroraWeatherBench2Sequence(
            wb2,
            start_time=self.start_time,
            end_time=self.end_time,
            region=self.region,
            timestep=self.timestep,
            context_steps=self.context_steps,
            target_steps=self.target_steps,
            surf_vars=self.surf_vars,
            atmos_vars=self.atmos_vars,
            static_vars=self.static_vars,
        )
        for split in ('train', 'val', 'test', 'predict'):
            setattr(self, f'{split}_dataset', seq)

    def on_after_batch_transfer(
        self, batch: dict[str, Any], dataloader_idx: int
    ) -> dict[str, Any]:
        return batch
