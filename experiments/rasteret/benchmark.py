#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Benchmark first-batch latency: TorchGeo ``RasterDataset`` vs ``RasteretDataset``.

Reproduces one row of the PR benchmark table per invocation. Each run is a fresh
process, so torchgeo's GDAL/``vsicurl`` cache starts empty and it opens every COG
for the first time -- the real per-epoch cost.

Fair comparison: a full temporal-range query is used so *both* datasets read every
overlapping scene. (A ``GeoSampler``'s instant temporal slice would make date-aware
Rasteret select fewer scenes than date-blind ``RasterDataset``, inflating the ratio.)
Rasteret's ``Collection`` is prebuilt; the build is a one-time step, excluded from the
timings -- Rasteret ``init`` below is just loading the prebuilt index.

Examples::

    # same-cloud: Earth Search (sentinel-cogs S3)
    python benchmark.py --source earthsearch --mode spatial --sampler random

    # cross-cloud: Planetary Computer (Azure COGs)
    python benchmark.py --source pc --mode timeseries --sampler grid
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import rasteret
import torch
from pyproj import CRS
from pystac_client import Client

from torchgeo.datasets import RasterDataset, RasteretDataset
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

# Per-source STAC endpoint, band asset key, and whether hrefs need PC signing.
SOURCES = {
    'earthsearch': {
        'dataset': 'earthsearch/sentinel-2-l2a',
        'stac': 'https://earth-search.aws.element84.com/v1',
        'asset': 'red',
        'sign': False,
        'gdal': {'AWS_NO_SIGN_REQUEST': 'YES'},
    },
    'pc': {
        'dataset': 'pc/sentinel-2-l2a',
        'stac': 'https://planetarycomputer.microsoft.com/api/stac/v1',
        'asset': 'B04',
        'sign': True,
        'gdal': {},
    },
}
GDAL_ENV = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'GDAL_MAX_RAW_BLOCK_CACHE_SIZE': '200000000',
    'VSI_CURL_CACHE_SIZE': '200000000',
}


@contextmanager
def gdal_env(extra: dict[str, str]) -> Iterator[None]:
    """Apply GDAL read tuning (and any per-source vars) for the torchgeo read."""
    env = {**GDAL_ENV, **extra}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def full_range_batch(
    ds: RasterDataset, locations: list[tuple], chip: int, res: float
) -> tuple[torch.Tensor, int]:
    """Read each (x, y) location over the dataset's full temporal range.

    Querying the full range makes both datasets select every overlapping scene,
    so the comparison is over equal work. Returns the batch and the scene count.
    """
    _, _, t = ds.bounds
    ts = slice(t.start, t.stop, None)
    scenes = ds.index.iloc[
        ds.index.index.overlaps(pd.Interval(t.start, t.stop, closed='both'))
    ].shape[0]
    batch = torch.stack([ds[x, y, ts]['image'] for x, y in locations])
    return batch, scenes


def main() -> None:
    """Run one (source, mode, sampler) benchmark row and print it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=list(SOURCES), default='earthsearch')
    parser.add_argument('--mode', choices=['spatial', 'timeseries'], default='spatial')
    parser.add_argument('--sampler', choices=['random', 'grid'], default='random')
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('W', 'S', 'E', 'N'),
        default=(77.30, 12.35, 77.38, 12.43),  # one MGRS tile (43PGP) near Bangalore
    )
    parser.add_argument('--date-range', default='2023-01-01/2023-06-30')
    parser.add_argument('--band', default='B04')
    parser.add_argument('--chip', type=int, default=256)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--cloud-lt', type=int, default=100)
    parser.add_argument('--workspace', default=str(Path.home() / 'rasteret_workspace'))
    args = parser.parse_args()

    src = SOURCES[args.source]
    bbox = tuple(args.bbox)
    time_series = args.mode == 'timeseries'

    # Prebuild (or load) the Rasteret collection. Build time is NOT measured.
    tag = args.date_range.replace('/', '_')
    ws = Path(args.workspace).expanduser() / f'{args.source}_{tag}'
    ws.mkdir(parents=True, exist_ok=True)
    if not glob.glob(f'{ws}/*_stac'):
        rasteret.build(
            src['dataset'],
            name='bench',
            bbox=bbox,
            date_range=tuple(args.date_range.split('/')),
            workspace_dir=ws,
        )
    collection = rasteret.load(glob.glob(f'{ws}/*_stac')[0]).subset(
        cloud_cover_lt=args.cloud_lt
    )

    # torchgeo scene paths from the same STAC query (PC hrefs signed via pystac).
    modifier = None
    if src['sign']:
        import planetary_computer

        modifier = planetary_computer.sign_inplace
    items = list(
        Client.open(src['stac'], modifier=modifier)
        .search(
            collections=['sentinel-2-l2a'],
            bbox=list(bbox),
            datetime=args.date_range,
            query={'eo:cloud_cover': {'lt': args.cloud_lt}},
        )
        .items()
    )
    if time_series:
        # Keep one scene per date so date-blind RasterDataset stacks the same number
        # of timesteps as datetime-aware Rasteret.
        by_date: dict[str, object] = {}
        for item in items:
            by_date.setdefault(item.properties['datetime'][:10], item)
        items = list(by_date.values())
    paths = [f'/vsicurl/{item.assets[src["asset"]].href}' for item in items]

    # Rasteret derives the output CRS/resolution from the prebuilt index (no raster
    # is opened); reuse them so torchgeo reads onto the identical grid.
    t0 = time.perf_counter()
    rasteret_ds = RasteretDataset(
        collection=collection, bands=[args.band], time_series=time_series
    )
    rasteret_init = (time.perf_counter() - t0) * 1000
    crs: CRS = rasteret_ds.crs
    res = rasteret_ds.res

    if args.sampler == 'grid':
        sampler = GridGeoSampler(rasteret_ds, size=args.chip, stride=args.chip // 2)
    else:
        sampler = RandomGeoSampler(rasteret_ds, size=args.chip, length=args.batch)
    locations = [(q[0], q[1]) for q in sampler][: args.batch]

    t0 = time.perf_counter()
    rasteret_batch, scenes = full_range_batch(rasteret_ds, locations, args.chip, res)
    rasteret_read = (time.perf_counter() - t0) * 1000

    with gdal_env(src['gdal']):
        t0 = time.perf_counter()
        torchgeo_ds = RasterDataset(
            paths=paths, crs=crs, res=res, time_series=time_series
        )
        torchgeo_init = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        torchgeo_batch, _ = full_range_batch(torchgeo_ds, locations, args.chip, res)
        torchgeo_read = (time.perf_counter() - t0) * 1000

    speedup = (torchgeo_init + torchgeo_read) / (rasteret_init + rasteret_read)
    match = tuple(rasteret_batch.shape) == tuple(torchgeo_batch.shape)
    print(
        f'{args.source} {args.mode}/{args.sampler}: {scenes} scenes '
        f'shape={tuple(rasteret_batch.shape)} match={match}'
    )
    print(
        f'  Rasteret: {rasteret_init:.0f} ms init + {rasteret_read:.0f} ms first batch'
    )
    print(
        f'  TorchGeo: {torchgeo_init:.0f} ms init + {torchgeo_read:.0f} ms first batch'
    )
    print(f'  Speedup (init + first batch): {speedup:.1f}x')


if __name__ == '__main__':
    main()
