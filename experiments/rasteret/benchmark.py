#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Benchmark first-batch latency for native TorchGeo vs Rasteret-backed TorchGeo.

Examples::

    python benchmark.py --mode both --sampler both
    python benchmark.py --mode spatial
    python benchmark.py --mode timeseries --sampler random
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(os.environ.get('RASTERET_WORKSPACE', str(Path.home() / 'rasteret_workspace'))).expanduser()
DEFAULT_COLLECTION = 'bangalore'
GDAL_ENV = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'AWS_NO_SIGN_REQUEST': 'YES',
    'GDAL_MAX_RAW_BLOCK_CACHE_SIZE': '200000000',
    'GDAL_SWATH_SIZE': '200000000',
    'VSI_CURL_CACHE_SIZE': '200000000',
}
BUILD_PRESETS: dict[str, dict[str, Any]] = {
    'bangalore': {
        'dataset': 'earthsearch/sentinel-2-l2a',
        'name': 'bangalore',
        'bbox': (77.55, 13.01, 77.58, 13.08),
        'date_range': ('2024-01-01', '2024-06-30'),
    }
}


def resolve_collection(path: str, workspace: Path) -> str | None:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return str(candidate) if candidate.exists() else None


def ensure_collection(path: str, workspace: Path) -> str | None:
    preset = BUILD_PRESETS.get(path)
    if preset is not None:
        import rasteret
        collection = rasteret.build(
            preset['dataset'],
            name=preset['name'],
            bbox=preset['bbox'],
            date_range=preset['date_range'],
            workspace_dir=workspace,
        )
        built_path = workspace / f'{collection.name}_stac'
        if built_path.exists():
            return str(built_path)
        return resolve_collection(collection.name, workspace)

    resolved = resolve_collection(path, workspace)
    if resolved is not None:
        return resolved
    return None


def env_meta() -> dict[str, str]:
    meta = {'timestamp': datetime.now(UTC).isoformat(), 'os': platform.system(), 'arch': platform.machine(), 'python': platform.python_version()}
    for name in ('rasteret', 'torchgeo', 'rasterio'):
        try:
            module = __import__(name)
        except ImportError:
            continue
        meta[name] = getattr(module, '__version__', '?')
    return meta


def scene_paths(collection: Any, band: str) -> list[str]:
    from rasteret.constants import BandRegistry

    mapping = BandRegistry.get(collection.data_source)
    asset_key = mapping.get(band, band) if isinstance(mapping, dict) else band
    assets = collection.dataset.to_table(columns=['assets']).column('assets').to_pylist()
    paths = [
        f"/vsicurl/{asset['href']}"
        for row in assets
        if isinstance(row, dict)
        for asset in [row.get(asset_key) or row.get(band)]
        if isinstance(asset, dict) and isinstance(asset.get('href'), str)
    ]
    if not paths:
        raise ValueError(f'no URLs for band {band}')
    return paths


def target_epsg(collection: Any) -> int:
    values = collection.dataset.to_table(columns=['proj:epsg']).column('proj:epsg').to_pylist()
    epsgs = sorted({int(value) for value in values if value is not None})
    if not epsgs:
        raise ValueError('collection has no valid proj:epsg values')
    if len(epsgs) != 1:
        raise ValueError(
            'benchmark requires a single-CRS collection; '
            f'found EPSG values {epsgs}. Pick a single-zone collection.'
        )
    return epsgs[0]


def target_resolution(collection: Any, band: str) -> tuple[float, float]:
    from rasteret.core.utils import normalize_transform

    column = f'{band}_metadata'
    values = collection.dataset.to_table(columns=[column]).column(column).to_pylist()
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            scale_x, _, scale_y, _ = normalize_transform(value.get('transform'))
        except (TypeError, ValueError):
            continue
        return abs(float(scale_x)), abs(float(scale_y))
    raise ValueError(f'collection has no valid transform metadata for band {band}')


def same_resolution(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return math.isclose(left[0], right[0], rel_tol=1e-9, abs_tol=1e-12) and math.isclose(
        left[1], right[1], rel_tol=1e-9, abs_tol=1e-12
    )


@contextmanager
def gdal_env() -> Any:
    old = {key: os.environ.get(key) for key in GDAL_ENV}
    os.environ.update(GDAL_ENV)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_batch(ds: Any, sampler_name: str, chip_size: int, batch_size: int) -> tuple[Any, list[int]]:
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples
    from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

    if sampler_name == 'random':
        sampler = RandomGeoSampler(ds, size=chip_size, length=batch_size)
    else:
        sampler = GridGeoSampler(ds, size=chip_size, stride=max(chip_size // 2, 1))

    batch = next(iter(DataLoader(ds, sampler=sampler, batch_size=batch_size, num_workers=0, collate_fn=stack_samples)))
    key = 'image' if 'image' in batch else 'mask'
    return batch, [int(v) for v in batch[key].shape]


def time_first_batch(ds: Any, sampler_name: str, chip_size: int, batch_size: int) -> tuple[float, list[int]]:
    t0 = time.perf_counter()
    _batch, shape = make_batch(ds, sampler_name, chip_size, batch_size)
    return (time.perf_counter() - t0) * 1000, shape


def benchmark_pair(
    collection_path: str,
    band: str,
    sampler_name: str,
    chip_size: int,
    batch_size: int,
    time_series: bool,
) -> dict[str, Any]:
    import rasteret
    from pyproj import CRS
    from torchgeo.datasets import RasterDataset, RasteretDataset

    collection = rasteret.load(collection_path)
    if band not in set(collection.bands):
        raise ValueError(f'band {band} not found in {Path(collection_path).name}')

    crs = CRS.from_epsg(target_epsg(collection))
    res = target_resolution(collection, band)
    mode = 'timeseries' if time_series else 'spatial'
    print(
        f'{mode}/{sampler_name}: {Path(collection_path).name} '
        f'(band={band}, res={res}, chip={chip_size}, batch={batch_size})'
    )

    t0 = time.perf_counter()
    rasteret_ds = RasteretDataset(
        collection=collection,
        bands=[band],
        crs=crs,
        res=res,
        time_series=time_series,
    )
    rasteret_init = (time.perf_counter() - t0) * 1000
    rasteret_res = tuple(float(value) for value in rasteret_ds.res)
    if not same_resolution(rasteret_res, res):
        raise ValueError(f'RasteretDataset used {rasteret_res}, expected {res}')
    try:
        rasteret_read, rasteret_shape = time_first_batch(rasteret_ds, sampler_name, chip_size, batch_size)
    finally:
        close = getattr(rasteret_ds, 'close', None)
        if callable(close):
            close()

    with gdal_env():
        t0 = time.perf_counter()
        torchgeo_ds = RasterDataset(
            paths=scene_paths(collection, band),
            crs=crs,
            res=res,
            time_series=time_series,
        )
        torchgeo_init = (time.perf_counter() - t0) * 1000
        torchgeo_res = tuple(float(value) for value in torchgeo_ds.res)
        if not same_resolution(torchgeo_res, res):
            raise ValueError(f'RasterDataset used {torchgeo_res}, expected {res}')
        torchgeo_read, torchgeo_shape = time_first_batch(torchgeo_ds, sampler_name, chip_size, batch_size)

    if rasteret_shape != torchgeo_shape:
        raise ValueError(
            'benchmark produced different sample shapes: '
            f'Rasteret={rasteret_shape}, TorchGeo={torchgeo_shape}'
        )

    speedup_read = torchgeo_read / rasteret_read
    speedup_total = (torchgeo_init + torchgeo_read) / (rasteret_init + rasteret_read)

    print(f'  Rasteret: init={rasteret_init:.0f}ms first_batch={rasteret_read:.0f}ms shape={rasteret_shape}')
    print(f'  TorchGeo: init={torchgeo_init:.0f}ms first_batch={torchgeo_read:.0f}ms shape={torchgeo_shape}')
    print(f'  speedup: first_batch={speedup_read:.1f}x total={speedup_total:.1f}x')

    return {
        'collection': Path(collection_path).name,
        'band': band,
        'sampler': sampler_name,
        'time_series': time_series,
        'chip_size': chip_size,
        'batch_size': batch_size,
        'resolution': [round(res[0], 12), round(res[1], 12)],
        'rasteret': {'init_ms': round(rasteret_init, 1), 'first_batch_ms': round(rasteret_read, 1), 'sample_shape': rasteret_shape},
        'torchgeo_native': {'init_ms': round(torchgeo_init, 1), 'first_batch_ms': round(torchgeo_read, 1), 'sample_shape': torchgeo_shape},
        'speedup_first_batch': round(speedup_read, 2),
        'speedup_total': round(speedup_total, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--collection',
        default=DEFAULT_COLLECTION,
        help='Benchmark preset, collection path, or workspace-relative collection name.',
    )
    parser.add_argument('--mode', choices=['spatial', 'timeseries', 'both'], default='both')
    parser.add_argument('--sampler', choices=['random', 'grid', 'both'], default='both')
    parser.add_argument('--band', default='B04')
    parser.add_argument('--chip-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workspace', type=str, default=str(WORKSPACE))
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    collection_path = ensure_collection(args.collection, Path(args.workspace).expanduser())
    if collection_path is None:
        parser.error(f'collection not found: {args.collection}')

    meta = env_meta()
    print(f"environment: python={meta.get('python', '?')} torchgeo={meta.get('torchgeo', '?')} rasteret={meta.get('rasteret', '?')} rasterio={meta.get('rasterio', '?')}")

    samplers = ['random', 'grid'] if args.sampler == 'both' else [args.sampler]
    modes = [False, True] if args.mode == 'both' else [args.mode == 'timeseries']
    results: dict[str, Any] = {'environment': meta, 'runs': {}}
    for time_series in modes:
        for sampler_name in samplers:
            key = f'{"timeseries" if time_series else "spatial"}_{sampler_name}'
            results['runs'][key] = benchmark_pair(
                collection_path,
                args.band,
                sampler_name,
                args.chip_size,
                args.batch_size,
                time_series,
            )

    print('\nsummary:')
    for key, run in results['runs'].items():
        print(f"  {key}: first_batch={run['speedup_first_batch']:.1f}x total={run['speedup_total']:.1f}x")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f'saved: {args.output}')


if __name__ == '__main__':
    main()
