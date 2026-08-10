#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Script for generating test data for the BioMassters dataset."""

import hashlib
import os
import shutil
import tarfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

SIZE = 4  # pixels, kept tiny so the fixtures stay small

FEATURE_DTYPE = 'uint16'
TARGET_DTYPE = 'float32'

np.random.seed(0)

ROOT = os.path.dirname(os.path.abspath(__file__))

# chip ids used for each split
CHIPS = {'train': ['0000a', '0000b'], 'test': ['0001a']}

# months (0-indexed, first month is September) for which fake imagery is
# generated -- both sensors share the same months here so that every chip has
# a fully aligned (S1, S2) pair for at least one month
MONTHS = [0, 1]


def write_tif(path: str, num_bands: int, dtype: str) -> None:
    """Write a tiny, valid, single-chip GeoTIFF."""
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': num_bands,
        'height': SIZE,
        'width': SIZE,
        'crs': 'EPSG:4326',
        'transform': Affine(0.0001, 0.0, 0.0, 0.0, -0.0001, 0.0),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        for band in range(1, num_bands + 1):
            data = np.random.randint(0, 100, size=(SIZE, SIZE)).astype(dtype)
            dst.write(data, band)


def build_features(
    features_dir: str, split: str, rows: list[dict[str, object]]
) -> None:
    """Generate feature GeoTIFFs for ``split`` and append metadata rows."""
    os.makedirs(features_dir, exist_ok=True)
    for chip_id in CHIPS[split]:
        for satellite, num_bands in (('S1', 2), ('S2', 3)):
            for month in MONTHS:
                filename = f'{chip_id}_{satellite}_{month:02d}.tif'
                write_tif(
                    os.path.join(features_dir, filename),
                    num_bands=num_bands,
                    dtype=FEATURE_DTYPE,
                )
                rows.append(
                    {
                        'chip_id': chip_id,
                        'filename': filename,
                        'satellite': satellite,
                        'split': split,
                        'month': month,
                        'corresponding_agbm': f'{chip_id}_agbm.tif',
                    }
                )


def build_targets(agbm_dir: str) -> None:
    """Generate AGBM target GeoTIFFs for the train split."""
    os.makedirs(agbm_dir, exist_ok=True)
    for chip_id in CHIPS['train']:
        write_tif(
            os.path.join(agbm_dir, f'{chip_id}_agbm.tif'),
            num_bands=1,
            dtype=TARGET_DTYPE,
        )


def make_tar_gz(archive_path: str, source_dir: str, arcname: str) -> None:
    """Create a ``.tar.gz`` archive containing ``source_dir`` as ``arcname``."""
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(source_dir, arcname=arcname)


def split_file(path: str, num_parts: int) -> None:
    """Split ``path`` into ``num_parts`` sequential ``<path>aa``, ``<path>ab``, ... parts.

    Concatenating the parts back together in order (as BioMassters._extract does)
    reproduces the original file exactly.
    """
    with open(path, 'rb') as f:
        data = f.read()
    chunk_size = max(1, -(-len(data) // num_parts))  # ceil division
    suffixes = [f'a{chr(ord("a") + i)}' for i in range(num_parts)]
    starts = range(0, len(data), chunk_size)
    for suffix, start in zip(suffixes, starts, strict=False):
        chunk = data[start : start + chunk_size]
        with open(f'{path}{suffix}', 'wb') as out:
            out.write(chunk)
    os.remove(path)


def main() -> None:
    work_dir = os.path.join(ROOT, '_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    rows: list[dict[str, object]] = []
    train_features_dir = os.path.join(work_dir, 'train_features')
    test_features_dir = os.path.join(work_dir, 'test_features')
    agbm_dir = os.path.join(work_dir, 'train_agbm')

    build_features(train_features_dir, 'train', rows)
    build_features(test_features_dir, 'test', rows)
    build_targets(agbm_dir)

    # metadata csv, columns match what BioMassters.__init__ / __getitem__ expect
    metadata = pd.DataFrame(
        rows,
        columns=[
            'chip_id',
            'filename',
            'satellite',
            'split',
            'month',
            'corresponding_agbm',
        ],
    )
    metadata_path = os.path.join(ROOT, 'biomassters_features_metadata.csv')
    metadata.to_csv(metadata_path, index=False)

    # train_features.tar.gz -> split into 4 parts (aa, ab, ac, ad)
    train_tar = os.path.join(ROOT, 'train_features.tar.gz')
    make_tar_gz(train_tar, train_features_dir, arcname='train_features')
    split_file(train_tar, num_parts=4)

    # test_features.tar.gz -> split into 2 parts (aa, ab)
    test_tar = os.path.join(ROOT, 'test_features.tar.gz')
    make_tar_gz(test_tar, test_features_dir, arcname='test_features')
    split_file(test_tar, num_parts=2)

    # train_agbm.tar.gz, not split
    agbm_tar = os.path.join(ROOT, 'train_agbm.tar.gz')
    make_tar_gz(agbm_tar, agbm_dir, arcname='train_agbm')

    shutil.rmtree(work_dir)

    # Print sha256 checksums of the generated fixtures for reference. These are
    # *not* the real upstream BioMassters.checksums (which are pinned to the
    # real Hugging Face archives); tests that exercise checksum handling
    # monkeypatch BioMassters.checksums to match these fixture files instead.
    generated_filenames = [
        'train_features.tar.gzaa',
        'train_features.tar.gzab',
        'train_features.tar.gzac',
        'train_features.tar.gzad',
        'test_features.tar.gzaa',
        'test_features.tar.gzab',
        'train_agbm.tar.gz',
        'biomassters_features_metadata.csv',
    ]
    for filename in generated_filenames:
        path = os.path.join(ROOT, filename)
        with open(path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        print(f'{filename}: {digest}')


if __name__ == '__main__':
    main()
