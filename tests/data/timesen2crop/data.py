#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Generate a tiny synthetic TimeSen2Crop archive for testing.

Mirrors the real Zenodo release layout:

* top-level folder ``TimeSen2Crop/``
* per-tile subfolder ``<tile>/`` containing ``dates.csv``
  (with a leading ``acquisition_date`` header) and integer-named class
  subfolders
* per-sample CSV with header ``B1,B2,...,B9,Flag``
"""

import os
import shutil
import zipfile

import numpy as np

NUM_BANDS = 9
TILES = {'33TUN': 5, '2019_33UVP': 7}
CLASSES = (0, 1)
SAMPLES_PER_CLASS = 2
HEADER = ','.join([f'B{i + 1}' for i in range(NUM_BANDS)] + ['Flag'])
DATES_HEADER = 'acquisition_date'

np.random.seed(0)


def write_sample(path: str, num_steps: int) -> None:
    bands = np.random.randint(0, 4000, size=(num_steps, NUM_BANDS), dtype=np.int32)
    condition = np.random.randint(0, 4, size=(num_steps, 1), dtype=np.int32)
    arr = np.concatenate([bands, condition], axis=1)
    np.savetxt(path, arr, delimiter=',', fmt='%d', header=HEADER, comments='')


def write_dates(path: str, num_steps: int) -> None:
    dates = [f'2017{(9 + i) % 12 + 1:02d}{(i % 28) + 1:02d}' for i in range(num_steps)]
    with open(path, 'w') as f:
        f.write(DATES_HEADER + '\n')
        f.write('\n'.join(dates))


if __name__ == '__main__':
    root = 'timesen2crop'
    extracted = os.path.join(root, 'TimeSen2Crop')

    if os.path.isdir(extracted):
        shutil.rmtree(extracted)
    # Also clean up the previous test layout that used ``Dataset/``.
    legacy = os.path.join(root, 'Dataset')
    if os.path.isdir(legacy):
        shutil.rmtree(legacy)
    archive = os.path.join(root, 'TimeSen2Crop.zip')
    if os.path.exists(archive):
        os.remove(archive)
    cache = os.path.join(root, 'cache')
    if os.path.isdir(cache):
        shutil.rmtree(cache)

    for tile, T in TILES.items():
        tile_dir = os.path.join(extracted, tile)
        os.makedirs(tile_dir, exist_ok=True)
        write_dates(os.path.join(tile_dir, 'dates.csv'), T)
        for class_id in CLASSES:
            class_dir = os.path.join(tile_dir, str(class_id))
            os.makedirs(class_dir, exist_ok=True)
            for i in range(SAMPLES_PER_CLASS):
                write_sample(os.path.join(class_dir, f'{i}.csv'), T)

    # Build the zip with fixed timestamps so the file is byte-stable across
    # runs and the committed md5 in the test stays valid.
    zip_path = os.path.join(root, 'TimeSen2Crop.zip')
    fixed_time = (2021, 1, 1, 0, 0, 0)
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(extracted):
        for name in filenames:
            paths.append(os.path.join(dirpath, name))
    paths.sort()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for full in paths:
            arcname = os.path.relpath(full, root)
            info = zipfile.ZipInfo(arcname, date_time=fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(full, 'rb') as src:
                zf.writestr(info, src.read())

    shutil.rmtree(extracted)
