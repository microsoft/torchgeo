# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import tarfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def generate_data(
    path: str, filename: str, height: int, width: int, include_hydro: bool = False
) -> None:
    max_value = 1000.0
    min_value = 0.0
    interval = max_value - min_value
    folders = ['s1_raw', 'DEM', 'mask', 'hydro']
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'crs': CRS.from_epsg(4326),
        'transform': Affine(
            0.0001287974837883981,
            0.0,
            14.438064999669106,
            0.0,
            -8.989523639880024e-05,
            45.71617928533084,
        ),
        'blockysize': 1,
        'tiled': False,
        'interleave': 'pixel',
        'height': height,
        'width': width,
    }
    data = {
        's1_raw': np.random.rand(2, height, width).astype(np.float32) * interval
        - min_value,
        'DEM': np.random.rand(1, height, width).astype(np.float32) * interval
        - min_value,
        'mask': np.random.randint(low=0, high=2, size=(1, height, width)).astype(
            np.uint8
        ),
    }

    if include_hydro:
        data['hydro'] = (
            np.random.rand(1, height, width).astype(np.float32) * interval - min_value
        )

    for folder in folders:
        folder_path = os.path.join(path, folder)
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, filename)
        profile2 = profile.copy()
        profile2['count'] = 2 if folder == 's1_raw' else 1
        if folder in data:
            with rasterio.open(filepath, mode='w', **profile2) as src:
                src.write(data[folder])


def generate_tar_gz(src: str, dst: str) -> None:
    with tarfile.open(dst, 'w:gz') as tar:
        tar.add(src, arcname=src)


def split_tar(path: str, dst: str, nparts: int) -> None:
    fstats = os.stat(path)
    size = fstats.st_size
    chunk = size // nparts

    with open(path, 'rb') as fp:
        for idx in range(nparts):
            part_path = os.path.join(dst, f'activations.tar.{idx:03}.gz.part')

            bytes_to_write = chunk if idx < nparts - 1 else size - fp.tell()
            with open(part_path, 'wb') as dst_fp:
                dst_fp.write(fp.read(bytes_to_write))


def generate_folders_and_metadata(datapath: str, metadatapath: str) -> None:
    folders_splits = [
        ('EMSR000', 'train'),
        ('EMSR001', 'train'),
        ('EMSR003', 'val'),
        ('EMSR004', 'test'),
    ]
    num_files = {'EMSR000': 3, 'EMSR001': 2, 'EMSR003': 2, 'EMSR004': 1}
    num_hydro = {'EMSR001': 2, 'EMSR003': 1, 'EMSR004': 1}
    metadata = {}
    for folder, split in folders_splits:
        data = {}
        data['title'] = 'Test flood'
        data['type'] = 'Flood'
        data['country'] = 'N/A'
        data['start'] = '2014-11-06T17:57:00'
        data['end'] = '2015-01-29T12:47:04'
        data['lat'] = 45.82427031690563
        data['lon'] = 14.484407562009336
        data['subset'] = split
        data['delineations'] = [f'{folder}_00']

        count_hydro = 0

        dst_folder = os.path.join(datapath, f'{folder}-0')
        for idx in range(num_files[folder]):
            include_hydro = count_hydro < num_hydro.get(folder, 0)
            generate_data(
                dst_folder,
                filename=f'{folder}-{idx}.tif',
                height=16,
                width=16,
                include_hydro=include_hydro,
            )
            if include_hydro:
                count_hydro += 1

        metadata[folder] = data

    generate_tar_gz(src='activations', dst='activations.tar.gz')
    split_tar(path='activations.tar.gz', dst='.', nparts=2)
    os.remove('activations.tar.gz')
    with open(os.path.join(metadatapath, 'activations.json'), 'w') as fp:
        json.dump(metadata, fp)


if __name__ == '__main__':
    datapath = os.path.join(os.getcwd(), 'activations')
    metadatapath = os.getcwd()

    generate_folders_and_metadata(datapath, metadatapath)
