#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine

np.random.seed(0)

SIZE = 200
ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_PROFILE = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'nodata': 0,
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'blockxsize': SIZE,
    'blockysize': SIZE,
    'tiled': False,
    'compress': 'lzw',
    'interleave': 'band',
}

# One entry per sample: the 3rd sample contains a plume, the others do not.
# The same samples are used for the train, val and test splits.
SAMPLES = [
    {
        'id_location': '3bc442gz-d44gh',
        'id_loc_image': '00x42hg65xy-d6734g',
        'crs': 'EPSG:32732',
        'origin': (812690.0, 9420810.0),
        'wind_u': -1.04,
        'wind_v': 3.445676,
        'vza': 7.034637327,
        'sza': 22.78845,
        'percentage_clear': 99.8,
        'tile': 'S2A_MSIL1C_2029-12-17',
        'isplume': False,
        'ch4_fluxrate': 0.0,
        'ch4_fluxrate_std': 0.0,
        'satellite': 'Torchgeo',
        'tile_date': '2029-12-17 04:50:51+00:00',
        'last_update': '2030-12-05T02:27:58+00:00',
        'location_name': 'WP1',
        'country': 'somewhere',
        'lon': 27.3,
        'lat': -2.55,
        'offshore': True,
        'background_image_tile': np.nan,
        'window': (np.nan, np.nan, np.nan, np.nan),
        'footprint': 'POLYGON ((27.307916 -2.543850, 27.290062 -2.543360, 27.292052 -2.561353, 27.308780 -2.556554, 27.307916 -2.543850))',
        'plume': 'MULTIPOLYGON EMPTY',
        'case_study': 'Offshore',
        'wind_source': 'trust',
    },
    {
        'id_location': '7ka913dq-p20xe',
        'id_loc_image': '51r84tz90mb-q2819w',
        'crs': 'EPSG:32641',
        'origin': (331830.0, 4434010.0),
        'wind_u': 2.31,
        'wind_v': -1.874523,
        'vza': 6.512903884,
        'sza': 38.40172,
        'percentage_clear': 97.4,
        'tile': 'S2A_MSIL1C_2029-06-03',
        'isplume': False,
        'ch4_fluxrate': 0.0,
        'ch4_fluxrate_std': 0.0,
        'satellite': 'Torchgeo',
        'tile_date': '2029-06-03 11:42:09+00:00',
        'last_update': '2030-02-14T08:15:42+00:00',
        'location_name': 'WP2',
        'country': 'nowhere',
        'lon': 48.7,
        'lat': 31.9,
        'offshore': False,
        'background_image_tile': 'S2B_MSIL1C_2029-05-29',
        'window': (np.nan, np.nan, np.nan, np.nan),
        'footprint': 'POLYGON ((48.708091 31.908000, 48.692053 31.906425, 48.695192 31.888509, 48.710978 31.890734, 48.708091 31.908000))',
        'plume': 'MULTIPOLYGON EMPTY',
        'case_study': 'Nowhere',
        'wind_source': 'trust',
    },
    {
        'id_location': '9mq205vh-t77cz',
        'id_loc_image': '28d03kf71ws-h5036n',
        'crs': 'EPSG:32632',
        'origin': (575940.0, 3108180.0),
        'wind_u': -4.1,
        'wind_v': 6.093417,
        'vza': 4.870221155,
        'sza': 27.65913,
        'percentage_clear': 100.0,
        'tile': 'S2B_MSIL1C_2029-09-21',
        'isplume': True,
        'ch4_fluxrate': 5120.75,
        'ch4_fluxrate_std': 2210.4,
        'satellite': 'Torchgeo',
        'tile_date': '2029-09-21 09:27:33+00:00',
        'last_update': '2030-03-02T16:05:11+00:00',
        'location_name': 'WP3',
        'country': 'elsewhere',
        'lon': 15.2,
        'lat': 12.4,
        'offshore': False,
        'background_image_tile': 'S2B_MSIL1C_2029-09-11',
        'window': (62.0, 44.0, 38.0, 57.0),
        'footprint': 'POLYGON ((15.209672 12.407864, 15.190221 12.407307, 15.188077 12.392049, 15.207945 12.392623, 15.209672 12.407864))',
        'plume': 'MULTIPOLYGON (((15.208788 12.400000, 15.208065 12.401355, 15.206309 12.402575, 15.203318 12.403094, 15.201391 12.402956, 15.200307 12.402498, 15.198470 12.402823, 15.195681 12.403128, 15.193405 12.402811, 15.193382 12.401992, 15.195591 12.400760, 15.196545 12.399244, 15.195473 12.398032, 15.195531 12.397337, 15.195855 12.396943, 15.198020 12.396933, 15.200559 12.397269, 15.201995 12.397343, 15.202025 12.397338, 15.204013 12.397517, 15.205951 12.398528, 15.208788 12.400000)))',
        'case_study': 'Elsewhere',
        'wind_source': 'trust',
    },
]

COLUMNS = [
    's2path',
    'plumepath',
    'cloudmaskpath',
    'ch4path',
    'wind_u',
    'wind_v',
    'vza',
    'sza',
    'percentage_clear',
    'tile',
    'isplume',
    'ch4_fluxrate',
    'ch4_fluxrate_std',
    'satellite',
    'tile_date',
    'notified',
    'id_location',
    'last_update',
    'location_name',
    'country',
    'lon',
    'lat',
    'offshore',
    'sector',
    'observability',
    'background_image_tile',
    'crs',
    'transform_a',
    'transform_b',
    'transform_c',
    'transform_d',
    'transform_e',
    'transform_f',
    'width',
    'height',
    'window_row_off',
    'window_col_off',
    'window_height',
    'window_width',
    'footprint',
    'plume',
    'split_name',
    'case_study',
    'wind_source',
    'id_loc_image',
]

num_samples = {'train': 3, 'val': 3, 'test': 3}


def profile_for(sample: dict, dtype: str, count: int) -> dict:
    profile = BASE_PROFILE.copy()
    profile['dtype'] = dtype
    profile['count'] = count
    profile['crs'] = sample['crs']
    x, y = sample['origin']
    profile['transform'] = Affine(10.0, 0.0, x, 0.0, -10.0, y)
    if count > 1:
        profile['interleave'] = 'pixel'
    return profile


def write(fn: str, profile: dict, data: np.ndarray) -> None:
    with rasterio.open(os.path.join(ROOT, fn), 'w', **profile) as dst:
        dst.write(data)


def create_s2(fn: str, sample: dict) -> None:
    # 6 bands of the plume scene followed by 6 bands of the reference scene
    profile = profile_for(sample, 'uint16', 12)
    data = np.random.randint(
        np.iinfo(np.uint16).max, size=(12, SIZE, SIZE), dtype=np.uint16
    )
    write(fn, profile, data)


def create_mask(fn: str, sample: dict) -> None:
    # plume mask and cloud mask
    profile = profile_for(sample, 'uint8', 1)
    data = np.random.randint(2, size=(1, SIZE, SIZE), dtype=np.uint8)
    write(fn, profile, data)


def create_ch4(fn: str, sample: dict) -> None:
    profile = profile_for(sample, 'float64', 1)
    data = np.random.uniform(0, 1e20, size=(1, SIZE, SIZE)).astype(np.float64)
    write(fn, profile, data)


def create_row(sample: dict, split: str) -> dict:
    prefix = f'{sample["id_location"]}_{sample["id_loc_image"]}'
    row_off, col_off, height, width = sample['window']
    x, y = sample['origin']
    return {
        's2path': f'{prefix}_s2.tif',
        'plumepath': f'{prefix}_label.tif' if sample['isplume'] else np.nan,
        'cloudmaskpath': f'{prefix}_cloudmask.tif',
        'ch4path': f'{prefix}_ch4.tif' if sample['isplume'] else np.nan,
        'wind_u': sample['wind_u'],
        'wind_v': sample['wind_v'],
        'vza': sample['vza'],
        'sza': sample['sza'],
        'percentage_clear': sample['percentage_clear'],
        'tile': sample['tile'],
        'isplume': sample['isplume'],
        'ch4_fluxrate': sample['ch4_fluxrate'],
        'ch4_fluxrate_std': sample['ch4_fluxrate_std'],
        'satellite': sample['satellite'],
        'tile_date': sample['tile_date'],
        'notified': False,
        'id_location': sample['id_location'],
        'last_update': sample['last_update'],
        'location_name': sample['location_name'],
        'country': sample['country'],
        'lon': sample['lon'],
        'lat': sample['lat'],
        'offshore': sample['offshore'],
        'sector': 'Oil and Gas',
        'observability': 'clear',
        'background_image_tile': sample['background_image_tile'],
        'crs': sample['crs'],
        'transform_a': 10.0,
        'transform_b': 0.0,
        'transform_c': x,
        'transform_d': 0.0,
        'transform_e': -10.0,
        'transform_f': y,
        'width': SIZE,
        'height': SIZE,
        'window_row_off': row_off,
        'window_col_off': col_off,
        'window_height': height,
        'window_width': width,
        'footprint': sample['footprint'],
        'plume': sample['plume'],
        'split_name': 'train_2023',
        'case_study': sample['case_study'],
        'wind_source': sample['wind_source'],
        'id_loc_image': sample['id_loc_image'],
    }


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)

    # Create the rasters, shared between all splits
    for sample in SAMPLES:
        prefix = f'{sample["id_location"]}_{sample["id_loc_image"]}'
        create_s2(f'{prefix}_s2.tif', sample)
        create_mask(f'{prefix}_cloudmask.tif', sample)
        if sample['isplume']:
            create_mask(f'{prefix}_label.tif', sample)
            create_ch4(f'{prefix}_ch4.tif', sample)

    # Create the split manifests
    for split, n in num_samples.items():
        rows = [create_row(sample, split) for sample in SAMPLES[:n]]
        df = pd.DataFrame(rows, columns=COLUMNS)
        df.to_csv(os.path.join(ROOT, f'{split}.csv'), index=False)
