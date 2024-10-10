#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
from copy import deepcopy
from datetime import datetime, timedelta

import h5py
import numpy as np

meta_dummy_dict = {
    'S2_DATE': '2018-07-16',
    'S2_type': 'l1c',
    'CRS': 'EPSG:32721',
    'lat': -14.499441524746077,
    'lon': -56.98355999998649,
}

num_tiles = 10

meta_id_strings = [str(i) for i in range(num_tiles)]

modalities = {
    'aster': {'bands': 2, 'dtype': np.int16},
    'biome': {'bands': 14, 'dtype': np.uint8},
    'canopy_height_eth': {'bands': 2, 'dtype': np.int8},
    'dynamic_world': {'bands': 1, 'dtype': np.uint8},
    'eco_region': {'bands': 846, 'dtype': np.uint16},
    'era5': {'bands': 12, 'dtype': np.float32},
    'esa_worldcover': {'bands': 1, 'dtype': np.uint8},
    'sentinel1': {'bands': 8, 'dtype': np.float32},
    'sentinel2': {'bands': 13, 'dtype': np.uint16},
    'sentinel2_cloudmask': {'bands': 1, 'dtype': np.uint16},
    'sentinel2_cloudprod': {'bands': 1, 'dtype': np.uint16},
    'sentinel2_scl': {'bands': 1, 'dtype': np.uint16},
}

all_modality_bands = {
    'sentinel2': [
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8A',
        'B8',
        'B9',
        'B10',
        'B11',
        'B12',
    ],
    'sentinel2_cloudmask': ['QA60'],
    'sentinel2_cloudprod': ['MSK_CLDPRB'],
    'sentinel2_scl': ['SCL'],
    'sentinel1_asc': ['VV', 'VH', 'HH', 'HV'],
    'sentinel1_desc': ['VV', 'VH', 'HH', 'HV'],
    'aster': ['b1', 'slope'],  # elevation and slope
    'era5': [
        'prev_temperature_2m',  # previous month avg temp
        'prev_temperature_2m_min',  # previous month min temp
        'prev_temperature_2m_max',  # previous month max temp
        'prev_total_precipitation_sum',  # previous month total precip
        'curr_temperature_2m',  # current month avg temp
        'curr_temperature_2m_min',  # current month min temp
        'curr_temperature_2m_max',  # current month max temp
        'curr_total_precipitation_sum',  # current month total precip
        '0_temperature_2m_mean',  # year avg temp
        '1_temperature_2m_min_min',  # year min temp
        '2_temperature_2m_max_max',  # year max temp
        '3_total_precipitation_sum_sum',  # year total precip
    ],
    'dynamic_world': ['label'],
    'canopy_height_eth': ['height', 'std'],
    'lat': ['sin', 'cos'],
    'lon': ['sin', 'cos'],
    'biome': ['biome'],
    'eco_region': ['eco_region'],
    'month': ['sin_month', 'cos_month'],
    'esa_worldcover': ['Map'],
}


def create_hd5f(dataset_name: str, px_dim: tuple[int]) -> list[dict[str, str]]:
    # Create the HDF5 file
    with h5py.File(f'{dataset_name}.h5', 'w') as h5file:
        # Create datasets for each modality
        for modality, modal_info in modalities.items():
            bands = modal_info['bands']
            if modality in ['era5', 'eco_region', 'biome']:
                h5file.create_dataset(
                    modality, (num_tiles, bands), dtype=modal_info['dtype']
                )
            else:
                h5file.create_dataset(
                    modality, (num_tiles, bands, *px_dim), dtype=modal_info['dtype']
                )

        # Create datasets for metadata
        h5file.create_dataset('lat', (num_tiles, 2), dtype=np.float32)
        h5file.create_dataset('lon', (num_tiles, 2), dtype=np.float32)
        h5file.create_dataset('month', (num_tiles, 2), dtype=np.int32)
        h5file.create_dataset(
            'metadata',
            (num_tiles,),
            dtype=np.dtype([('meta_id', 'S10'), ('S2_type', 'S3')]),
        )

        # Populate the datasets with sample data
        tile_info = {}
        for i in range(num_tiles):
            for modality in modalities:
                if modality == 'dynamic_world':
                    old_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    data = np.random.choice(old_values, size=(bands, *px_dim))
                elif modality == 'esa_worldcover':
                    old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
                    data = np.random.choice(old_values, size=(bands, *px_dim))
                elif modality == 'era5':
                    # only vector not image data
                    data = np.random.random(size=(bands,))
                elif modality in ['biome', 'eco_region']:
                    data = np.random.randint(0, 2, size=(bands,))
                elif modality == 'sentinel2':
                    data = np.random.randint(0, 65535, size=(bands, *px_dim))
                elif modality in ['aster', 'canopy_height_eth', 'sentinel1']:
                    data = np.random.random(size=(bands, *px_dim))
                elif modality in [
                    'sentinel2_cloudmask',
                    'sentinel2_cloudprod',
                    'sentinel2_scl',
                ]:
                    data = np.random.randint(0, 2, size=(bands, *px_dim))

                data = data.astype(modal_info['dtype'])
                h5file[modality][i] = data

            # add other data for lat, lon, month
            h5file['lat'][i] = np.random.random(size=(2,))
            h5file['lon'][i] = np.random.random(size=(2,))
            h5file['month'][i] = np.random.random(size=(2,))

            # Assign S2_type and store in metadata
            S2_type = np.random.choice(['l1c', 'l2a']).encode('utf-8')
            meta_id = str(i).encode('utf-8')
            h5file['metadata'][i] = (meta_id, S2_type)

            # Collect tile info for JSON file
            tile_meta = meta_dummy_dict.copy()

            tile_meta['S2_type'] = S2_type.decode('utf-8')
            # in all_Modality_bands era5 contains the data instead `prev` and `curr` prefixes
            date_str = tile_meta['S2_DATE']
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            curr_month_str = date_obj.strftime('%Y%m')
            prev_month_obj = date_obj.replace(day=1) - timedelta(days=1)
            prev_month_str = prev_month_obj.strftime('%Y%m')
            curr_sample_bands = deepcopy(all_modality_bands)
            curr_sample_bands['era5'] = [
                b.replace('curr', curr_month_str).replace('prev', prev_month_str)
                for b in curr_sample_bands['era5']
            ]
            tile_meta['BANDS'] = curr_sample_bands
            tile_info[str(i)] = tile_meta

    return tile_info


extra_band_stats = {
    'sentinel2_l1c': {'bands': 13, 'dtype': np.uint16},
    'sentinel2_l2a': {'bands': 13, 'dtype': np.uint16},
    'lat': {'bands': 2, 'dtype': np.float32},
    'lon': {'bands': 2, 'dtype': np.float32},
    'month': {'bands': 2, 'dtype': np.float32},
}

band_modalities = {
    k: v
    for k, v in {**modalities, **extra_band_stats}.items()
    if k not in {'biome', 'eco_region', 'dynamic_world', 'esa_worldcover'}
}

# Create JSON files for band stats and splits
# sentinel 2 has l1c and l2a but there is only a common sentinel 2 data entry
band_stats = {
    modality: {
        'mean': np.random.random(size=(mod_info['bands'])).tolist(),
        'std': np.random.random(size=(mod_info['bands'])).tolist(),
        'min': np.random.random(size=(mod_info['bands'])).tolist(),
        'max': np.random.random(size=(mod_info['bands'])).tolist(),
    }
    for modality, mod_info in band_modalities.items()
}

train_split = num_tiles
val_split = 0
test_split = 0

splits = {
    'train': list(range(train_split)),
    'val': list(range(train_split, train_split + val_split)),
    'test': list(range(train_split + val_split, num_tiles)),
}

if __name__ == '__main__':
    filenames = {
        'MMEarth': {'dirname': 'data_1M_v001', 'px_dim': (128, 128)},
        'MMEarth64': {'dirname': 'data_1M_v001_64', 'px_dim': (64, 64)},
        'MMEarth100k': {'dirname': 'data_100k_v001', 'px_dim': (128, 128)},
    }
    for key, vals in filenames.items():
        dirname = vals['dirname']
        # remove existing files
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

        # create directory
        os.makedirs(dirname)
        tile_info = create_hd5f(os.path.join(dirname, dirname), vals['px_dim'])

        print(f'{key} data file and JSON files created successfully.')

        with open(os.path.join(dirname, f'{dirname}_splits.json'), 'w') as f:
            json.dump(splits, f, indent=4)

        with open(os.path.join(dirname, f'{dirname}_band_stats.json'), 'w') as f:
            json.dump(band_stats, f, indent=4)

        with open(os.path.join(dirname, f'{dirname}_tile_info.json'), 'w') as f:
            json.dump(tile_info, f, indent=4)
