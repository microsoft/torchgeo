#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import hashlib
import json
import os
import shutil
import string
from collections.abc import Sequence

import numpy as np
import rasterio
from pyproj import CRS

# General hyperparams / speed up by choosing a small value (actual values will be 256)
IMG_SIZE = 32
DUMMY_DATA_SIZE = {'train': 10, 'test': 5}

# Directory structure
root_dir = '{0}/FLAIR2'
splits: Sequence[str] = ('train', 'test')
dir_names: dict[str, dict[str, str]] = {
    'train': {
        'img': 'flair_aerial_train',
        'sen': 'flair_sen_train',
        'msk': 'flair_labels_train',
    },
    'test': {
        'img': 'flair_2_aerial_test',
        'sen': 'flair_2_sen_test',
        'msk': 'flair_2_labels_test',
    },
}
dir_names_toy: dict[str, dict[str, str]] = {
    'train': {
        'img': 'flair_2_toy_aerial_train',
        'sen': 'flair_2_toy_sen_train',
        'msk': 'flair_2_toy_labels_train',
    },
    'test': {
        'img': 'flair_2_toy_aerial_test',
        'sen': 'flair_2_toy_sen_test',
        'msk': 'flair_2_toy_labels_test',
    },
}
# Replace with random digits and letters
sub_sub_dir_format = 'D{0}_{1}/Z{2}_{3}'


# Aerial specifics
aerial_all_bands: Sequence[str] = ('B01', 'B02', 'B03', 'B04', 'B05')
aerial_pixel_values: list[int] = list(range(256))
aerial_profile = {
    'dtype': np.uint8,
    'count': len(aerial_all_bands),
    'width': IMG_SIZE,
    'height': IMG_SIZE,
    'crs': CRS.from_epsg(2154),
    'driver': 'GTiff',
}
aerial_format = '.tif'

# Sentinel specifics
sentinel_format = ['.npy', '.npy', '.txt']
sentinel_pixel_values: list[int] = list(range(20000))
sentinel_name_format = 'SEN2_sp_D{0}_{1}_Z{2}_{3}'
SENTINEL_IMG_SIZE = [156, 207]


# Label specifics
# Theoretically the labels are between 1 and 18, although the flair2 description groups all pixels >13 into one class
labels_pixel_values: list[int] = list(range(1, 19))
labels_profile = {
    'dtype': np.byte,
    'count': 1,
    'width': IMG_SIZE,
    'height': IMG_SIZE,
    # In theory the date is broken at the moment but assume correct for now
    'crs': CRS.from_epsg(2154),
    'driver': 'GTiff',
}
labels_format = '.tif'

centroids_dict: dict[str, list[int]] = {}


def populate_sub_sub_dirs(
    dir_path: str, seed: int, type: str, domain_year_zone_location: Sequence[str]
) -> None:
    # Create random 6 digit number (might have leading zeros)
    rng: np.random.Generator = np.random.default_rng(seed)

    # Mimic 1:n mapping of aerial/mask to sentinel data
    file_id_0 = rng.integers(1000000)
    file_id_1 = rng.integers(1000000)
    id_str_0 = f'{file_id_0:06}'
    id_str_1 = f'{file_id_1:06}'
    match type:
        case 'img':
            create_aerial_image(dir_path, rng, id_str_0)
            create_aerial_image(dir_path, rng, id_str_1)
        case 'sen':
            create_sentinel_arrays(dir_path, rng, domain_year_zone_location)
        case 'msk':
            create_label_mask(dir_path, rng, id_str_0)
            create_label_mask(dir_path, rng, id_str_1)
        case _:
            raise ValueError(f'Unknown type: {type}')


def create_aerial_image(dir_path: str, rng: np.random.Generator, id: str) -> None:
    centroids_dict[f'IMG_{id}.tif'] = [IMG_SIZE // 2, IMG_SIZE // 2]
    with rasterio.open(
        os.path.join(dir_path, f'IMG_{id}{aerial_format}'), 'w', **aerial_profile
    ) as src:
        for i in range(len(aerial_all_bands)):
            data = rng.choice(
                aerial_pixel_values, size=(IMG_SIZE, IMG_SIZE), replace=True
            ).astype(np.uint8)
            src.write(data, i + 1)


def create_sentinel_arrays(
    dir_path: str, rng: np.random.Generator, domain_year_zone_location: Sequence[str]
) -> None:
    # Create random numpy array of shape (T, 10, H, W) and save it as .npy
    time_num_samples = rng.integers(1, 10)
    data = rng.choice(
        sentinel_pixel_values,
        size=(
            time_num_samples,
            10,
            rng.choice(SENTINEL_IMG_SIZE),
            rng.choice(SENTINEL_IMG_SIZE),
        ),
        replace=True,
    ).astype(np.uint8)
    snow_cloud_mask = rng.choice(
        [0, 100],
        size=(
            time_num_samples,
            rng.choice(SENTINEL_IMG_SIZE),
            rng.choice(SENTINEL_IMG_SIZE),
        ),
        replace=True,
    ).astype(np.uint8)

    name = sentinel_name_format.format(*domain_year_zone_location)
    np.save(os.path.join(dir_path, f'{name}_data{sentinel_format[0]}'), data)
    np.save(
        os.path.join(dir_path, f'{name}_masks{sentinel_format[1]}'), snow_cloud_mask
    )

    # Create products.txt
    with open(os.path.join(dir_path, f'{name}_products{sentinel_format[2]}'), 'w') as f:
        for _ in range(time_num_samples):
            f.write('S2A_MSIL2A_20210415T105021_N0300_R051_T31UDP_20210415T135921')


def create_label_mask(dir_path: str, rng: np.random.Generator, id: str) -> None:
    data = rng.choice(
        labels_pixel_values, size=(IMG_SIZE, IMG_SIZE), replace=True
    ).astype(np.byte)
    with rasterio.open(
        os.path.join(dir_path, f'MSK_{id}{labels_format}'), 'w', **labels_profile
    ) as src:
        src.write(data, 1)


def create_metadata(root_dir: str) -> None:
    # Create file flair-2_centroids_sp_to_patch.json
    with open(os.path.join(root_dir, 'flair-2_centroids_sp_to_patch.json'), 'w') as f:
        json.dump(centroids_dict, f)


if __name__ == '__main__':
    root_dir = root_dir.format(os.getcwd())

    # Remove the root directory if it exists
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    def create_dir_structure(
        root_dir: str, dir_names: dict[str, dict[str, str]]
    ) -> None:
        # Create the directory structure
        for split in splits:
            for i in range(DUMMY_DATA_SIZE[split]):
                # Reproducible and the same for all types
                seed = int(hashlib.md5(f'{split}{i}'.encode()).hexdigest(), 16)
                rng: np.random.Generator = np.random.default_rng(seed)

                random_domain = rng.integers(100, 1000)
                random_year = rng.integers(2010, 2023)
                random_zone = rng.integers(10, 23)
                random_area = ''.join(rng.choice(list(string.ascii_uppercase), size=2))
                for type, sub_dir in dir_names[split].items():
                    # E.g. D123_2021/Z1_UF
                    sub_sub_dir = sub_sub_dir_format.format(
                        random_domain, random_year, random_zone, random_area
                    )

                    # type adds last directory, one of: img, sen, msk
                    dir_path = os.path.join(root_dir, sub_dir, sub_sub_dir, type)
                    os.makedirs(dir_path, exist_ok=True)

                    # Required for sentinel data arrays (npy) and products.txt
                    domain_year_zone_location = [
                        str(random_domain),
                        str(random_year),
                        str(random_zone),
                        random_area,
                    ]
                    populate_sub_sub_dirs(
                        dir_path, seed, type, domain_year_zone_location
                    )

                    create_metadata(root_dir)

    root_dir_toy = os.path.join(root_dir, 'flair_2_toy_dataset')
    os.makedirs(root_dir_toy, exist_ok=True)
    create_dir_structure(root_dir, dir_names)
    create_dir_structure(root_dir_toy, dir_names_toy)

    # zip each directory/file in root_dir
    for element in glob.glob(f'{root_dir}/**'):
        print(element)
        if os.path.isdir(element):
            # Make archive with element name as root directory
            shutil.make_archive(
                element, 'zip', os.path.dirname(element), os.path.basename(element)
            )
        else:
            shutil.make_archive(
                element.removesuffix('.json'),
                'zip',
                os.path.dirname(element),
                os.path.basename(element),
            )

    # for toy, zip the entire root_dir_toy
    shutil.make_archive(
        root_dir_toy,
        'zip',
        os.path.dirname(root_dir_toy),
        os.path.basename(root_dir_toy),
    )

    # Rename flair-2_centroids_sp_to_patch.json to flair_2_centroids_sp_to_patch.json to replicate
    # the inconsistency from the actual flair2 dataset
    old_metadata_path = os.path.join(root_dir, 'flair-2_centroids_sp_to_patch.zip')
    new_metadata_path = os.path.join(root_dir, 'flair_2_centroids_sp_to_patch.zip')
    if os.path.exists(old_metadata_path):
        os.rename(old_metadata_path, new_metadata_path)

    # Compute md5 for each zip file
    with open(os.path.join(root_dir, 'md5s.txt'), 'w') as md5_file:
        for element in glob.glob(f'{root_dir}/**'):
            if element.endswith('.zip'):
                filename = os.path.basename(element)
                with open(element, 'rb') as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
                    md5_file.write(f'{filename}: {md5}\n')
