#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import tarfile

import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = (100, 100)
TRAIN_CITIES = ['aachen', 'bergisch', 'bielefeld']
TEST_CITIES = ['duesseldorf']
CLASSES = [
    'background',
    'forest',
    'water',
    'agricultural',
    'residential,commercial,industrial',
    'grassland,swamp,shrubbery',
    'railway,trainstation',
    'highway,squares',
    'airport,shipyard',
    'roads',
    'buildings',
]
NUM_SAMPLES_PER_CITY = 2


def create_directories(cities: list[str]) -> None:
    for city in cities:
        if os.path.exists(city):
            shutil.rmtree(city)
        os.makedirs(city, exist_ok=True)


def generate_dummy_data(cities: list[str]) -> None:
    for city in cities:
        for i in range(NUM_SAMPLES_PER_CITY):
            utm_coords = f'{i}_{i}'
            rgb_image = np.random.randint(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
            dem_image = np.random.randint(0, 256, IMAGE_SIZE, dtype=np.uint8)
            seg_image = np.random.randint(0, len(CLASSES), IMAGE_SIZE, dtype=np.uint8)

            Image.fromarray(rgb_image).save(os.path.join(city, f'{utm_coords}_rgb.jp2'))
            Image.fromarray(dem_image).save(os.path.join(city, f'{utm_coords}_dem.tif'))
            Image.fromarray(seg_image).save(os.path.join(city, f'{utm_coords}_seg.tif'))


def create_tarball(output_filename: str, source_dirs: list[str]) -> None:
    with tarfile.open(output_filename, 'w:gz') as tar:
        for source_dir in source_dirs:
            tar.add(source_dir, arcname=os.path.basename(source_dir))


def calculate_md5(filename: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Main function
def main() -> None:
    train_cities = TRAIN_CITIES
    test_cities = TEST_CITIES

    create_directories(train_cities)
    create_directories(test_cities)

    generate_dummy_data(train_cities)
    generate_dummy_data(test_cities)

    tarball_name = 'nrw_dataset.tar.gz'
    create_tarball(tarball_name, train_cities + test_cities)

    md5sum = calculate_md5(tarball_name)
    print(f'MD5 checksum: {md5sum}')


if __name__ == '__main__':
    main()
