#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


def generate_test_data(root: str) -> None:
    """Generate fake VisDrone data."""
    for split in ('train', 'val', 'test-dev'):
        directory = os.path.join(root, f'VisDrone2019-DET-{split}')
        image_directory = os.path.join(directory, 'images')
        annotation_directory = os.path.join(directory, 'annotations')
        os.makedirs(image_directory, exist_ok=True)
        os.makedirs(annotation_directory, exist_ok=True)

        array = np.zeros((8, 8, 3), dtype=np.uint8)
        Image.fromarray(array).save(os.path.join(image_directory, '000001.jpg'))
        with open(os.path.join(annotation_directory, '000001.txt'), 'w') as file:
            file.write('1,2,3,4,1,1,0,0\n')
            file.write('0,0,1,1,0,1,0,0\n')
            file.write('0,0,1,1,1,11,0,0\n')


if __name__ == '__main__':
    generate_test_data(os.path.dirname(__file__))
