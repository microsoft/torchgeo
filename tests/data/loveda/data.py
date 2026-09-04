# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    'Test/Rural/images_png/01.png',
    'Test/Urban/images_png/01.png',
    'Train/Rural/images_png/01.png',
    'Train/Urban/images_png/01.png',
    'Val/Rural/images_png/01.png',
    'Val/Urban/images_png/01.png',
]:
    write_image(
        path,
        {
            'driver': 'PNG',
            'dtype': 'uint8',
            'count': 3,
            'height': 2,
            'width': 2,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

for path in [
    'Train/Rural/masks_png/01.png',
    'Train/Urban/masks_png/01.png',
    'Val/Rural/masks_png/01.png',
    'Val/Urban/masks_png/01.png',
]:
    write_image(
        path,
        {
            'driver': 'PNG',
            'dtype': 'uint8',
            'count': 1,
            'height': 2,
            'width': 2,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

with zipfile.ZipFile('Test.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for member in ['Test/Rural/images_png/01.png', 'Test/Urban/images_png/01.png']:
        archive.write(member, member)

with zipfile.ZipFile('Train.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for member in [
        'Train/Urban/images_png/01.png',
        'Train/Urban/masks_png/01.png',
        'Train/Rural/images_png/01.png',
        'Train/Rural/masks_png/01.png',
    ]:
        archive.write(member, member)

with zipfile.ZipFile('Val.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for member in [
        'Val/Rural/images_png/01.png',
        'Val/Rural/masks_png/01.png',
        'Val/Urban/images_png/01.png',
        'Val/Urban/masks_png/01.png',
    ]:
        archive.write(member, member)
