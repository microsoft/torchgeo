# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile
from pathlib import Path

from tests.data.utils import write_image

for path in [
    'test/images/0.tif',
    'test/images/1.tif',
    'test/images/2.tif',
    'test/images/3.tif',
    'train/part1/images/0.tif',
    'train/part1/images/1.tif',
    'train/part1/images/2.tif',
    'train/part1/images/3.tif',
    'train/part2/images/0.tif',
    'train/part2/images/1.tif',
    'train/part2/images/2.tif',
    'train/part2/images/3.tif',
    'validation/images/0.tif',
    'validation/images/1.tif',
    'validation/images/2.tif',
    'validation/images/3.tif',
]:
    write_image(
        path,
        {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 3,
            'height': 2,
            'width': 2,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            'compress': 'lzw',
        },
    )

Path('test').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'test/images0.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['images/0.tif', 'images/1.tif']:
        archive.write(Path('test') / member, member)

Path('test').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'test/images1.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['images/2.tif']:
        archive.write(Path('test') / member, member)

Path('test').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'test/images2.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['images/3.tif']:
        archive.write(Path('test') / member, member)

Path('train/part1').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'train/part1/images.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['images/2.tif', 'images/0.tif', 'images/3.tif', 'images/1.tif']:
        archive.write(Path('train/part1') / member, member)

Path('train/part1').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'train/part1/labelXml.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'labelXml/1.xml',
        'labelXml/0.xml',
        'labelXml/3.xml',
        'labelXml/2.xml',
    ]:
        archive.write(Path('train/part1') / member, member)

Path('train/part2').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'train/part2/images.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['images/2.tif', 'images/0.tif', 'images/3.tif', 'images/1.tif']:
        archive.write(Path('train/part2') / member, member)

Path('train/part2').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'train/part2/labelXmls.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'labelXml/1.xml',
        'labelXml/0.xml',
        'labelXml/3.xml',
        'labelXml/2.xml',
    ]:
        archive.write(Path('train/part2') / member, member)

Path('validation').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'validation/images.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['images/2.tif', 'images/0.tif', 'images/3.tif', 'images/1.tif']:
        archive.write(Path('validation') / member, member)

Path('validation').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    'validation/labelXmls.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'labelXml/1.xml',
        'labelXml/0.xml',
        'labelXml/3.xml',
        'labelXml/2.xml',
    ]:
        archive.write(Path('validation') / member, member)
